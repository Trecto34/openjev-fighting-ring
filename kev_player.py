"""Tetris player using jaredpalmer/kev-0.5b (Jev-inspired decision model).

Architecture:
- Base: Qwen/Qwen2.5-0.5B (frozen)
- LoRA adapter: rank 16 on all 24 attention and MLP projections
- PointerHead: 2 linear maps 896 -> 256, scaled dot-product between <decide> and </opt> hidden states
- Block-causal branch mask allowing single-pass multi-option decision evaluation
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import snapshot_download

from tetris_env import Tetris, compute_board_features
from tetris_prompts import build_pair, get_move_descriptors, PROMPT_STRATEGIES
from tetris_player import _candidate_moves, _unique_pairs, play_episode

SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]
_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


def user_tokens(tok, text: str) -> List[int]:
    """Tokenize text without producing reserved control tokens."""
    return tok(_SPECIAL_RE.sub(r"<¦\1¦>", str(text)), add_special_tokens=False).input_ids


class PointerHead(nn.Module):
    def __init__(self, d: int, dp: int = 256, dtype=torch.float32):
        super().__init__()
        self.q = nn.Linear(d, dp, dtype=dtype)
        self.k = nn.Linear(d, dp, dtype=dtype)
        self.scale = 1.0 / math.sqrt(dp)

    def forward(self, h_decide: torch.Tensor, h_opts: torch.Tensor) -> torch.Tensor:
        # h_decide: [d], h_opts: [K, d] -> logits [K]
        return (self.k(h_opts) @ self.q(h_decide)) * self.scale


def branch_mask_batch(segs: List[List[int]], device: str = "cpu", dtype=torch.float32) -> torch.Tensor:
    """Block-causal attention mask: attend(i, j) iff j <= i and (seg[j] == 0 or seg[j] == seg[i])."""
    L = max(len(s) for s in segs)
    s = torch.full((len(segs), L), -1, device=device)
    for b, seg in enumerate(segs):
        s[b, :len(seg)] = torch.tensor(seg, device=device)
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))
    same = (s[:, None, :] == s[:, :, None]) | (s[:, None, :] == 0)
    valid_key = (s != -1)[:, None, :]
    allow = (causal[None] & same & valid_key) | torch.eye(L, dtype=torch.bool, device=device)[None]
    return torch.zeros(len(segs), L, L, dtype=dtype, device=device).masked_fill(~allow, torch.finfo(dtype).min)[:, None]


def encode_record(tok, rec: Dict, max_state: int = 384, max_branch: int = 4096) -> Dict:
    """Pack record: [<state> ...] then per-question [<q> instr <opt> o </opt> ... <decide>]."""
    state_tokens = user_tokens(tok, rec["state"])
    S = [tok.convert_tokens_to_ids(SPECIAL[0])] + state_tokens[: max_state - 1]
    ids = list(S)
    seg = [0] * len(S)
    pos = list(range(len(S)))

    q_id, o_id, c_id, d_id = (tok.convert_tokens_to_ids(t) for t in SPECIAL[1:])
    decide_idx, opt_idx = [], []

    for k, q in enumerate(rec["questions"], start=1):
        br = [q_id] + user_tokens(tok, q["instr"])
        oi = []
        for o in q["options"]:
            br += [o_id] + user_tokens(tok, o) + [c_id]
            oi.append(len(br) - 1)
        br.append(d_id)
        base = len(ids)
        ids += br
        seg += [k] * len(br)
        pos += list(range(len(S), len(S) + len(br)))
        decide_idx.append(base + len(br) - 1)
        opt_idx.append([base + i for i in oi])

    return {
        "ids": ids,
        "seg": seg,
        "pos": pos,
        "decide_idx": decide_idx,
        "opt_idx": opt_idx
    }


MNLI_CRITERIA = [
    "contradiction: The hypothesis contradicts the premise",
    "entailment: The hypothesis follows from the premise",
    "neutral: The hypothesis may or may not be true given the premise"
]


class KevPlayer:
    """Tetris controller powered by jaredpalmer/kev-0.5b decision model."""

    display_name = "kev-0.5b Decision Model (CPU)"

    def __init__(
        self,
        repo_id: str = "jaredpalmer/kev-0.5b",
        mode: str = "choice",
        strategy: str = "qualitative",
        top_k: Optional[int] = 6,
        device: Optional[str] = None,
        num_threads: int = 6
    ):
        self.repo_id = repo_id
        self.mode = mode.lower()  # "choice" or "nli"
        self.strategy = strategy
        self.top_k = top_k
        self.device = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.display_name = f"kev-0.5b Decision Model ({'CUDA GPU (FP16)' if 'cuda' in self.device else 'CPU'})"
        if self.device == "cpu" and num_threads > 0:
            torch.set_num_threads(num_threads)

        print(f"[*] Loading Kev decision model from '{repo_id}' on {self.device} ({self.dtype})...")
        t0 = time.time()
        try:
            kev_dir = snapshot_download(repo_id, local_files_only=True)
        except Exception:
            kev_dir = snapshot_download(repo_id)
        meta = torch.load(os.path.join(kev_dir, "head.pt"), map_location="cpu")
        self.base_name = meta.get("base", "Qwen/Qwen2.5-0.5B")

        self.tok = AutoTokenizer.from_pretrained(self.base_name)
        attn_impl = "sdpa" if hasattr(F, "scaled_dot_product_attention") else "eager"
        base_lm = AutoModelForCausalLM.from_pretrained(
            self.base_name,
            dtype=self.dtype,
            attn_implementation=attn_impl
        ).model

        self.lm = PeftModel.from_pretrained(base_lm, kev_dir).to(self.device, dtype=self.dtype)
        self.head = PointerHead(self.lm.config.hidden_size, dp=256, dtype=self.dtype).to(self.device)
        self.head.load_state_dict({k: v.to(self.device, dtype=self.dtype) for k, v in meta["head"].items()})
        self.lm.eval()
        self.head.eval()

        self._last_latents: Optional[np.ndarray] = None
        self._last_best_idx: int = 0
        self._last_info: Dict[str, Any] = {}

        # Warmup forward pass on GPU
        if self.device.startswith("cuda"):
            try:
                rec_w = {"state": "Tetris", "questions": [{"instr": "warmup", "options": ["opt1", "opt2"]}]}
                enc_w = encode_record(self.tok, rec_w)
                ids_w = torch.tensor([enc_w["ids"]], device=self.device)
                pos_w = torch.tensor([enc_w["pos"]], device=self.device)
                mask_w = branch_mask_batch([enc_w["seg"]], device=self.device, dtype=self.dtype)
                opt_w = torch.tensor(enc_w["opt_idx"][0], device=self.device)
                with torch.inference_mode():
                    h_w = self.lm(input_ids=ids_w, position_ids=pos_w, attention_mask=mask_w).last_hidden_state[0]
                    _ = self.head(h_w[enc_w["decide_idx"][0]], h_w[opt_w])
                    torch.cuda.synchronize()
            except Exception:
                pass

        print(f"[+] Loaded kev-0.5b (mode={self.mode}) in {time.time() - t0:.2f}s on {self.device}")

    @property
    def encoder(self):
        return self

    def latents(self, pairs=None) -> np.ndarray:
        """Returns hidden states of evaluated moves for telemetry visualization."""
        if self._last_latents is not None:
            return self._last_latents
        return np.zeros((1, 48), dtype=np.float32)

    def _score_choice(self, env: Tetris, candidate_moves: List[Tuple[int, int]]) -> Tuple[List[float], np.ndarray, np.ndarray]:
        """Direct Choice formulation: all legal candidates are options of one question."""
        feats = env.state()["features"]
        opts = []
        for rot, col in candidate_moves:
            desc = get_move_descriptors(env, rot, col)
            sim = desc["sim_feats"]
            opts.append(
                f"Placement (rot={rot}, col={col}): clears {desc['lines']} lines, "
                f"{desc['delta_holes']} holes, landing height {desc['landing_h']}, "
                f"bumpiness {sim['bumpiness']}, max height {sim['max_height']}"
            )

        state_str = (
            f"Game: Tetris. Active piece: {env.current_piece}. Next: {env.next_piece}. "
            f"Current board height: {feats['max_height']}/20, holes: {feats['holes']}, bumpiness: {feats['bumpiness']}."
        )

        rec = {
            "state": state_str,
            "questions": [{
                "instr": "Choose the optimal Tetris move. Prioritize line clears, low landing height, flat surface, and zero trapped holes.",
                "options": opts
            }]
        }

        enc = encode_record(self.tok, rec)
        ids = torch.tensor([enc["ids"]], device=self.device)
        pos = torch.tensor([enc["pos"]], device=self.device)
        mask = branch_mask_batch([enc["seg"]], device=self.device, dtype=self.dtype)

        with torch.inference_mode():
            h = self.lm(input_ids=ids, position_ids=pos, attention_mask=mask).last_hidden_state[0]
            opt_tensors = torch.tensor(enc["opt_idx"][0], device=self.device)
            logits = self.head(h[enc["decide_idx"][0]], h[opt_tensors])
            probs = F.softmax(logits.float(), dim=-1).cpu().numpy()
            opt_latents = h[opt_tensors].float().cpu().numpy()

        scores = probs.tolist()
        # Shape into [CON, ENT, NEU] proxy for compatible UI bar rendering
        # High choice prob -> High ENT, Low CON
        cand_nli = []
        for p in probs:
            p_ent = float(p)
            p_con = float(max(0.0, 1.0 - p_ent) * 0.7)
            p_neu = float(max(0.0, 1.0 - p_ent - p_con))
            cand_nli.append([p_con, p_ent, p_neu])

        return scores, np.asarray(cand_nli, dtype=np.float32), opt_latents

    def _score_nli(self, env: Tetris, candidate_moves: List[Tuple[int, int]]) -> Tuple[List[float], np.ndarray, np.ndarray]:
        """NLI formulation: each candidate is an entailment question evaluated in one packed sequence."""
        questions = []
        for rot, col in candidate_moves:
            desc = get_move_descriptors(env, rot, col)
            premise, hyp = build_pair(desc, strategy=self.strategy)
            questions.append({
                "instr": f'Hypothesis: "{hyp}" How does it relate to the outcome premise? {premise}',
                "options": MNLI_CRITERIA
            })

        rec = {
            "state": "Tetris Board Evaluation: assessing candidate tetromino placements for safety and line clearance.",
            "questions": questions
        }

        enc = encode_record(self.tok, rec)
        ids = torch.tensor([enc["ids"]], device=self.device)
        pos = torch.tensor([enc["pos"]], device=self.device)
        mask = branch_mask_batch([enc["seg"]], device=self.device, dtype=self.dtype)

        with torch.inference_mode():
            h = self.lm(input_ids=ids, position_ids=pos, attention_mask=mask).last_hidden_state[0]
            cand_probs = []
            latents_list = []
            for d_idx, o_idx in zip(enc["decide_idx"], enc["opt_idx"]):
                opt_t = torch.tensor(o_idx, device=self.device)
                lg = self.head(h[d_idx], h[opt_t])
                pb = F.softmax(lg.float(), dim=-1).cpu().numpy()  # [con, ent, neu]
                cand_probs.append(pb)
                latents_list.append(h[d_idx].float().cpu().numpy())

        cand_probs = np.asarray(cand_probs, dtype=np.float32)  # [N, 3] -> [CON, ENT, NEU]
        scores = (cand_probs[:, 1] - cand_probs[:, 0]).tolist()
        return scores, cand_probs, np.asarray(latents_list, dtype=np.float32)

    def score_moves(self, env: Tetris, candidate_moves: List[Tuple[int, int]]) -> Tuple[List[float], np.ndarray]:
        if self.mode == "nli":
            scores, probs, latents = self._score_nli(env, candidate_moves)
        else:
            scores, probs, latents = self._score_choice(env, candidate_moves)
        self._last_latents = latents
        return scores, probs

    def choose_move(self, env: Tetris) -> Tuple[Tuple[int, int], Dict]:
        legal_moves = env.get_legal_moves()
        if not legal_moves:
            return (0, 0), {}

        candidate_moves = _candidate_moves(env, legal_moves, self.top_k)

        t0 = time.perf_counter()
        scores, probs = self.score_moves(env, candidate_moves)
        lat_ms = (time.perf_counter() - t0) * 1000

        best_idx = int(np.argmax(scores))
        best_move = candidate_moves[best_idx]
        self._last_best_idx = best_idx

        # Calculate choice confidence
        k_options = len(candidate_moves)
        max_prob = float(scores[best_idx]) if self.mode == "choice" else float(probs[best_idx, 1])
        conf = (max_prob - 1.0 / k_options) / (1.0 - 1.0 / k_options) if k_options > 1 else 1.0
        conf = float(np.clip(conf, 0.0, 1.0))

        info = {
            "lat_ms": lat_ms,
            "best_score": float(scores[best_idx]),
            "best_p_ent": float(probs[best_idx, 1]),
            "best_p_con": float(probs[best_idx, 0]),
            "best_p_neu": float(probs[best_idx, 2]),
            "confidence": conf,
            "n_candidates": len(candidate_moves),
            "total_legal": len(legal_moves),
            "candidates": [list(m) for m in candidate_moves],
            "cand_scores": [float(x) for x in scores],
            "cand_probs": [[float(v) for v in row] for row in probs],
            "mode": f"kev-0.5b ({self.mode})"
        }
        if self._last_latents is not None and len(self._last_latents) > best_idx:
            info["best_latent"] = self._last_latents[best_idx].tolist()

        return best_move, info


def main():
    parser = argparse.ArgumentParser(description="kev-0.5b Tetris Player")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--render", action="store_true", help="Render ANSI animated game")
    parser.add_argument("--delay", type=float, default=0.05, help="Render delay per step in seconds")
    parser.add_argument("--top-k", type=int, default=6, help="Candidate moves pre-filtered per piece (0 for all)")
    parser.add_argument("--mode", default="choice", choices=["choice", "nli"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()

    env = Tetris(seed=args.seed, max_steps=args.max_steps)
    player = KevPlayer(
        mode=args.mode,
        top_k=None if args.top_k <= 0 else args.top_k,
        num_threads=args.threads
    )

    print(f"\n--- Starting kev-0.5b evaluation ({args.episodes} episode(s), mode={args.mode}) ---")
    for ep in range(args.episodes):
        print(f"\nEpisode {ep + 1}/{args.episodes}:")
        env.reset()
        play_episode(env, player, render=args.render, delay=args.delay)


if __name__ == "__main__":
    main()
