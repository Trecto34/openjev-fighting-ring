"""Tetris player using convaiinnovations/laya (RLCD Decision Model).

Architecture:
- Backbone: ModernBERT-large (395M bidirectional encoder)
- Decision Head: 2-layer TransformerEncoder + option marker scorer + act/escalate head (421M total params)
- Optimization: Trained with Reinforcement Learning for Calibrated Decisions (RLCD)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from huggingface_hub import snapshot_download

from tetris_env import Tetris, compute_board_features
from tetris_prompts import get_move_descriptors, build_pair
from tetris_player import _candidate_moves, play_episode


class LayaPlayer:
    """Tetris controller powered by convaiinnovations/laya decision model."""

    display_name = "Laya RLCD Decision Model (CPU)"

    def __init__(
        self,
        repo_id: str = "convaiinnovations/laya",
        top_k: Optional[int] = 6,
        device: Optional[str] = None,
        num_threads: int = 6
    ):
        self.repo_id = repo_id
        self.top_k = top_k
        self.device = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.display_name = f"Laya RLCD Decision Model ({'CUDA GPU (BF16)' if 'cuda' in self.device else 'CPU'})"
        if self.device == "cpu" and num_threads > 0:
            torch.set_num_threads(num_threads)

        print(f"[*] Loading Laya decision model from '{repo_id}' on {self.device}...")
        t0 = time.time()
        try:
            self.model_dir = snapshot_download(repo_id, local_files_only=True)
        except Exception:
            self.model_dir = snapshot_download(repo_id)

        # Import laya's bundled api and common definitions
        if self.model_dir not in sys.path:
            sys.path.insert(0, self.model_dir)

        from rl_agent_api import RLAgent
        from rl_common import DecisionModel, amp_dtype, build_model
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        class FastRLAgent(RLAgent):
            def __init__(self, model_dir, device=None):
                with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
                    self.cfg = json.load(f)
                self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
                self.tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
                if self.device.type == "cuda":
                    ecfg = AutoConfig.from_pretrained(os.path.join(model_dir, "encoder"))
                    with torch.device("cuda"):
                        enc = AutoModel.from_config(ecfg, attn_implementation="sdpa")
                        self.model = DecisionModel(enc, self.cfg["head_layers"], len(self.cfg["act_costs"]) + 1)
                    weights = load_file(os.path.join(model_dir, "model.safetensors"), device="cuda")
                    self.model.load_state_dict(weights, strict=True)
                    self.model.eval()
                else:
                    self.model = build_model(self.cfg, encoder_dir=os.path.join(model_dir, "encoder"))
                    self.model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
                    self.model.to(self.device).eval()

                self.model.encoder.config.reference_compile = False
                self.temperature = self.cfg.get("temperature", [1.0, 1.0, 1.0])
                self.temperature_by_options = self.cfg.get("temperature_by_options", {})
                self.dtype = amp_dtype(self.cfg.get("amp_dtype", "fp16"))
                if self.device.type == "cuda" and torch.cuda.get_device_capability(self.device)[0] < 8:
                    self.dtype = torch.float16

        self.agent = FastRLAgent(self.model_dir, device=self.device)

        # Warmup forward pass on GPU
        if "cuda" in self.device:
            try:
                _ = self.agent.system_one("Tetris", {"placement": {"type": "choice", "instructions": "warmup", "criteria": {"0_0": "opt1", "0_1": "opt2"}}})
                torch.cuda.synchronize()
            except Exception:
                pass

        self._last_latents: Optional[np.ndarray] = None
        self._last_best_idx: int = 0
        self._last_info: Dict[str, Any] = {}

        print(f"[+] Loaded laya in {time.time() - t0:.2f}s on {self.device}")

    @property
    def encoder(self):
        return self

    def latents(self, pairs=None) -> np.ndarray:
        """Returns hidden states of evaluated moves for telemetry visualization."""
        if self._last_latents is not None:
            return self._last_latents
        return np.zeros((1, 48), dtype=np.float32)

    def score_moves(self, env: Tetris, candidate_moves: List[Tuple[int, int]]) -> Tuple[List[float], np.ndarray]:
        """Evaluates candidate placements as options in a single choice question."""
        feats = env.state()["features"]
        criteria = {}
        key_list = []
        for rot, col in candidate_moves:
            key = f"{rot}_{col}"
            key_list.append(key)
            desc = get_move_descriptors(env, rot, col)
            sim = desc["sim_feats"]
            criteria[key] = (
                f"clears {desc['lines']} lines, {desc['delta_holes']} new holes, "
                f"landing height {desc['landing_h']}, bumpiness {sim['bumpiness']}"
            )

        state_str = (
            f"Tetris game status. Active tetromino: {env.current_piece}. Next piece: {env.next_piece}. "
            f"Board max height: {feats['max_height']}/20, holes: {feats['holes']}, bumpiness: {feats['bumpiness']}."
        )

        res = self.agent.system_one(
            state=state_str,
            questions={
                "placement": {
                    "type": "choice",
                    "instructions": "Select the best placement for the current Tetris piece. Prioritize clearing lines, flat stack, and zero buried holes.",
                    "criteria": criteria
                }
            }
        )

        ans = res["answers"]["placement"]
        probs_dict = ans.get("probabilities", {})
        scores = [probs_dict.get(k, 0.0) for k in key_list]

        # Normalize scores to sum to 1
        s_sum = sum(scores) or 1.0
        scores = [s / s_sum for s in scores]

        # Format [CON, ENT, NEU] proxy for UI telemetry
        cand_probs = []
        for p in scores:
            p_ent = float(p)
            p_con = float(max(0.0, 1.0 - p_ent) * 0.7)
            p_neu = float(max(0.0, 1.0 - p_ent - p_con))
            cand_probs.append([p_con, p_ent, p_neu])

        # Cache act probability and confidence
        self._last_act_prob = ans.get("rl_agent", {}).get("act_probability", 1.0)
        self._last_confidence = ans.get("confidence", 0.5)

        # Generate pseudo-latent from option probabilities for visualizer
        latents = []
        for p in scores:
            vec = np.zeros(48, dtype=np.float32)
            vec[0] = p
            vec[1] = 1.0 - p
            vec[2] = self._last_act_prob
            vec[3:10] = np.sin(np.arange(7) * (p + 0.1))
            latents.append(vec)
        self._last_latents = np.asarray(latents, dtype=np.float32)

        return scores, np.asarray(cand_probs, dtype=np.float32)

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

        info = {
            "lat_ms": lat_ms,
            "best_score": float(scores[best_idx]),
            "best_p_ent": float(probs[best_idx, 1]),
            "best_p_con": float(probs[best_idx, 0]),
            "best_p_neu": float(probs[best_idx, 2]),
            "confidence": getattr(self, "_last_confidence", float(scores[best_idx])),
            "act_prob": getattr(self, "_last_act_prob", 1.0),
            "n_candidates": len(candidate_moves),
            "total_legal": len(legal_moves),
            "candidates": [list(m) for m in candidate_moves],
            "cand_scores": [float(x) for x in scores],
            "cand_probs": [[float(v) for v in row] for row in probs],
            "mode": "laya (rlcd)"
        }
        if self._last_latents is not None and len(self._last_latents) > best_idx:
            info["best_latent"] = self._last_latents[best_idx].tolist()

        return best_move, info


def main():
    parser = argparse.ArgumentParser(description="Laya Tetris Player")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--render", action="store_true", help="Render ANSI animated game")
    parser.add_argument("--delay", type=float, default=0.05, help="Render delay per step in seconds")
    parser.add_argument("--top-k", type=int, default=6, help="Candidate moves pre-filtered per piece (0 for all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()

    env = Tetris(seed=args.seed, max_steps=args.max_steps)
    player = LayaPlayer(
        top_k=None if args.top_k <= 0 else args.top_k,
        num_threads=args.threads
    )

    print(f"\n--- Starting Laya evaluation ({args.episodes} episode(s)) ---")
    for ep in range(args.episodes):
        print(f"\nEpisode {ep + 1}/{args.episodes}:")
        env.reset()
        play_episode(env, player, render=args.render, delay=args.delay)


if __name__ == "__main__":
    main()
