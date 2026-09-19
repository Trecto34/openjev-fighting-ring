"""Tetris player using AlexWortega/openjev NLI cross-encoder."""
import argparse
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from tetris_env import Tetris, dellacherie_eval, oracle_policy
from tetris_prompts import PROMPT_STRATEGIES, generate_candidate_pairs
from modeling_openjev import OpenJevCrossEncoder, LatentMLPHead, TetrisRankHead, ENT, CON


def _quick_move_score(env: Tetris, move: Tuple[int, int], curr_holes: Optional[float] = None) -> float:
    """Rank candidates with the same board evaluation used by the oracle."""
    sim_grid, lines, landing_h = env.simulate_placement(move[0], move[1])
    return dellacherie_eval(sim_grid, lines, landing_h)


def _candidate_moves(env: Tetris, legal_moves: List[Tuple[int, int]], top_k: Optional[int]):
    if top_k is None or top_k <= 0 or len(legal_moves) <= top_k:
        return legal_moves
    return sorted(legal_moves, key=lambda move: _quick_move_score(env, move), reverse=True)[:top_k]


def _unique_pairs(pairs):
    """Return unique pairs and an expansion index, preserving first-seen order."""
    unique, indices, positions = [], [], {}
    for pair in pairs:
        idx = positions.get(pair)
        if idx is None:
            idx = len(unique)
            positions[pair] = idx
            unique.append(pair)
        indices.append(idx)
    return unique, np.asarray(indices, dtype=np.int64)


class OpenJevPlayer:
    def __init__(
        self,
        model_path: str = "AlexWortega/openjev",
        subfolder: str = "qwen3.5-4b-nli",
        strategy: str = "qualitative",
        top_k: Optional[int] = None,
        device: Optional[str] = None,
        dtype=torch.bfloat16,
        bs: int = 32
    ):
        print(f"[*] Loading OpenJevCrossEncoder from {model_path} ({subfolder})...")
        t0 = time.time()
        self.strategy = strategy
        self.top_k = top_k
        self.encoder = OpenJevCrossEncoder(
            path=model_path,
            subfolder=subfolder,
            device=device,
            dtype=dtype,
            bs=bs
        )
        print(f"[+] Loaded openjev in {time.time() - t0:.2f}s on {self.encoder.device}")

    def score_moves(self, env: Tetris, legal_moves: List[Tuple[int, int]]) -> Tuple[List[float], np.ndarray]:
        """Scores each legal move using P(entailment) - P(contradiction)."""
        pairs = generate_candidate_pairs(env, legal_moves, strategy=self.strategy)
        unique_pairs, indices = _unique_pairs(pairs)
        probs = self.encoder.predict(unique_pairs)[indices]  # [CON, ENT, NEU]
        latents = getattr(self.encoder, "last_latents", None)
        self._last_latents = latents[indices] if latents is not None else None
        # Net entailment score: P(entailment) - P(contradiction)
        scores = (probs[:, ENT] - probs[:, CON]).tolist()
        return scores, probs

    def choose_move(self, env: Tetris) -> Tuple[Tuple[int, int], Dict]:
        legal_moves = env.get_legal_moves()
        if not legal_moves:
            return (0, 0), {}

        # If top_k is set, pre-filter candidate moves to top_k
        candidate_moves = _candidate_moves(env, legal_moves, self.top_k)

        t0 = time.perf_counter()
        scores, probs = self.score_moves(env, candidate_moves)
        lat_ms = (time.perf_counter() - t0) * 1000

        best_idx = int(np.argmax(scores))
        best_move = candidate_moves[best_idx]

        info = {
            "lat_ms": lat_ms,
            "best_score": scores[best_idx],
            "best_p_ent": float(probs[best_idx, ENT]),
            "best_p_con": float(probs[best_idx, CON]),
            "best_p_neu": float(probs[best_idx, 2]),
            "n_candidates": len(candidate_moves),
            "total_legal": len(legal_moves),
            # Per-candidate results, for the activation graph's decision layer.
            "candidates": [list(m) for m in candidate_moves],
            "cand_scores": [float(x) for x in scores],
            "cand_probs": [[float(v) for v in row] for row in probs]
        }
        if self._last_latents is not None:
            info["best_latent"] = self._last_latents[best_idx].tolist()
        return best_move, info


class OraclePlayer:
    def choose_move(self, env: Tetris) -> Tuple[Tuple[int, int], Dict]:
        t0 = time.perf_counter()
        move = oracle_policy(env)
        lat_ms = (time.perf_counter() - t0) * 1000
        return move, {"lat_ms": lat_ms}


class RandomPlayer:
    def __init__(self, rng=None):
        self.rng = rng or np.random.RandomState(42)

    def choose_move(self, env: Tetris) -> Tuple[Tuple[int, int], Dict]:
        legal_moves = env.get_legal_moves()
        if not legal_moves:
            return (0, 0), {"lat_ms": 0.0}
        idx = self.rng.choice(len(legal_moves))
        return legal_moves[idx], {"lat_ms": 0.0}


def play_episode(
    env: Tetris,
    player,
    render: bool = False,
    delay: float = 0.0,
    verbose: bool = True
) -> Dict:
    state = env.reset()
    total_reward = 0
    move_latencies = []
    move_records = []

    if render:
        os.system('clear' if os.name == 'posix' else 'cls')
        print(env.render_ascii())

    while not env.done:
        move, info = player.choose_move(env)
        rot, col = move
        state, reward, done = env.step(rot, col)
        total_reward += reward
        lat = info.get("lat_ms", 0.0)
        move_latencies.append(lat)

        rec = {
            "step": env.pieces_placed,
            "piece": state["current_piece"],
            "move": [rot, col],
            "reward": reward,
            "lines": env.lines_cleared,
            "score": env.score,
            "lat_ms": lat
        }
        if "best_p_ent" in info:
            rec["p_ent"] = info["best_p_ent"]
            rec["p_con"] = info["best_p_con"]
        move_records.append(rec)

        if verbose and not render:
            ent_str = f" | ENT: {info['best_p_ent']:.3f} CON: {info['best_p_con']:.3f}" if "best_p_ent" in info else ""
            print(f"  [Step {env.pieces_placed:3d}] Placed {state['current_piece']} at (rot={rot}, col={col}){ent_str} | Lines: {env.lines_cleared} | Score: {env.score} ({lat/1000:.1f}s)", flush=True)

        if render:
            os.system('clear' if os.name == 'posix' else 'cls')
            print(env.render_ascii(highlight_col=col))
            if "best_p_ent" in info:
                print(f"openjev Entailment: {info['best_p_ent']:.3f} | Contradiction: {info['best_p_con']:.3f} | Latency: {lat:.1f}ms")
            if delay > 0:
                time.sleep(delay)

    avg_lat = float(np.mean(move_latencies)) if move_latencies else 0.0
    summary = {
        "pieces_placed": env.pieces_placed,
        "lines_cleared": env.lines_cleared,
        "score": env.score,
        "avg_lat_ms": avg_lat,
        "moves": move_records
    }
    if verbose:
        print(f"Game Over! Pieces: {env.pieces_placed} | Lines: {env.lines_cleared} | Score: {env.score} | Avg Latency: {avg_lat:.1f}ms")
    return summary


def train_latent_mlp(
    player: OpenJevPlayer,
    episodes: int = 10,
    noise: float = 0.15,
    out_dir: str = "mlp_head"
) -> LatentMLPHead:
    """Collects noisy oracle rollouts, extracts cross-encoder latents, and trains LatentMLPHead with soft BCE."""
    print(f"[*] Collecting training data from {episodes} noisy oracle rollouts...")
    oracle = OraclePlayer()
    pairs_all = []
    gold_all = []
    qid_all = []

    q_counter = 0
    for ep in range(episodes):
        env = Tetris(seed=1000 + ep, max_steps=100)
        env.reset()
        while not env.done:
            legal_moves = env.get_legal_moves()
            if not legal_moves:
                break
            best_move, _ = oracle.choose_move(env)

            # Generate pairs for all legal moves
            pairs = generate_candidate_pairs(env, legal_moves, strategy=player.strategy)
            for m_idx, (rot, col) in enumerate(legal_moves):
                pairs_all.append(pairs[m_idx])
                is_gold = 1.0 if (rot, col) == best_move else 0.0
                gold_all.append(is_gold)
                qid_all.append(q_counter)
            q_counter += 1

            # Step with noisy oracle to explore states
            if np.random.rand() < noise:
                step_move = legal_moves[np.random.choice(len(legal_moves))]
            else:
                step_move = best_move
            env.step(step_move[0], step_move[1])

    print(f"[+] Collected {len(pairs_all)} candidate pairs across {q_counter} decision points.")
    print("[*] Extracting frozen cross-encoder latents...")
    X = player.encoder.latents(pairs_all)
    gold = np.array(gold_all, dtype=np.float32)
    qid = np.array(qid_all)

    print("[*] Fitting LatentMLPHead with soft BCE...")
    head = LatentMLPHead(d=X.shape[1], hidden=512, dropout=0.1, eps=0.1, epochs=40, patience=6)
    head.fit(X, gold, qid, val_frac=0.15)
    print(f"[+] LatentMLPHead trained! Val Acc: {head.val_acc:.3f}")
    head.save(out_dir)
    print(f"[+] Saved MLP head to {out_dir}")
    return head


def main():
    parser = argparse.ArgumentParser(description="openjev Tetris Player")
    parser.add_argument("--mode", default="zero-shot", choices=["zero-shot", "oracle", "random", "train-mlp", "mlp", "kev", "laya", "spark"])
    parser.add_argument("--strategy", default="qualitative", choices=PROMPT_STRATEGIES)
    parser.add_argument("--ckpt", default="AlexWortega/openjev")
    parser.add_argument("--subfolder", default="qwen3.5-4b-nli")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--render", action="store_true", help="Render ANSI animated game")
    parser.add_argument("--delay", type=float, default=0.05, help="Render delay per step in seconds")
    parser.add_argument("--top-k", type=int, default=3, help="Evaluate top-K candidate placements (or 0 for all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--mlp-dir", default="mlp_head")
    parser.add_argument("--kev-mode", default="choice", choices=["choice", "nli"])
    parser.add_argument("--spark-url", default="http://127.0.0.1:8095", help="Endpoint for djev-spark DGX server")
    args = parser.parse_args()

    top_k = None if args.top_k <= 0 else args.top_k
    env = Tetris(seed=args.seed, max_steps=args.max_steps)

    if args.mode == "oracle":
        player = OraclePlayer()
    elif args.mode == "random":
        player = RandomPlayer(np.random.RandomState(args.seed))
    elif args.mode == "kev":
        from kev_player import KevPlayer
        player = KevPlayer(mode=args.kev_mode, top_k=top_k, device=args.device)
    elif args.mode == "laya":
        from laya_player import LayaPlayer
        player = LayaPlayer(top_k=top_k, device=args.device)
    elif args.mode == "spark":
        from djev_spark_player import DjevSparkPlayer
        player = DjevSparkPlayer(endpoint_url=args.spark_url, top_k=top_k, device=args.device)
    elif args.mode == "zero-shot":
        player = OpenJevPlayer(
            model_path=args.ckpt,
            subfolder=args.subfolder,
            strategy=args.strategy,
            top_k=top_k,
            device=args.device,
            bs=args.bs
        )
    elif args.mode == "train-mlp":
        player = OpenJevPlayer(
            model_path=args.ckpt,
            subfolder=args.subfolder,
            strategy=args.strategy,
            device=args.device,
            bs=args.bs
        )
        train_latent_mlp(player, episodes=15, out_dir=args.mlp_dir)
        return
    else:
        raise ValueError(f"Mode {args.mode} not supported yet.")

    print(f"\n--- Starting {args.mode} evaluation ({args.episodes} episode(s)) ---")
    for ep in range(args.episodes):
        print(f"\nEpisode {ep + 1}/{args.episodes}:")
        env.reset()
        play_episode(env, player, render=args.render, delay=args.delay)


if __name__ == "__main__":
    main()


def _load_roberta_head(safetensors_path: str):
    """Pull the 4 classification-head tensors out of a safetensors file (no torch)."""
    import json as _json, struct
    with open(safetensors_path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = _json.loads(fh.read(n))
        base = 8 + n

        def get(name):
            meta = hdr[name]
            s, e = meta["data_offsets"]
            fh.seek(base + s)
            dt = {"F32": np.float32, "F16": np.float16}[meta["dtype"]]
            return np.frombuffer(fh.read(e - s), dtype=dt).reshape(meta["shape"]).astype(np.float32)

        return (get("classifier.dense.weight"), get("classifier.dense.bias"),
                get("classifier.out_proj.weight"), get("classifier.out_proj.bias"))


class VulkanNLIPlayer:
    """NLI cross-encoder run by llama.cpp on the GPU.

    llama-server (Vulkan) runs the 24 transformer layers and returns the pooled CLS
    vector; the classification head (dense -> tanh -> out_proj) is two small matmuls,
    done here in numpy. Verified against transformers: max prob diff 0.0024 (f16).
    """

    def __init__(self, url: str = "http://127.0.0.1:8091",
                 model_dir: str = "/home/server/models/roberta-large-mnli",
                 strategy: str = "qualitative", top_k: Optional[int] = None,
                 policy_head: Optional[str] = None):
        import json as _json
        from urllib.request import urlopen
        self.url = url.rstrip("/")
        self.strategy = strategy
        self.top_k = top_k
        self.policy_head = TetrisRankHead.load(policy_head, device="cpu") if policy_head else None
        # Name the actual weights, not the technique — otherwise the board just says
        # "NLI Cross-Encoder" and you cannot tell which model is playing it.
        suffix = " + Tetris head" if self.policy_head else ""
        self.display_name = f"{os.path.basename(model_dir.rstrip('/'))} NLI (Vulkan GPU){suffix}"

        cfg = _json.load(open(os.path.join(model_dir, "config.json")))
        # This model labels [CON, NEU, ENT]; the rest of the code expects [CON, ENT, NEU].
        by_name = {v.lower()[:3]: int(k) for k, v in cfg["id2label"].items()}
        self.perm = [by_name["con"], by_name["ent"], by_name["neu"]]

        self.W1, self.b1, self.W2, self.b2 = _load_roberta_head(os.path.join(model_dir, "model.safetensors"))
        try:
            urlopen(self.url + "/health", timeout=5).read()
        except Exception as e:
            raise RuntimeError(f"llama-server not reachable at {self.url} ({e}). "
                               f"Start it with: llama-server -m <gguf> --port 8091 -ngl 99 "
                               f"--embeddings --pooling cls --embd-normalize -1") from e
        print(f"[+] {self.display_name} ready via {self.url}")

    def _embed(self, texts: List[str]) -> np.ndarray:
        import json as _json
        from urllib.request import Request, urlopen
        req = Request(self.url + "/embedding", _json.dumps({"content": texts}).encode(),
                      {"Content-Type": "application/json"})
        rows = _json.load(urlopen(req, timeout=120))
        return np.asarray([r["embedding"][0] for r in rows], dtype=np.float32)

    def latents(self, pairs) -> np.ndarray:
        """Pooled CLS vectors — the real hidden state feeding the classifier.

        Served from the last scoring batch when possible: the graph always asks for a
        pair we just scored, so re-encoding it would double the work per piece.
        """
        texts = [f"{p}</s></s>{h}" for p, h in pairs]
        cached = getattr(self, "_cls_cache", {})
        if all(t in cached for t in texts):
            return np.stack([cached[t] for t in texts])
        return self._embed(texts)

    @property
    def encoder(self):  # the telemetry code asks the player for its encoder
        return self

    def _head(self, X: np.ndarray) -> np.ndarray:
        H = np.tanh(X @ self.W1.T + self.b1)
        Z = H @ self.W2.T + self.b2
        Z = Z - Z.max(axis=1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(axis=1, keepdims=True)
        return P[:, self.perm]  # -> [CON, ENT, NEU]

    def score_moves(self, env: Tetris, legal_moves: List[Tuple[int, int]]) -> Tuple[List[float], np.ndarray]:
        pairs = generate_candidate_pairs(env, legal_moves, strategy=self.strategy)
        unique_pairs, indices = _unique_pairs(pairs)
        unique_texts = [f"{p}</s></s>{h}" for p, h in unique_pairs]
        X = self._embed(unique_texts)
        self._cls_cache = dict(zip(unique_texts, X))
        if self.policy_head is None:
            probs = self._head(X)[indices]
            scores = probs[:, ENT] - probs[:, CON]
        else:
            policy_scores = self.policy_head.predict(X)
            scores = policy_scores[indices]
            p = 1.0 / (1.0 + np.exp(-np.clip(scores, -30.0, 30.0)))
            probs = np.stack([1.0 - p, p, np.zeros_like(p)], axis=1)
        self._last_latents = X[indices]
        return scores.tolist(), probs

    def choose_move(self, env: Tetris) -> Tuple[Tuple[int, int], Dict]:
        legal_moves = env.get_legal_moves()
        if not legal_moves:
            return (0, 0), {}
        candidate_moves = legal_moves
        candidate_moves = _candidate_moves(env, legal_moves, self.top_k)

        t0 = time.perf_counter()
        scores, probs = self.score_moves(env, candidate_moves)
        lat_ms = (time.perf_counter() - t0) * 1000

        best_idx = int(np.argmax(scores))
        info = {
            "lat_ms": lat_ms,
            "best_score": scores[best_idx],
            "best_p_ent": float(probs[best_idx, ENT]),
            "best_p_con": float(probs[best_idx, CON]),
            "best_p_neu": float(probs[best_idx, 2]),
            "n_candidates": len(candidate_moves),
            "total_legal": len(legal_moves),
            "candidates": [list(m) for m in candidate_moves],
            "cand_scores": [float(x) for x in scores],
            "cand_probs": [[float(v) for v in row] for row in probs],
        }
        if hasattr(self, "_last_latents"):
            info["best_latent"] = self._last_latents[best_idx].tolist()
        return candidate_moves[best_idx], info


class QwenMovePlayer:
    """Qwen2.5-1.5B-Instruct ranking placements by constrained single-token decode.

    The repo harshatheg/Qwen-2.5-1B-RLCD ships no weights — it is a decoding-engine demo
    whose own core/engine_torch.py loads Qwen/Qwen2.5-1.5B-Instruct, so that is the model
    used here, served by llama.cpp on Vulkan. Following that repo's idea: every candidate
    is asked in ONE batch and the answer is a single constrained token, so P(yes) - P(no)
    comes back without generating any prose.
    """

    YES = {"yes", "Yes", "YES"}
    NO = {"no", "No", "NO"}

    SYSTEM = "You are an expert Tetris player. Answer with a single word: yes or no."

    def __init__(self, url: str = "http://127.0.0.1:8092",
                 strategy: str = "qualitative", top_k: Optional[int] = None, n_probs: int = 40):
        from urllib.request import urlopen
        self.url = url.rstrip("/")
        self.strategy = strategy
        self.top_k = top_k
        self.n_probs = n_probs
        self.display_name = "Qwen2.5-1.5B-Instruct (Vulkan GPU)"
        self._last_dist = None
        try:
            urlopen(self.url + "/health", timeout=5).read()
        except Exception as e:
            raise RuntimeError(f"llama-server not reachable at {self.url} ({e}). Start it with: "
                               f"llama-server -m qwen2.5-1.5b-instruct-q4_k_m.gguf --port 8092 -ngl 99 -np 32") from e
        print(f"[+] {self.display_name} ready via {self.url}")

    def _prompt(self, premise: str) -> str:
        return (f"<|im_start|>system\n{self.SYSTEM}<|im_end|>\n"
                f"<|im_start|>user\n{premise}\nIs this a good move?<|im_end|>\n"
                f"<|im_start|>assistant\n")

    def _ask(self, premises: List[str]) -> List[Dict]:
        import json as _json
        from urllib.request import Request, urlopen
        body = {"prompt": [self._prompt(p) for p in premises], "n_predict": 1,
                "n_probs": self.n_probs, "temperature": 0, "cache_prompt": True}
        req = Request(self.url + "/completion", _json.dumps(body).encode(), {"Content-Type": "application/json"})
        out = _json.load(urlopen(req, timeout=120))
        return out if isinstance(out, list) else [out]

    def score_moves(self, env: Tetris, legal_moves: List[Tuple[int, int]]) -> Tuple[List[float], np.ndarray]:
        pairs = generate_candidate_pairs(env, legal_moves, strategy=self.strategy)
        unique_pairs, indices = _unique_pairs(pairs)
        results = self._ask([premise for premise, _ in unique_pairs])
        results = [results[int(i)] for i in indices]
        probs, dists = [], []
        for res in results:
            top = res["completion_probabilities"][0]["top_logprobs"]
            y = sum(math.exp(t["logprob"]) for t in top if t["token"].strip() in self.YES)
            n = sum(math.exp(t["logprob"]) for t in top if t["token"].strip() in self.NO)
            probs.append([n, y, max(0.0, 1.0 - y - n)])  # [CON, ENT, NEU]
            dists.append([math.exp(t["logprob"]) for t in top])
        probs = np.asarray(probs, dtype=np.float32)
        self._last_dist = dists
        return (probs[:, ENT] - probs[:, CON]).tolist(), probs

    def latents(self, pairs) -> np.ndarray:
        """The model's own answer distribution — its real output activations.

        There is no pooled hidden state to read over llama-server's completion API, so the
        graph shows the next-token probabilities that the decision is actually made from.
        """
        if self._last_dist:
            return np.asarray([self._last_dist[self._last_best]], dtype=np.float32)
        return np.zeros((1, self.n_probs), dtype=np.float32)

    @property
    def encoder(self):
        return self

    def choose_move(self, env: Tetris) -> Tuple[Tuple[int, int], Dict]:
        legal_moves = env.get_legal_moves()
        if not legal_moves:
            return (0, 0), {}
        candidate_moves = _candidate_moves(env, legal_moves, self.top_k)

        t0 = time.perf_counter()
        scores, probs = self.score_moves(env, candidate_moves)
        lat_ms = (time.perf_counter() - t0) * 1000

        best_idx = int(np.argmax(scores))
        self._last_best = best_idx
        return candidate_moves[best_idx], {
            "lat_ms": lat_ms,
            "best_score": scores[best_idx],
            "best_p_ent": float(probs[best_idx, ENT]),
            "best_p_con": float(probs[best_idx, CON]),
            "best_p_neu": float(probs[best_idx, 2]),
            "n_candidates": len(candidate_moves),
            "total_legal": len(legal_moves),
            "candidates": [list(m) for m in candidate_moves],
            "cand_scores": [float(x) for x in scores],
            "cand_probs": [[float(v) for v in row] for row in probs],
        }
