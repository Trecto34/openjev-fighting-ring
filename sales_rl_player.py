"""SalesRLAgent Tetris Player: Reinforcement Learning Sales Conversion Model (arXiv:2503.23303)

This controller connects DeepMostInnovations/sales-conversion-model-reinf-learning
(PPO over BGE-large sequence embeddings predicting turn-by-turn deal conversion probability)
to play Tetris by treating:
  - The Tetris Board as the Customer with business objections (holes, height, bumpiness)
  - Candidate Placements as Sales Pitches proposing value solutions (line clears, stabilization)
  - The PPO Model as the Conversion Oracle predicting deal closure probability [0.0, 1.0]
"""

import argparse
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from stable_baselines3 import PPO

from tetris_env import Tetris, compute_board_features
from tetris_player import _candidate_moves


class SalesRLTetrisPlayer:
    """Tetris controller powered by the Sales Conversion PPO Model."""

    display_name = "SalesRLAgent (PPO Conversion Model)"

    def __init__(
        self,
        model_path: str = "sales_conversion_model.zip",
        embedding_model_name: str = "BAAI/bge-large-en-v1.5",
        top_k: Optional[int] = 6,
        device: str = "cpu"
    ):
        self.top_k = top_k
        self.device = torch.device(device)
        self.model_path = model_path
        self.embedding_model_name = embedding_model_name

        print(f"[*] Loading BGE embedding model '{embedding_model_name}' on {self.device}...")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        self.embedder = AutoModel.from_pretrained(embedding_model_name).to(self.device)
        self.embedder.eval()
        print(f"[+] Loaded embedding model in {time.time() - t0:.2f}s")

        print(f"[*] Loading Sales Conversion PPO model from '{model_path}'...")
        t1 = time.time()
        self.ppo = PPO.load(model_path, device="cpu")
        print(f"[+] Loaded Sales PPO model in {time.time() - t1:.2f}s")

        self.prob_history: List[float] = []
        self._last_latents: Optional[np.ndarray] = None

    def reset(self):
        """Reset the conversation probability history for a new game."""
        self.prob_history = []
        self._last_latents = None

    def _build_customer_objections(self, env: Tetris) -> str:
        """Formulate the current board's structural flaws as customer objections."""
        feats = env.state()["features"]
        holes = feats["holes"]
        bumpiness = feats["bumpiness"]
        height = feats["max_height"]

        if holes == 0 and height < 6:
            return "Customer is highly satisfied, running lean infrastructure with 0 defects and clean stack."
        elif holes > 2 or height > 12:
            return (f"Customer is expressing critical objections: {holes} hidden operational holes, "
                    f"severe stack risk at height {height}/20, and high friction (bumpiness {bumpiness}). "
                    "Customer urgently demands risk mitigation and immediate line clearance.")
        else:
            return (f"Customer reports moderate friction: {holes} defects, stack height {height}/20, "
                    f"and bumpiness {bumpiness}. Seeking reliable throughput.")

    def _build_sales_pitch(
        self,
        env: Tetris,
        rot: int,
        col: int,
        lines: int,
        delta_holes: int,
        landing_h: int,
        sim_feats: Dict,
        eng: float,
        eff: float
    ) -> str:
        """Formulate candidate placement as a value-driven SaaS sales pitch."""
        piece = env.current_piece
        if lines > 0:
            value_prop = f"delivers immediate ROI by clearing {lines} lines of operational debt"
        elif delta_holes == 0:
            value_prop = "maintains zero-defect alignment without creating any new holes"
        else:
            value_prop = f"requires acceptable compromise creating {delta_holes} temporary holes"

        return (
            f"Sales Proposal: Deploy piece {piece} at column {col}, orientation {rot}. "
            f"This solution {value_prop}, landing at height {landing_h}. "
            f"Resulting metrics: customer engagement {eng:.2f}, sales effectiveness {eff:.2f}, "
            f"post-deal stack height {sim_feats['max_height']}."
        )

    def score_moves(
        self,
        env: Tetris,
        candidate_moves: List[Tuple[int, int]]
    ) -> Tuple[List[float], Dict]:
        """Score candidate placements by predicted sales conversion probability."""
        curr_holes = env.state()["features"]["holes"]
        curr_step = env.pieces_placed

        pitches: List[str] = []
        metrics_list: List[List[float]] = []

        for rot, col in candidate_moves:
            sim_grid, lines, landing_h = env.simulate_placement(rot, col)
            sim_feats = compute_board_features(sim_grid)
            delta_holes = sim_feats["holes"] - curr_holes
            is_dead = sim_feats["max_height"] >= 20

            # 1. customer_engagement [0.0 - 1.0]: board stability / low holes & bumpiness
            eng = max(0.0, min(1.0, 1.0 - (sim_feats["holes"] * 0.15 + sim_feats["bumpiness"] * 0.04)))

            # 2. sales_effectiveness [0.0 - 1.0]: lines cleared vs defect creation
            eff = max(0.0, min(1.0, 0.5 + (lines * 0.30) - (delta_holes * 0.35) - (landing_h * 0.02)))

            # 3. outcome: 1.0 if board survives, 0.0 if topped out
            outcome = 0.0 if is_dead else 1.0

            # 4. progress: game advancement stage [0.0 - 1.0]
            progress = min(1.0, curr_step / 80.0)

            # 5. pitch text
            pitch = self._build_sales_pitch(env, rot, col, lines, delta_holes, landing_h, sim_feats, eng, eff)
            pitches.append(pitch)
            metrics_list.append([eng, eff, float(curr_step), outcome, progress])

        # Batch tokenization and BGE text embedding (1024 dimensions)
        inputs = self.tokenizer(pitches, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.embedder(**inputs)
            # CLS token pooling normalized to unit sphere
            embs = outputs[0][:, 0]
            embs = torch.nn.functional.normalize(embs, p=2, dim=1).cpu().numpy()

        n_cands = len(candidate_moves)
        # Assemble 1040-dimensional state vector:
        # [0:1024]   -> text embedding
        # [1024:1029] -> [customer_engagement, sales_effectiveness, conversation_length, outcome, progress]
        # [1029]      -> turn_number
        # [1030:1040] -> last 10 conversion probabilities (zero-padded)
        obs_batch = np.zeros((n_cands, 1040), dtype=np.float32)
        obs_batch[:, :1024] = embs

        for i in range(n_cands):
            obs_batch[i, 1024:1029] = metrics_list[i]
            obs_batch[i, 1029] = float(curr_step)
            for h_idx, past_p in enumerate(self.prob_history[-10:]):
                obs_batch[i, 1030 + h_idx] = past_p

        # Run PPO forward pass and extract layer activations for the neural graph
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs_batch, device="cpu")
            # Layer activations from features_extractor linear_network:
            # [0]: Linear(1040, 512), [1]: ReLU, [2]: Linear(512, 256), [3]: ReLU, [4]: Linear(256, 64), [5]: ReLU
            h0 = self.ppo.policy.features_extractor.linear_network[0](obs_tensor)
            h1 = self.ppo.policy.features_extractor.linear_network[1](h0)
            h2 = self.ppo.policy.features_extractor.linear_network[2](h1)
            h3 = self.ppo.policy.features_extractor.linear_network[3](h2)
            h4 = self.ppo.policy.features_extractor.linear_network[4](h3)
            latents = self.ppo.policy.features_extractor.linear_network[5](h4).cpu().numpy()

            # PPO action prediction
            preds, _ = self.ppo.predict(obs_batch, deterministic=True)

        scores = preds.flatten().tolist()
        self._last_latents = latents

        extra = {
            "pitches": pitches,
            "metrics": metrics_list,
            "latents": latents
        }
        return scores, extra

    def choose_move(self, env: Tetris) -> Tuple[Tuple[int, int], Dict]:
        legal_moves = env.get_legal_moves()
        if not legal_moves:
            return (0, 0), {"lat_ms": 0.0}

        # Filter candidate moves to top_k if specified
        candidate_moves = _candidate_moves(env, legal_moves, self.top_k)

        t0 = time.perf_counter()
        scores, extra = self.score_moves(env, candidate_moves)
        lat_ms = (time.perf_counter() - t0) * 1000

        best_idx = int(np.argmax(scores))
        best_move = candidate_moves[best_idx]
        best_prob = float(scores[best_idx])

        # Record this conversion probability in the sliding window history
        self.prob_history.append(best_prob)

        # Map conversion probability into decision confidence:
        # ENT = Deal closed / High conversion probability
        # CON = Deal lost / Objections unresolved
        # NEU = Margin / Under negotiation
        p_ent = round(best_prob, 3)
        p_con = round(max(0.0, 1.0 - best_prob), 3)
        p_neu = round(0.05, 3)

        info = {
            "lat_ms": lat_ms,
            "best_score": best_prob,
            "conversion_prob": best_prob,
            "best_p_ent": p_ent,
            "best_p_con": p_con,
            "best_p_neu": p_neu,
            "best_pitch": extra["pitches"][best_idx],
            "customer_objections": self._build_customer_objections(env),
            "customer_engagement": extra["metrics"][best_idx][0],
            "sales_effectiveness": extra["metrics"][best_idx][1],
            "n_candidates": len(candidate_moves),
            "total_legal": len(legal_moves),
            "candidates": [list(m) for m in candidate_moves],
            "cand_scores": [float(s) for s in scores],
            "cand_probs": [[p_con, p_ent, p_neu] for _ in candidate_moves],
            "best_latent": extra["latents"][best_idx].tolist() if extra.get("latents") is not None else None
        }
        return best_move, info


def play_sales_rl_episode(
    env: Tetris,
    player: SalesRLTetrisPlayer,
    render: bool = False,
    delay: float = 0.0,
    verbose: bool = True
) -> Dict:
    """Run a single Tetris game episode with the Sales RL Player."""
    player.reset()
    state = env.reset()
    total_reward = 0
    move_latencies = []
    move_records = []

    if render:
        os.system("clear" if os.name == "posix" else "cls")
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
            "lat_ms": lat,
            "conversion_prob": info.get("conversion_prob", 0.0),
            "pitch": info.get("best_pitch", "")
        }
        move_records.append(rec)

        if verbose and not render:
            p_val = info.get("conversion_prob", 0.0)
            status_emoji = "🟢" if p_val >= 0.7 else ("🟡" if p_val >= 0.4 else "🔴")
            print(
                f"  [Step {env.pieces_placed:3d}] {status_emoji} Placed {state['current_piece']} "
                f"at (rot={rot}, col={col}) | Deal Conv Prob: {p_val * 100:.1f}% | "
                f"Lines: {env.lines_cleared} | Score: {env.score} ({lat:.1f}ms)",
                flush=True
            )

        if render:
            os.system("clear" if os.name == "posix" else "cls")
            print(env.render_ascii(highlight_col=col))
            p_val = info.get("conversion_prob", 0.0)
            print(f"Deal Closing Probability: {p_val * 100:.1f}% | Latency: {lat:.1f}ms")
            print(f"Pitch: {info.get('best_pitch', '')[:90]}...")
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
        print(
            f"\n[+] Game Over! Pieces Placed: {env.pieces_placed} | "
            f"Lines Cleared: {env.lines_cleared} | Score: {env.score} | "
            f"Avg Latency: {avg_lat:.1f}ms\n"
        )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Tetris with Sales RL Conversion Model (arXiv:2503.23303)")
    parser.add_argument("--model-path", default="sales_conversion_model.zip", help="Path to PPO model zip")
    parser.add_argument("--embedding-model", default="BAAI/bge-large-en-v1.5", help="Embedding model name or path")
    parser.add_argument("--top-k", type=int, default=6, help="Candidate moves to evaluate per piece (0 for all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for piece generation")
    parser.add_argument("--render", action="store_true", help="Render ASCII board in terminal")
    parser.add_argument("--delay", type=float, default=0.05, help="Delay between moves when rendering")
    parser.add_argument("--max-steps", type=int, default=100, help="Max pieces before termination")
    args = parser.parse_args()

    print(f"=== Starting SalesRLAgent Tetris Session ===")
    player = SalesRLTetrisPlayer(
        model_path=args.model_path,
        embedding_model_name=args.embedding_model,
        top_k=(None if args.top_k <= 0 else args.top_k)
    )
    env = Tetris(seed=args.seed, max_steps=args.max_steps)
    summary = play_sales_rl_episode(env, player, render=args.render, delay=args.delay, verbose=True)
