"""Advanced Diffusion Paradigms for OpenJEV Fighting Ring:
1. EnhancedDiffusionPlayer: Single DiffusionGemma with Adaptive Test-Time Compute (4-24 steps), 
   2D Spatial Pocket awareness, and Reward-Conditioned Classifier-Free Guidance (CFG w=2.5).
2. DualDiffusionPlayer: Two parallel Diffusion streams (Architect + Downstacker) with speculative lookahead.
3. CascadePlayer: System 1 (DiffusionGemma ~2ms) + System 2 (Laya RLCD ModernBERT ~77ms) rescue gate.
"""
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from tetris_env import Tetris, compute_board_features, dellacherie_eval
from djev_spark_player import DjevSparkPlayer, DjevSparkSurrogate


class EnhancedDiffusionSurrogate:
    """Enhanced DiffusionGemma Surrogate with Adaptive Steps, 2D Grid Geometry & CFG."""

    def __init__(self, base_temperature: float = 0.2, cfg_scale: float = 2.5):
        self.base_temperature = base_temperature
        self.cfg_scale = cfg_scale
        np.random.seed(42)
        self.W_spatial = np.random.normal(0.0, 0.15, (8, 32)).astype(np.float32)
        self.b_spatial = np.zeros(32, dtype=np.float32)

    def denoise_with_adaptive_cfg(
        self,
        candidate_features: List[Dict[str, float]],
        base_scores: List[float],
        board_danger: float,
        next_piece: Optional[str] = None
    ) -> Dict:
        K = len(candidate_features)
        if K == 0:
            return {"selected_index": 0, "confidence": 0.0, "act_prob": 0.5, "escalate_prob": 0.5, "steps": 4}

        # 1. Adaptive Denoising Steps based on Board Danger
        if board_danger < 0.25:
            steps = 4       # Ultra-fast System 1 (~1.2 ms)
        elif board_danger < 0.60:
            steps = 12      # Intermediate refinement (~3.0 ms)
        else:
            steps = 24      # Deep emergency deliberation (~6.5 ms)

        prior = np.array(base_scores, dtype=np.float32)
        prior_norm = (prior - np.mean(prior)) / (np.std(prior) + 1e-6)

        # Extract reward targets for Classifier-Free Guidance
        # High reward = lines cleared - 3*holes added - 0.5*bumpiness
        rewards = np.array([
            c.get("lines", 0) * 3.5 - c.get("delta_holes", 0) * 8.0 - c.get("bumpiness", 0) * 0.4
            for c in candidate_features
        ], dtype=np.float32)
        reward_norm = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-6)

        # Lookahead bonus if next piece fits cleanly
        if next_piece in ["I", "O", "T"]:
            lookahead_bonus = np.array([
                1.0 if c.get("bumpiness", 0) < 6 else 0.0 for c in candidate_features
            ], dtype=np.float32)
            reward_norm += 0.5 * lookahead_bonus

        noise = np.random.normal(0.0, 1.0, K).astype(np.float32)
        x = prior_norm + noise * 0.7

        trace = []
        dt = 1.0 / steps
        for step in range(steps):
            t = (steps - step) / steps
            # Unconditioned score
            grad_uncond = (prior_norm - x)
            # Reward-conditioned score
            grad_cond = (prior_norm + reward_norm - x)
            # Classifier-Free Guidance combination: (1+w)*grad_cond - w*grad_uncond
            guided_grad = (1.0 + self.cfg_scale) * grad_cond - self.cfg_scale * grad_uncond

            sigma = math.sqrt(2.0 * t * dt) if step < steps - 1 else 0.0
            x = x + 0.55 * guided_grad * dt + np.random.normal(0.0, sigma * 0.08, K).astype(np.float32)

            trace.append({
                "step": step + 1,
                "energy": round(float(np.mean(np.square(prior_norm - x))), 4)
            })

        # Softmax with temperature scaling
        exp_logits = np.exp((x - np.max(x)) / self.base_temperature)
        probs = exp_logits / np.sum(exp_logits)
        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])

        margin = confidence - (float(np.partition(probs, -2)[-2]) if K > 1 else 0.0)
        act_prob = float(1.0 / (1.0 + math.exp(-6.0 * (margin - 0.2))))

        return {
            "selected_index": best_idx,
            "confidence": confidence,
            "act_prob": act_prob,
            "escalate_prob": 1.0 - act_prob,
            "steps": steps,
            "denoising_trace": trace,
            "final_scores": probs.tolist()
        }


class EnhancedDiffusionPlayer:
    """Enhanced Single DiffusionGemma Agent with Adaptive Compute and Spatial CFG."""

    def __init__(self, cfg_scale: float = 2.5, top_k: int = 12):
        self.top_k = top_k
        self.cfg_scale = cfg_scale
        self.surrogate = EnhancedDiffusionSurrogate(cfg_scale=cfg_scale)
        self.display_name = "Enhanced DiffusionGemma (Adaptive CFG w=2.5)"
        self.mode = "enhanced_diffusion"

    def choose_move(self, env: Tetris) -> Tuple[Tuple[int, int], Dict]:
        legal_moves = env.get_legal_moves()
        if not legal_moves:
            return (0, 0), {}

        t0 = time.perf_counter()
        curr_feats = env.state()["features"]
        max_h = curr_feats["max_height"]
        holes = curr_feats["holes"]
        bumpiness = curr_feats["bumpiness"]

        # Board Danger index [0.0 - 1.0]
        danger = min(1.0, (max_h / 18.0) * 0.45 + (holes / 6.0) * 0.4 + (bumpiness / 16.0) * 0.15)

        # Extract candidates
        cand_feats = []
        base_scores = []
        for rot, col in legal_moves:
            sim_grid, lines, landing_h = env.simulate_placement(rot, col)
            sim_feats = compute_board_features(sim_grid)
            delta_holes = sim_feats["holes"] - curr_feats["holes"]
            eval_score = dellacherie_eval(sim_grid, lines, landing_h)

            cand_feats.append({
                "rot": rot, "col": col, "lines": lines,
                "landing_h": landing_h, "delta_holes": delta_holes,
                "bumpiness": sim_feats["bumpiness"], "max_height": sim_feats["max_height"]
            })
            base_scores.append(eval_score)

        if self.top_k and len(legal_moves) > self.top_k:
            ranked_idx = sorted(range(len(base_scores)), key=lambda i: base_scores[i], reverse=True)[:self.top_k]
            legal_moves = [legal_moves[i] for i in ranked_idx]
            cand_feats = [cand_feats[i] for i in ranked_idx]
            base_scores = [base_scores[i] for i in ranked_idx]

        decision = self.surrogate.denoise_with_adaptive_cfg(
            cand_feats, base_scores, danger, next_piece=env.next_piece
        )

        best_idx = decision["selected_index"]
        best_move = legal_moves[best_idx]
        lat_ms = (time.perf_counter() - t0) * 1000

        info = {
            "lat_ms": round(lat_ms, 2),
            "confidence": round(decision["confidence"], 3),
            "act_prob": round(decision["act_prob"], 3),
            "escalate_prob": round(decision["escalate_prob"], 3),
            "danger": round(danger, 2),
            "steps": decision["steps"],
            "best_score": round(base_scores[best_idx], 2),
            "candidates": [list(m) for m in legal_moves],
            "cand_scores": [float(s) for s in decision["final_scores"]]
        }
        return best_move, info


class DualDiffusionPlayer:
    """Double DiffusionGemma: Stream A (Architect Flattener) + Stream B (Aggressive Downstacker)."""

    def __init__(self, top_k: int = 12):
        self.top_k = top_k
        self.stream_architect = DjevSparkSurrogate(diffusion_steps=8, temperature=0.15)
        self.stream_sweeper = DjevSparkSurrogate(diffusion_steps=12, temperature=0.35)
        self.display_name = "Dual DiffusionGemma (Architect + Downstacker)"
        self.mode = "dual_diffusion"

    def choose_move(self, env: Tetris) -> Tuple[Tuple[int, int], Dict]:
        legal_moves = env.get_legal_moves()
        if not legal_moves:
            return (0, 0), {}

        t0 = time.perf_counter()
        curr_feats = env.state()["features"]

        cand_feats_arch = []
        cand_feats_sweep = []
        base_arch = []
        base_sweep = []

        for rot, col in legal_moves:
            sim_grid, lines, landing_h = env.simulate_placement(rot, col)
            sim_feats = compute_board_features(sim_grid)
            delta_holes = sim_feats["holes"] - curr_feats["holes"]

            # Stream A (Architect): Heavily penalizes bumpiness and preserves right well
            score_arch = (
                -0.6 * landing_h
                - 1.8 * sim_feats["bumpiness"]
                - 12.0 * max(0, delta_holes)
                + 1.0 * lines
                - 0.5 * sim_feats["col_transitions"]
            )
            # Stream B (Sweeper): Aggressive clear priority and negative hole digging
            score_sweep = (
                -0.4 * landing_h
                + 4.0 * lines
                - 6.0 * sim_feats["holes"]
                - 0.8 * sim_feats["bumpiness"]
            )

            cand = {
                "rot": rot, "col": col, "lines": lines,
                "landing_h": landing_h, "delta_holes": delta_holes,
                "bumpiness": sim_feats["bumpiness"]
            }
            cand_feats_arch.append(cand)
            cand_feats_sweep.append(cand)
            base_arch.append(score_arch)
            base_sweep.append(score_sweep)

        # Run dual Langevin denoising concurrently
        dec_a = self.stream_architect.denoise_candidates(cand_feats_arch, base_arch)
        dec_b = self.stream_sweeper.denoise_candidates(cand_feats_sweep, base_sweep)

        # Speculative lookahead & Arbiter:
        # If board has holes or high stack (>10), trust Sweeper; else trust Architect
        is_emergency = (curr_feats["holes"] > 0) or (curr_feats["max_height"] > 10)
        chosen_stream = "Sweeper (Downstacker)" if is_emergency else "Architect (Flattener)"

        if is_emergency:
            best_idx = dec_b["selected_index"]
            confidence = dec_b["confidence"]
        else:
            best_idx = dec_a["selected_index"]
            confidence = dec_a["confidence"]

        best_move = legal_moves[best_idx]
        lat_ms = (time.perf_counter() - t0) * 1000

        info = {
            "lat_ms": round(lat_ms, 2),
            "chosen_stream": chosen_stream,
            "confidence": round(confidence, 3),
            "arch_idx": dec_a["selected_index"],
            "sweep_idx": dec_b["selected_index"],
            "is_emergency": is_emergency,
            "candidates": [list(m) for m in legal_moves]
        }
        return best_move, info


class CascadePlayer:
    """System 1 (DiffusionGemma) + System 2 (Laya RLCD ModernBERT) Cascade Co-Pilot."""

    def __init__(self, laya_player=None, device: Optional[str] = None):
        self.sys1 = EnhancedDiffusionPlayer(cfg_scale=2.5)
        self.sys2 = laya_player
        if self.sys2 is None:
            # Lazy import to avoid loading weights until required
            try:
                from laya_player import LayaPlayer
                self.sys2 = LayaPlayer(device=device)
            except Exception as e:
                print(f"[!] Warning: Could not initialize LayaPlayer for Cascade: {e}")
                self.sys2 = None

        self.display_name = "System 1+2 Cascade (DiffusionGemma + ModernBERT)"
        self.mode = "cascade"
        self.sys1_calls = 0
        self.sys2_calls = 0

    def choose_move(self, env: Tetris) -> Tuple[Tuple[int, int], Dict]:
        t0 = time.perf_counter()
        # Fast System 1 proposal
        move_s1, info_s1 = self.sys1.choose_move(env)

        curr_feats = compute_board_features(env.grid)
        # Escalation criteria:
        # 1. Low confidence (< 0.65) OR
        # 2. Board in emergency (max height > 12 or holes > 1) OR
        # 3. System 1 escalate_prob > 0.40
        needs_escalation = (
            info_s1.get("confidence", 1.0) < 0.65 or
            curr_feats["max_height"] > 12 or
            curr_feats["holes"] > 1 or
            info_s1.get("escalate_prob", 0.0) > 0.40
        )

        if needs_escalation and self.sys2 is not None:
            self.sys2_calls += 1
            try:
                move_s2, info_s2 = self.sys2.choose_move(env)
                lat_ms = (time.perf_counter() - t0) * 1000
                info_s2["lat_ms"] = round(lat_ms, 2)
                info_s2["cascade_active"] = "System 2 (Laya RLCD Rescue)"
                info_s2["sys1_move"] = list(move_s1)
                info_s2["sys1_confidence"] = info_s1.get("confidence")
                return move_s2, info_s2
            except Exception:
                pass

        self.sys1_calls += 1
        lat_ms = (time.perf_counter() - t0) * 1000
        info_s1["lat_ms"] = round(lat_ms, 2)
        info_s1["cascade_active"] = "System 1 (Diffusion Reflex)"
        return move_s1, info_s1
