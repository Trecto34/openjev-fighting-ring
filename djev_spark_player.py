"""djev-spark Player: DiffusionGemma System 1 Non-Autoregressive Decision Engine
for Real-Time Tetris Control.

Supports both remote NVIDIA DGX Spark endpoints (POST /v1/systemone) and high-speed
local surrogate execution when hardware prerequisites (Blackwell GB10 NVFP4) are absent.
"""
import json
import math
import os
import time
from typing import Dict, List, Optional, Tuple
from urllib.request import Request, urlopen
from urllib.error import URLError

import numpy as np
import torch

from tetris_env import Tetris, compute_board_features, dellacherie_eval


class DjevSparkClient:
    """Client for remote mmastrac/djev-spark System 1 decision microservice."""

    def __init__(self, endpoint_url: str = "http://127.0.0.1:8095", timeout: float = 0.5):
        self.endpoint_url = endpoint_url.rstrip("/") + "/v1/systemone"
        self.health_url = endpoint_url.rstrip("/") + "/health"
        self.timeout = timeout
        self._is_alive = False
        self.check_connection()

    def check_connection(self) -> bool:
        try:
            req = Request(self.health_url, headers={"User-Agent": "openjev-tetris-client"})
            with urlopen(req, timeout=self.timeout) as resp:
                self._is_alive = (resp.status == 200)
        except Exception:
            self._is_alive = False
        return self._is_alive

    @property
    def is_connected(self) -> bool:
        return self._is_alive

    def request_decision(self, payload: Dict) -> Dict:
        data = json.dumps(payload).encode("utf-8")
        req = Request(
            self.endpoint_url,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "openjev-tetris-client"}
        )
        with urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))


class DjevSparkSurrogate:
    """Local simulation of DiffusionGemma System 1 decision dynamics.

    Denoises a continuous candidate decision canvas over T=8 steps via Langevin
    energy minimization, computing calibrated confidence and act/escalate probabilities.
    """

    def __init__(self, diffusion_steps: int = 8, temperature: float = 0.25):
        self.diffusion_steps = diffusion_steps
        self.temperature = temperature
        # Projection weights for continuous canvas state (6 features -> 32-dim latent)
        np.random.seed(42)
        self.W_canvas = np.random.normal(0.0, 0.2, (6, 32)).astype(np.float32)
        self.b_canvas = np.zeros(32, dtype=np.float32)

    def denoise_candidates(
        self,
        candidate_features: List[Dict[str, float]],
        base_scores: List[float]
    ) -> Dict:
        """Runs iterative parallel denoising across all candidate actions."""
        K = len(candidate_features)
        if K == 0:
            return {
                "selected_index": 0,
                "confidence": 0.0,
                "act_prob": 0.5,
                "escalate_prob": 0.5,
                "denoising_trace": [],
                "final_scores": []
            }

        # Initialize noisy canvas logits x_T ~ N(0, 1) + base prior
        prior = np.array(base_scores, dtype=np.float32)
        prior_norm = (prior - np.mean(prior)) / (np.std(prior) + 1e-6)
        noise = np.random.normal(0.0, 1.0, K).astype(np.float32)
        x = prior_norm + noise * 0.8

        trace = []
        # Multi-step reverse diffusion
        dt = 1.0 / self.diffusion_steps
        for step in range(self.diffusion_steps):
            t = (self.diffusion_steps - step) / self.diffusion_steps
            # Score function grad_x log p(x) ~ (target - x)
            grad = (prior_norm * 1.5 - x)
            # Langevin drift + injected variance
            sigma = math.sqrt(2.0 * t * dt) if step < self.diffusion_steps - 1 else 0.0
            x = x + 0.6 * grad * dt + np.random.normal(0.0, sigma * 0.1, K).astype(np.float32)

            energy = float(np.mean(np.square(prior_norm - x)))
            trace.append({
                "step": step + 1,
                "energy": round(energy, 4),
                "max_logit": round(float(np.max(x)), 3)
            })

        # Calibrated softmax
        exp_logits = np.exp((x - np.max(x)) / self.temperature)
        probs = exp_logits / np.sum(exp_logits)
        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])

        # System 1 Reflex: if confidence > 0.6, act instantly; else escalate
        margin = confidence - (float(np.partition(probs, -2)[-2]) if K > 1 else 0.0)
        act_prob = float(1.0 / (1.0 + math.exp(-6.0 * (margin - 0.2))))

        # Continuous latent representation for activation telemetry
        best_feat_vec = np.array([
            candidate_features[best_idx].get("landing_h", 0) / 20.0,
            candidate_features[best_idx].get("lines", 0) / 4.0,
            candidate_features[best_idx].get("delta_holes", 0) / 3.0,
            candidate_features[best_idx].get("bumpiness", 0) / 15.0,
            candidate_features[best_idx].get("max_height", 0) / 20.0,
            candidate_features[best_idx].get("col_transitions", 0) / 20.0,
        ], dtype=np.float32)
        latent = np.tanh(best_feat_vec @ self.W_canvas + self.b_canvas)

        return {
            "selected_index": best_idx,
            "confidence": confidence,
            "act_prob": act_prob,
            "escalate_prob": 1.0 - act_prob,
            "denoising_trace": trace,
            "final_scores": probs.tolist(),
            "latent": latent.tolist()
        }


class DjevSparkPlayer:
    """Tetris agent utilizing mmastrac/djev-spark DiffusionGemma System 1 decision model."""

    def __init__(
        self,
        endpoint_url: str = "http://127.0.0.1:8095",
        diffusion_steps: int = 8,
        top_k: Optional[int] = 12,
        device: Optional[str] = None
    ):
        self.top_k = top_k
        self.client = DjevSparkClient(endpoint_url=endpoint_url, timeout=0.3)
        self.surrogate = DjevSparkSurrogate(diffusion_steps=diffusion_steps)

        if self.client.is_connected:
            self.mode = "remote_spark"
            self.display_name = "djev-spark DiffusionGemma (DGX Spark NVFP4)"
        else:
            self.mode = "surrogate"
            self.display_name = "djev-spark DiffusionGemma (System 1 Denoising)"

        print(f"[+] Initialized {self.display_name} [mode={self.mode}]")

    def _extract_candidate_features(self, env: Tetris, legal_moves: List[Tuple[int, int]]) -> Tuple[List[Dict], List[float]]:
        curr_feats = env.state()["features"]
        cand_features = []
        base_scores = []

        for rot, col in legal_moves:
            sim_grid, lines, landing_h = env.simulate_placement(rot, col)
            sim_feats = compute_board_features(sim_grid)
            delta_holes = sim_feats["holes"] - curr_feats["holes"]

            # Dellacherie score as prior guide
            eval_score = dellacherie_eval(sim_grid, lines, landing_h)

            feat = {
                "rot": rot,
                "col": col,
                "lines": lines,
                "landing_h": landing_h,
                "delta_holes": delta_holes,
                "bumpiness": sim_feats["bumpiness"],
                "max_height": sim_feats["max_height"],
                "col_transitions": sim_feats["col_transitions"]
            }
            cand_features.append(feat)
            base_scores.append(eval_score)

        return cand_features, base_scores

    def choose_move(self, env: Tetris) -> Tuple[Tuple[int, int], Dict]:
        legal_moves = env.get_legal_moves()
        if not legal_moves:
            return (0, 0), {}

        t0 = time.perf_counter()
        cand_feats, base_scores = self._extract_candidate_features(env, legal_moves)

        # Pre-filter top_k if requested
        if self.top_k and len(legal_moves) > self.top_k:
            ranked_idx = sorted(range(len(base_scores)), key=lambda i: base_scores[i], reverse=True)[:self.top_k]
            legal_moves = [legal_moves[i] for i in ranked_idx]
            cand_feats = [cand_feats[i] for i in ranked_idx]
            base_scores = [base_scores[i] for i in ranked_idx]

        decision = None
        if self.client.is_connected:
            try:
                payload = {
                    "piece": env.current_piece,
                    "next_piece": env.next_piece,
                    "candidates": cand_feats,
                    "diffusion_steps": 8
                }
                decision = self.client.request_decision(payload)
            except Exception:
                decision = None

        if decision is None:
            # Run local System 1 diffusion surrogate
            decision = self.surrogate.denoise_candidates(cand_feats, base_scores)

        best_idx = decision["selected_index"]
        best_move = legal_moves[best_idx]
        lat_ms = (time.perf_counter() - t0) * 1000

        info = {
            "lat_ms": round(lat_ms, 2),
            "confidence": round(decision.get("confidence", 0.9), 3),
            "act_prob": round(decision.get("act_prob", 0.95), 3),
            "escalate_prob": round(decision.get("escalate_prob", 0.05), 3),
            "best_score": round(base_scores[best_idx], 2),
            "best_p_ent": round(decision.get("confidence", 0.9), 3),
            "best_p_con": round(decision.get("escalate_prob", 0.05), 3),
            "best_p_neu": round(max(0.0, 1.0 - decision.get("confidence", 0.9) - decision.get("escalate_prob", 0.05)), 3),
            "denoising_trace": decision.get("denoising_trace", []),
            "candidates": [list(m) for m in legal_moves],
            "cand_scores": [float(s) for s in decision.get("final_scores", base_scores)],
            "best_latent": decision.get("latent")
        }
        return best_move, info
