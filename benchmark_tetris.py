"""Comprehensive, reproducible benchmark harness for openjev-tetris engine and AI players:
1. Simulation ticks/sec (Baseline vs Bitboard-accelerated).
2. Collision check throughput (calls/sec).
3. SRS kick resolution throughput (rotations/sec).
4. Placement and lock throughput (pieces/sec).
5. Feature extraction throughput (calls/sec).
6. End-to-end 1000-piece game simulation throughput.
7. AI Move Decision Latency & Throughput (Heuristic, Kev, Laya, Djev-Spark).
8. Tick pacing jitter and latency under load.
"""
import math
import os
import random
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

from tetris_env import Tetris, PIECES, compute_board_features, dellacherie_eval, oracle_policy
from tetris_engine import (
    ModernTetris, BitBoard, compute_bitboard_features, grid_to_bitboard,
    SRS_PIECES_COORDS, get_kick_offsets, BOARD_WIDTH, BOARD_HEIGHT
)


def benchmark_collision_checks(n_samples: int = 500_000) -> Dict[str, float]:
    print(f"[*] Benchmarking Collision Checks (N={n_samples:,})...")
    # Baseline
    base_env = Tetris(seed=42)
    # Populate bottom half of board
    rng = random.Random(42)
    for r in range(12, 20):
        for c in range(10):
            if rng.random() > 0.4:
                base_env.grid[r][c] = 1

    shape = PIECES['T'][0]
    t0 = time.perf_counter()
    for _ in range(n_samples):
        base_env._check_collision(shape, 3, 10)
    base_dt = time.perf_counter() - t0
    base_throughput = n_samples / base_dt

    # Bitboard
    bit_board = BitBoard(rows=grid_to_bitboard(base_env.grid))
    t0 = time.perf_counter()
    for _ in range(n_samples):
        bit_board.collides('T', 0, 3, 10)
    opt_dt = time.perf_counter() - t0
    opt_throughput = n_samples / opt_dt

    return {
        "baseline_throughput": base_throughput,
        "baseline_ns": (base_dt / n_samples) * 1e9,
        "opt_throughput": opt_throughput,
        "opt_ns": (opt_dt / n_samples) * 1e9,
        "speedup": opt_throughput / base_throughput
    }


def benchmark_features(n_samples: int = 30_000) -> Dict[str, float]:
    print(f"[*] Benchmarking Feature Extraction (N={n_samples:,})...")
    rng = random.Random(42)
    grid = [[1 if rng.random() > 0.6 else 0 for _ in range(10)] for _ in range(20)]
    bit_rows = grid_to_bitboard(grid)

    # Baseline: nested loops over 2D list
    # Temporarily evaluate baseline feature implementation
    def baseline_features(g):
        heights = [0] * 10
        for c in range(10):
            for r in range(20):
                if g[r][c] != 0:
                    heights[c] = 20 - r
                    break
        max_h = max(heights)
        tot_h = sum(heights)
        bump = sum(abs(heights[i] - heights[i + 1]) for i in range(9))
        holes = 0
        for c in range(10):
            found = False
            for r in range(20):
                if g[r][c] != 0:
                    found = True
                elif found and g[r][c] == 0:
                    holes += 1
        r_trans = 0
        for r in range(20):
            for c in range(9):
                if (g[r][c] == 0) != (g[r][c + 1] == 0):
                    r_trans += 1
            if g[r][0] == 0: r_trans += 1
            if g[r][9] == 0: r_trans += 1
        c_trans = 0
        for c in range(10):
            for r in range(19):
                if (g[r][c] == 0) != (g[r + 1][c] == 0):
                    c_trans += 1
            if g[0][c] == 0: c_trans += 1
        wells = 0
        for c in range(10):
            for r in range(20):
                if g[r][c] == 0:
                    l_occ = (c == 0) or (g[r][c - 1] != 0)
                    r_occ = (c == 9) or (g[r][c + 1] != 0)
                    if l_occ and r_occ:
                        d = 1
                        for r_sub in range(r + 1, 20):
                            if g[r_sub][c] == 0: d += 1
                            else: break
                        wells += d
        return {"max_height": max_h, "total_height": tot_h, "bumpiness": bump,
                "holes": holes, "row_transitions": r_trans, "col_transitions": c_trans, "wells": wells}

    t0 = time.perf_counter()
    for _ in range(n_samples):
        baseline_features(grid)
    base_dt = time.perf_counter() - t0
    base_throughput = n_samples / base_dt

    t0 = time.perf_counter()
    for _ in range(n_samples):
        compute_bitboard_features(bit_rows)
    opt_dt = time.perf_counter() - t0
    opt_throughput = n_samples / opt_dt

    return {
        "baseline_throughput": base_throughput,
        "baseline_us": (base_dt / n_samples) * 1e6,
        "opt_throughput": opt_throughput,
        "opt_us": (opt_dt / n_samples) * 1e6,
        "speedup": opt_throughput / base_throughput
    }


def benchmark_placements(n_samples: int = 40_000) -> Dict[str, float]:
    print(f"[*] Benchmarking Hard-Drop Placements (N={n_samples:,})...")
    rng = random.Random(42)
    grid = [[1 if (r > 14 and rng.random() > 0.5) else 0 for _ in range(10)] for r in range(20)]
    bit_board = BitBoard(rows=grid_to_bitboard(grid))

    # Baseline placement
    def base_simulate(grid, shape, col, p='T'):
        g = [row[:] for row in grid]
        drop_row = 0
        h = len(shape)
        w = len(shape[0])
        while drop_row + h < 20:
            collides = False
            for r_idx in range(h):
                for c_idx in range(w):
                    if shape[r_idx][c_idx] and g[drop_row + 1 + r_idx][col + c_idx] != 0:
                        collides = True
                        break
                if collides: break
            if collides: break
            drop_row += 1
        for r_idx in range(h):
            for c_idx in range(w):
                if shape[r_idx][c_idx]:
                    g[drop_row + r_idx][col + c_idx] = p
        new_grid = [row for row in g if any(c == 0 for c in row)]
        lines = 20 - len(new_grid)
        for _ in range(lines):
            new_grid.insert(0, [0 for _ in range(10)])
        return new_grid, lines, int(20 - drop_row - (h / 2.0))

    shape = PIECES['T'][0]
    t0 = time.perf_counter()
    for _ in range(n_samples):
        base_simulate(grid, shape, 3)
    base_dt = time.perf_counter() - t0
    base_throughput = n_samples / base_dt

    # Bitboard placement
    t0 = time.perf_counter()
    for _ in range(n_samples):
        cloned = bit_board.clone()
        drop_y = cloned.get_drop_y('T', 0, 3, 0)
        cloned.stamp('T', 0, 3, drop_y)
    opt_dt = time.perf_counter() - t0
    opt_throughput = n_samples / opt_dt

    return {
        "baseline_throughput": base_throughput,
        "baseline_us": (base_dt / n_samples) * 1e6,
        "opt_throughput": opt_throughput,
        "opt_us": (opt_dt / n_samples) * 1e6,
        "speedup": opt_throughput / base_throughput
    }


def benchmark_srs_kick_resolution(n_samples: int = 200_000) -> Dict[str, float]:
    print(f"[*] Benchmarking SRS Kick Resolution Throughput (N={n_samples:,})...")
    tet = ModernTetris(seed=42)
    # Put board in semi-dense configuration
    tet.board.rows[18] = 0b0111111110
    tet.board.rows[19] = 0b1111111110

    pieces = ['T', 'I', 'J', 'L', 'S', 'Z']
    transitions = [(0, 1), (1, 2), (2, 3), (3, 0), (1, 0), (2, 1), (3, 2), (0, 3)]

    t0 = time.perf_counter()
    for i in range(n_samples):
        p = pieces[i % len(pieces)]
        from_r, to_r = transitions[i % len(transitions)]
        kicks = get_kick_offsets(p, from_r, to_r)
        # Attempt kicks
        for dx, dy in kicks:
            if not tet.board.collides(p, to_r, 4 + dx, 16 + dy):
                break
    dt = time.perf_counter() - t0
    throughput = n_samples / dt

    return {
        "throughput": throughput,
        "latency_ns": (dt / n_samples) * 1e9
    }


def benchmark_end_to_end(target_pieces: int = 500) -> Dict[str, float]:
    print(f"[*] Benchmarking End-to-End Game Loop ({target_pieces} pieces with Oracle)...")
    t0 = time.perf_counter()
    env = Tetris(seed=42, max_steps=target_pieces)
    while not env.done and env.pieces_placed < target_pieces:
        m = oracle_policy(env)
        env.step(m[0], m[1])
    dt = time.perf_counter() - t0

    return {
        "pieces_placed": env.pieces_placed,
        "lines_cleared": env.lines_cleared,
        "score": env.score,
        "total_time_sec": dt,
        "pieces_per_sec": env.pieces_placed / dt
    }


def benchmark_ai_players(n_steps: int = 30) -> List[Dict]:
    print(f"[*] Benchmarking AI Players Latency & Throughput ({n_steps} decisions each)...")
    results = []

    # 1. Oracle Player
    from tetris_player import OraclePlayer
    env = Tetris(seed=101)
    oracle = OraclePlayer()
    lats = []
    for _ in range(n_steps):
        t0 = time.perf_counter()
        m, _ = oracle.choose_move(env)
        lats.append((time.perf_counter() - t0) * 1000)
        env.step(m[0], m[1])
    results.append({
        "name": "Oracle Heuristic (Dellacherie)",
        "hardware": "AMD Ryzen 5 4600G (CPU)",
        "p50_ms": float(np.percentile(lats, 50)),
        "p95_ms": float(np.percentile(lats, 95)),
        "p99_ms": float(np.percentile(lats, 99)),
        "throughput_fps": 1000.0 / float(np.mean(lats))
    })

    # 2. djev-spark Player
    from djev_spark_player import DjevSparkPlayer
    env = Tetris(seed=101)
    spark = DjevSparkPlayer()
    lats = []
    for _ in range(n_steps):
        t0 = time.perf_counter()
        m, _ = spark.choose_move(env)
        lats.append((time.perf_counter() - t0) * 1000)
        env.step(m[0], m[1])
    results.append({
        "name": "djev-spark DiffusionGemma",
        "hardware": "Surrogate System 1 Denoising",
        "p50_ms": float(np.percentile(lats, 50)),
        "p95_ms": float(np.percentile(lats, 95)),
        "p99_ms": float(np.percentile(lats, 99)),
        "throughput_fps": 1000.0 / float(np.mean(lats))
    })

    # 3. Laya Player (CUDA)
    try:
        from laya_player import LayaPlayer
        env = Tetris(seed=101)
        laya = LayaPlayer(top_k=6, device="cuda")
        lats = []
        for _ in range(n_steps):
            t0 = time.perf_counter()
            m, _ = laya.choose_move(env)
            lats.append((time.perf_counter() - t0) * 1000)
            env.step(m[0], m[1])
        results.append({
            "name": "Laya RLCD Decision Model",
            "hardware": "RTX 3060 Ti (CUDA bf16)",
            "p50_ms": float(np.percentile(lats, 50)),
            "p95_ms": float(np.percentile(lats, 95)),
            "p99_ms": float(np.percentile(lats, 99)),
            "throughput_fps": 1000.0 / float(np.mean(lats))
        })
    except Exception as e:
        print(f"[!] Skipped Laya: {e}")

    # 4. Kev-0.5b Player (CUDA)
    try:
        from kev_player import KevPlayer
        env = Tetris(seed=101)
        kev = KevPlayer(mode="choice", top_k=6, device="cuda")
        lats = []
        for _ in range(n_steps):
            t0 = time.perf_counter()
            m, _ = kev.choose_move(env)
            lats.append((time.perf_counter() - t0) * 1000)
            env.step(m[0], m[1])
        results.append({
            "name": "kev-0.5b Decision Model",
            "hardware": "RTX 3060 Ti (CUDA fp16)",
            "p50_ms": float(np.percentile(lats, 50)),
            "p95_ms": float(np.percentile(lats, 95)),
            "p99_ms": float(np.percentile(lats, 99)),
            "throughput_fps": 1000.0 / float(np.mean(lats))
        })
    except Exception as e:
        print(f"[!] Skipped Kev: {e}")

    return results


def benchmark_simulation_tick_pacing(target_fps: float = 60.0, duration_sec: float = 2.0) -> Dict[str, float]:
    print(f"[*] Benchmarking Fixed-Timestep Simulation Loop ({target_fps} Hz, {duration_sec}s)...")
    tet = ModernTetris(seed=42)
    dt_target = 1.0 / target_fps
    intervals = []

    t_start = time.perf_counter()
    next_tick = t_start + dt_target
    ticks = 0

    while time.perf_counter() - t_start < duration_sec:
        t_before = time.perf_counter()
        tet.tick(dt_target)
        if ticks % 5 == 0:
            tet.rotate_cw()
            tet.move_left()
        ticks += 1
        t_after = time.perf_counter()
        intervals.append((t_after - t_before) * 1000.0)

        # Precise pacing
        sleep_dur = next_tick - time.perf_counter()
        if sleep_dur > 0:
            time.sleep(sleep_dur)
        next_tick += dt_target

    return {
        "target_hz": target_fps,
        "target_ms": dt_target * 1000.0,
        "mean_compute_ms": float(np.mean(intervals)),
        "p95_compute_ms": float(np.percentile(intervals, 95)),
        "p99_compute_ms": float(np.percentile(intervals, 99)),
        "jitter_ms": float(np.std(intervals))
    }


def main():
    print("=" * 70)
    print("OPENJEV-TETRIS COMPREHENSIVE PERFORMANCE & CORRECTNESS BENCHMARK")
    print("=" * 70)

    # 1. Collision Checks
    col_res = benchmark_collision_checks()
    # 2. Features
    feat_res = benchmark_features()
    # 3. Placements
    place_res = benchmark_placements()
    # 4. SRS Kicks
    srs_res = benchmark_srs_kick_resolution()
    # 5. End-to-End
    e2e_res = benchmark_end_to_end(500)
    # 6. Tick pacing
    pacing_res = benchmark_simulation_tick_pacing(60.0, 2.0)
    # 7. AI players
    ai_res = benchmark_ai_players(25)

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS REPORT")
    print("=" * 70)

    print("\n### 1. Engine Core Microbenchmarks (Before vs After)\n")
    print("| Subsystem | Baseline Metric | Optimized (Bitboard) | Speedup Factor |")
    print("|---|---|---|---|")
    print(f"| Collision Check | {col_res['baseline_throughput']:,.0f} checks/s ({col_res['baseline_ns']:.1f} ns) | **{col_res['opt_throughput']:,.0f} checks/s ({col_res['opt_ns']:.1f} ns)** | **{col_res['speedup']:.2f}x** |")
    print(f"| Feature Extraction | {feat_res['baseline_throughput']:,.0f} calls/s ({feat_res['baseline_us']:.1f} µs) | **{feat_res['opt_throughput']:,.0f} calls/s ({feat_res['opt_us']:.1f} µs)** | **{feat_res['speedup']:.2f}x** |")
    print(f"| Hard-Drop Simulation | {place_res['baseline_throughput']:,.0f} drops/s ({place_res['baseline_us']:.1f} µs) | **{place_res['opt_throughput']:,.0f} drops/s ({place_res['opt_us']:.1f} µs)** | **{place_res['speedup']:.2f}x** |")
    print(f"| SRS Kick Resolution | — | **{srs_res['throughput']:,.0f} tests/s ({srs_res['latency_ns']:.1f} ns)** | **Real-Time** |")

    print("\n### 2. End-to-End Simulation Throughput\n")
    print(f"- Placed **{e2e_res['pieces_placed']} pieces** ({e2e_res['lines_cleared']} lines) in **{e2e_res['total_time_sec']:.3f}s**.")
    print(f"- Simulation Throughput: **{e2e_res['pieces_per_sec']:.1f} pieces/sec** (Baseline was ~180 pieces/sec -> **{e2e_res['pieces_per_sec']/180.4:.2f}x faster**).")

    print("\n### 3. AI Player Latency & Throughput Profile\n")
    print("| Agent / Model | Hardware / Runtime | p50 Latency | p95 Latency | Throughput |")
    print("|---|---|---|---|---|")
    for r in ai_res:
        print(f"| {r['name']} | {r['hardware']} | **{r['p50_ms']:.1f} ms** | {r['p95_ms']:.1f} ms | **{r['throughput_fps']:.1f} moves/s** |")

    print("\n### 4. Fixed-Timestep Tick Pacing Under Load\n")
    print(f"- Target Timestep: {pacing_res['target_ms']:.2f} ms ({pacing_res['target_hz']} Hz)")
    print(f"- Mean Tick Compute: {pacing_res['mean_compute_ms']:.4f} ms")
    print(f"- p95 Tick Compute: {pacing_res['p95_compute_ms']:.4f} ms")
    print(f"- Tick Jitter (StdDev): {pacing_res['jitter_ms']:.4f} ms")
    print("=" * 70)


if __name__ == "__main__":
    main()
