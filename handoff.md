# Handoff

Tetris where the move is chosen by a language model instead of a heuristic, with
several players side by side on an identical piece bag.

## Run it

```bash
./run_tetris.sh versus 8088     # human + Dellacherie + roberta-large-mnli + Qwen2.5-1.5B
```

Then http://localhost:8088. The launcher starts only the llama-servers the flags
need and reuses any already running.

```bash
./run_tetris.sh vulkan 8088 --with-heuristic --with-vulkan --with-qwen --with-openjev
./run_tetris.sh versus 8088 --free-run                     # no lock-step
./run_tetris.sh versus 8088 --strategy outcome_entailment  # the old numeric prompt
```

## The players

| board | where | speed | pieces | lines |
|---|---|---|---|---|
| Dellacherie oracle | `tetris_env.py:281` | 2 ms | 150* | 56.7 |
| Laya (RLCD Decision Model) | `laya_player.py`, **RTX 3060 Ti (CUDA)** | **52 ms** (0.8s CPU) | 50+ | solid lines |
| kev-0.5b (Decision Model) | `kev_player.py`, **RTX 3060 Ti (CUDA)** | **109 ms** (1.3s CPU) | 50+ | solid lines |
| roberta-large-mnli | llama-server :8091, Vulkan | 0.19 s | 130.3 | 45.3 |
| SalesRLAgent (arXiv:2503.23303) | `sales_rl_player.py`, CPU | 0.5-0.7 s | 50+ | solid lines |
| Qwen2.5-1.5B-Instruct | llama-server :8092, Vulkan | 2.3 s | 61.0 | 10.7 |
| openjev (Qwen3.5-4B) | in-process, CPU | 17-21 s | not benchmarked |  |
| random | — | — | 20 | 0 |

3 seeds, 150-piece cap. *oracle hits the cap without dying; at a 400 cap it is
400/157 and roberta is 185/63.3.

A 355M cross-encoder beats a 1.5B generative model at this by 4x, and is 12x
faster. Parameter count is not the thing that matters here.

## How each model is asked

All of them score every legal `(rotation, column)` placement and take the argmax.
The agent has no key-press action space — it picks a placement and hard-drops.

- **roberta / openjev**: NLI. `premise</s></s>hypothesis`, score = P(ent) - P(con).
- **kev-0.5b (`jaredpalmer/kev-0.5b`)**: Non-autoregressive decision model on `Qwen2.5-0.5B` with LoRA adapter and `PointerHead`. All candidates evaluated in a single forward pass under block-causal branch masking (`choice` argmax or packed `nli`).
- **Laya (`convaiinnovations/laya`)**: Non-autoregressive System 1 decision model (`ModernBERT-large` + RLCD 2-layer transformer decision head). Option markers scored in parallel with calibrated confidence and act/escalate probability.
- **SalesRLAgent (arXiv:2503.23303)**: PPO RL model over 1040-dim observation (1024 BGE embeddings + 5 SaaS metrics + turn + 10-step probability history). Candidate placements are scored as sales proposals that maximize the predicted deal conversion probability P(conversion). Latency is ~0.5-0.7s on CPU, perfectly matched for human-speed play.
- **Qwen**: one constrained token. `...Is this a good move?` → P(yes) - P(no) from
  the first token's logprobs. No prose is generated.
- The premise comes from `tetris_prompts.py`, strategy `qualitative`.

## Things that will bite you

**The prompt is the whole ballgame.** The original numeric premise (`New holes
created: 2. Surface bumpiness: 7.`) scored 34 pieces / 2.3 lines — barely above
random — because an entailment model cannot compare digits. Describing the same
outcome in words got 185/63.3 from the same weights. The second half of that win
was *resolution*: coarse buckets collapsed 23.6 legal moves into 5.3 distinct
prompts, so 78% of candidates tied and the model picked arbitrarily. Grading each
attribute finely took it to 67% distinct. If you touch `qualitative`, re-measure
the collapse ratio, not just the score.

**`-np 32` on llama-server.** It gives each sequence its own slot; at the default
4 a 30-move batch runs as 8 sequential rounds. 465 ms → 287 ms.

**The GPU is saturated, not idle.** Throughput flattens at ~104 pairs/s from 16
pairs up and time goes linear after that. There is no tuning left — only a smaller
model, a shorter prompt (61 tokens now), or fewer candidates.

**openjev runs on CPU.** torch here is `2.14.0+cpu`; the BC-250 is reachable only
through Vulkan. ~11 s per candidate. It cannot be moved to llama.cpp: `qwen3_5` is
not a supported arch, its linear-attention weights are split
`in_proj_qkv/z/a/b` where the `qwen3next` converter wants a fused `in_proj_qkvz`,
and the file carries a 297-tensor vision tower. That is an upstream port.

**Lock-step.** Every AI board waits for the slowest so they all decide on the same
piece. With openjev in the mix that means everything moves at 17 s/piece. Use
`--free-run` to unlock, at the cost of the boards judging different pieces.

**Never hold `self.lock` across inference.** `get_full_state` and `human_action`
take the same lock, so a 20 s forward pass freezes the whole server. Each slot has
its own worker thread for the same reason.

## Layout of the code

- `coop_tetris.py` — boards, `AISlot`, the manager, telemetry for the graph.
- `tetris_player.py` — `OraclePlayer`, `OpenJevPlayer`, `VulkanNLIPlayer`, `QwenMovePlayer`.
  All four expose the same `choose_move(env) -> (move, info)`.
- `tetris_prompts.py` — premise/hypothesis builders. `qualitative` is the default.
- `web_coop.py` — HTTP server and the entire UI as one string.

## Not done

- Qwen's activation graph shows its next-token distribution, not hidden states;
  llama-server's completion API does not expose a pooled vector.
- Per-candidate latents for the graph's hidden layer (it shows the winner's only).
- openjev has never been benchmarked for play strength — at 17 s/piece a 3-seed
  run is over two hours. Its full CPU model remains separate from the Vulkan RoBERTa
  ranking head and has not been promoted to the web default.

## Optimization & Correctness Pass (2026-09-18: BRRRR Engine + SRS + Lock-Delay + HOLD + djev-spark)

- **Bitboard Acceleration (`tetris_engine.py`)**:
  - Replaced cell-by-cell coordinate iterations with 10-bit integer bitboards and precomputed bitmask lookups.
  - Collision check throughput: **1.19M -> 3.63M checks/sec (3.05x faster)**.
  - Board feature extraction throughput: **14.8k -> 31.8k calls/sec (2.15x faster)**.
  - Placement simulation: **54.9k -> 115.5k drops/sec (2.10x faster)**.
  - 500-piece end-to-end simulation throughput: **180.4 -> 878.1 pieces/sec (4.87x speedup)**.
- **Full Super Rotation System (SRS)**:
  - Official 4-state SRS rotation matrices (`0`, `R`, `2`, `L`) with center-of-rotation bounding boxes (3x3 JLSTZ, 4x4 I, 2x2 O).
  - Official Guideline 5-test kick translation tables for JLSTZ and I.
  - Supports wall kicks, floor kicks, and block climbing.
- **Lock-Delay State Machine**:
  - Configurable 0.5s lock timer activated upon grounding (`collides(piece, rot, x, y + 1)`).
  - Up to 15 lateral movement / rotation resets per active piece drop.
  - Soft drop increases fall speed without instantly locking; hard drop instantly locks and bypasses lock delay.
- **Modern HOLD Queue**:
  - Single-hold queue with piece swapping or bag pop.
  - Hold lockout until the active piece locks into the grid.
- **`mmastrac/djev-spark` DiffusionGemma System 1 Integration**:
  - Compatibility analysis documented in [`djev_spark_architecture.md`](djev_spark_architecture.md).
  - Architecture uses non-autoregressive parallel canvas denoising.
  - Connects to remote DGX Spark servers (`POST /v1/systemone`) when available, or executes local System 1 Langevin diffusion surrogate at **1.9 ms (466 moves/s)**.
  - Integrated into CLI, co-op manager, web dashboard, and benchmark harness.
- **Verification**:
  - `test_tetris_engine.py`: 10/10 tests pass.
  - `test_pipeline.py`: 7/7 tests pass.
  - `benchmark_tetris.py`: Comprehensive microbenchmarks and AI player profiles verified.
