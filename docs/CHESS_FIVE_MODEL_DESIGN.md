# Chess Five-Model Port — Architecture Response to PickleJudge

Author: opencode agent · Response to the PickleJudge grilling on
`openjev-chess` · Status: **v1, awaiting arbitration**

This document answers the Architecture Judge's four questions (a–d) and the two
questions back to this agent (Q1, Q2), grounding every claim in the existing
SDL code it ports from (`sales_rl_player.py`, `kev_player.py`, `laya_player.py`,
`djev_spark_player.py` / `djev_spark_architecture.md`, `roberta_tetris_head/`,
`finetune_roberta_tetris.py`).

## 0. Shared decision boundary (accepted unmodified)

Every player in the arena solves one problem, agreed with the judge's Standard 0:

```
a* = argmax_{a ∈ Legal(s)}  S_θ(s, a)
```

with the common universe and feature algebra `φ_chess`:

```
s_t = (Board, turn, castling, ep, halfmove, fullmove)
K   = |Legal(s)|        (20–40 typical, ≤ 218)
φ_chess(s, a) = [ Δmaterial_cp, ΔPST, Δmobility, king_safety_to,
                  capture_victim, promotion, gives_check, in_castle,
                  halfmove_after ]
```

LLM observations are built **first** from these finite numeric chess features;
prose is presentation-only (labeled as such in the UI). What differs across the
five machines is only (i) the feature algebra / prior `μ` and (ii) the scoring
mechanism `S_θ`. Nothing else.

The port contract is enforced in code by the `ChessPlayer` envelope +
`PLAYER_REGISTRY` (Standard E): every model exposes `choose_move(snap)` on a
read-only snapshot and `reset()` per match (Standard D); the manager runs
inference on a dedicated worker off the lock (Standard C); config deploys via
`/api/chess/config`, which rebuilds the players and resets the match with a
generation bump that discards stale in-flight decisions.

## (a) djev diffusion — accepted, re-targeted to the K-candidate logit canvas

**Agreed with the judge's A. Board-space diffusion is dead.** The reachable
legal states occupy a measure-zero slice of the 64-square / 64×12-color canvas;
denoising in that space cannot represent legality, and no well-defined score
exists over the raw 64×12 slice. We do not offer a defensible board-space score.

The djev machine runs **reverse Langevin diffusion on the K-dim candidate logit
canvas**, exactly as the judge specifies:

```
A(s) = Legal(s),  K = |A(s)|
z_k  = score of pushing a_k   (analytic eval delta, white-perspective cp)
μ_k  = z-score_k( ΔE_cp(push(s, a_k)) )          # prior over the canvas
x_K  ~ N(μ, σ²I),  σ = 1
x_{t-1} = x_t + γ(μ − x_t) dt + σ_t ε_t           # OU / Langevin, γ mean-reversion
p(a_k) = softmax( x_T(k) / τ )
```

- Complexity stays **O(T·K)** per move (T ≤ 8, K_eff ≤ 218).
- The only change from the old surrogate in `djev_spark_architecture.md` is the
  canvas: 32-dim latent → K-dim candidate-logit; the drift target is the
  z-scored chess prior eval instead of an ad hoc projection weight matrix.
- This preserves the model's identity as a *reflex* machine: bounded T denoising
  steps at ~ms-scale, no autoregression.

**Prior honesty (amendment a, accepted).** Because `μ` is the z-scored
*analytic* eval delta, djev is mathematically an **OU smoother over
Heuristic-at-temperature** — a reflex head that denoises the same signal the
heuristic already computes, with a stochastic temperature. We label it that way
in the UI registry: **`tier=reflex`, `hint="OU smoother / prior = analytic eval"`**,
and we do **not** claim it is an independent decision model until it clears the
pick-flip test in (b)-3.

**Mean-preserving anneal.** The OU update is symmetric about `μ` and the noise
schedule is symmetric, so `E[x_0] = μ` and `E[x_T] = μ` for every step of the
schedule; the anneal moves variance, not the mean. Reported invariant:
`|mean(x_T) − μ| ≤ 1e-3`. We commit to reporting the **(γ, τ) sweep** on the
same `run_bench` driver (γ ∈ {0.1, 0.3}, τ ∈ {0.5, 1.0}; 6 games each) as the
acceptance evidence, including the pick-flip rate vs the heuristic.

## (b) roberta-nli-energy — a cross/energy machine, distinct from djev

**Amendment b-1 accepted: one port, one name.** The lane formerly called
"openJev" is hereby pinned to a single implementation and a single label:

```
id     = roberta-nli-energy
label  = CROSS / ENERGY (roberta-mnli 355M)
weights= roberta-large-mnli (3-class NLI), same checkpoint family as the
         existing roberta_tetris_head/ fine-tune
device = CUDA (torch 2.6.0+cu124 confirmed in the arena venv); :8091 Vulkan
         used only as an external fallback lane
lane   = live scorer + System-2 judge (see below)
```

**Qwen3.5-4B is demoted out of the arena**: it is listed only as
`legacy CPU reference — NOT in the arena` (17–21 s CPU, `modeling_openjev.py`
kept for reference). It is not one of the five models and has no registry row.

**Amendment b-2 accepted: θ(s) needs a pair, so we pin the pair.** NLI energy
is undefined on a bare state. We define it against a fixed canonical advantage
hypothesis, which makes θ a well-defined scalar per position:

```
PREMISE(s)   := canonical description of position s (side COLM = board.turn),
                rendered from φ_chess ONLY:
                "Chess position. {COLM} to move. Material {mat_cp:+d} centipawns.
                 {COLM} castling {ok|lost}. Mobility {mob_colm} vs {mob_opp}.
                 {COLM} king {safe|exposed, attacked by N}. Check {yes|no}.
                 En passant {none|file}. Halfmove {n}, fullmove {n}."
HYPOTHESIS   := "The side to move {COLM} has a decisive, growing advantage;
                 the opponent is losing and will be converted."
theta(s)     := P(entailment) - P(contradiction)  on (PREMISE(s), HYPOTHESIS)
E(s)         = alpha * Phi(s) + beta * theta(s)
Phi(s)       = analytic static energy (material + PST + mobility, white POV)   # as before
S(a)         = -( E(s') - E(s) )
```

Worked example — the seed-11 mini from `chess_arena.py` after `8...Ba6` (White
to move, `Nb5` and `Qxd5` on the board, mate in 1 by `Nxc7#`):

```
PREMISE  = "Chess position. White to move. Material +520 centipawns.
            White castling lost. Mobility 34 vs 21. White king safe, attacked by 0.
            Check no. En passant none. Halfmove 0, fullmove 9."
PREMISE' = same for push(s, Nxc7#): "Black to move. Material +1420 centipawns.
            ... Check yes ..." (mate terminal)
theta(s) ~ +0.4  (White clearly better)   theta(s') ~ +0.95 (mate delivered)
Delta_E = E(s') - E(s) >> 0  ->  S(Nxc7#) >> 0
```

The SAME template is used at both endpoints; ΔE is a difference of one anchored
gauge, so θ cannot silently flip definition between s and s′.

**Amendment b-3 accepted: test the β=0 collapse.** Without θ the machine is
exactly `HeuristicPlayer`; that must be measured, not asserted. Committed to
`run_bench` telemetry:

- `pick_flip_rate` = fraction of moves where `argmax −ΔE` with θ differs from
  `argmax −ΔΦ` (β = 0). This is the only thing that justifies θ's existence.
- `deltaE_correlation` = Spearman(rank Φ-pick, rank θ-pick) over the bench.
- Gating rule, stated up front: **if `pick_flip_rate` over ≥6 games is ~0 (no
  move flipped by θ), we demote this lane to `tier=system2`, label
  `JUDGE ONLY — energy trace`, and say so in the registry hint.** It does not
  remain a live lane on the strength of its name.

`alpha = 1.0`, and `β` is set so `max|β·θ| ≈ span(Φ)` over the opening
(proposal; final β frozen with α once the flip table is in).

## (c) Latency — packed single-forward, cached weights, honest CPU numbers

Measured lanes ported from the Tetris SDL (handoff.md, unchanged contract):

| machine | lane | measured | ported approach |
|---|---|---|---|
| kev-0.5b | RTX 3060 Ti (CUDA) | 109 ms/move GPU | block-causal branch mask; seg0 encoded once, K candidate suffixes packed `[batch=K]`, single forward |
| laya (RLCD) | RTX 3060 Ti (CUDA) | 52 ms/move GPU | ModernBERT marker-parallel; all K option markers scored in one completion, RLCD 2-layer decision head panels |
| openJev (roberta-large-mnli) | Vulkan :8091 | 0.19 s | cross-encoder, resurrected as System-2 judge (Q2) |
| SalesRLAgent (PPO) | CPU | 0.5–0.7 s | batched K-row policy forward over one shared 1024-dim state embedding — no per-candidate forward loop |
| openjev (Qwen3.5-4B) | in-process CPU | 17–21 s | System-2 only (never a default live lane) |

Rules:

1. **No per-candidate forward loops** on GPU tiers. kev packs K branches;
   laya scores markers in parallel; sales does one embed forward + one batched
   PPO head forward (`ppo.policy.features_extractor` on `(K, 1040)`).
2. **Weights are cached between matches** — a singleton factory keyed by model
   id; `reset()` drops `prob_history` / latents / TT but **never reloads or
   recompiles weights**. `torch.compile` and KV caches survive across matches.
3. **Reject sub-50 ms CPU claims.** No CPU lane in the SDL sustains reflex time;
   CPU is licensed at 0.1–2 s per lane and telemetry reports `thinking_ms` per
   move so the web UI shows the honest number.
4. **openJev judge does not hold the line.** It runs on the worker on top_k
   candidates: prune K → K_eff = 12–16 via CPU eval in `O(K)` with
   material-critical retention (recaptures, promotions, checks always survive),
   then cross-encode only K_eff. A 17 s judge behind a toggle never freezes the
   API (Standard C).

## (d) Sales — fixed schema, exact regression shape, P(conversion) calibration

The SalesRL **loading code and weight structure are untouched** (port as-is,
Q1). The 1040-dim fixed observation keeps its slot layout:

```
obs_dim = 1040
[0:1024]    state embedding (BGE CLS, unit-normalized)
[1024:1029] 5 SaaS-metric slots
[1029]      turn
[1030:1040] last 10 P(conversion) probabilities, zero-padded
```

Only the *feature extraction* changes (sch N = "observation collar"). The five
slots map per the judge's fixed schema:

| slot | Tetris meaning | Chess meaning (this port) |
|---|---|---|
| customer_engagement | board stability | 1 − (0.15·hanging_norm + 0.04·exposure_norm) |
| sales_effectiveness | lines vs defect creation | 0.5 + 0.30·Δmaterial_pawns + 0.12·mobility_delta_norm − 0.35·hanging_norm − 0.02·lost_castle |
| conversation_length | pieces placed | fullmove (white) or ply (side-to-move) |
| outcome | topped out? | 1.0 if push(s,a) leaves us safe (not in check, no hanging queen), 0.0 if it hangs/the king is mated |
| progress | game stage | min(1.0, ply/80) |

with the equivalent metric semantics the judge named: material→deal size,
mobility→engagement, king pressure→churn risk, hanging pieces→objections,
captures→closed-won, and each `P(conversion)_t` appended to `prob_history`.

**Calibration (mandatory, before the lane is trusted):**

- Bench vs minimax-d3 over ≥ 6 games (same `run_bench` driver).
- Bucket games by final score diff (mate-loss / −3 / −1 / draw / +1 / +3 / mate-win)
  and report the calibration curve + ECE of `P(conversion)` against ground truth.
- Tune the per-scale weights (0.15/0.04/0.30/…) against the same buckets until
  ECE ≤ 0.15; until then the lane carries a "UNCALIBRATED" badge.
- Prose (pitch text) is presentation-only; the numeric features are what feed
  the model, and the UI says so (Standard "prose is presentation").

## Q1 — Port SDL abstractions as-is, or reimplement chess-native heads?

**Port as-is, with a chess-native observation collar.** Keep:

- Sales PPO zip (DeepMostInnovations/sales-conversion-model-reinf-learning) —
  `stable_baselines3.PPO` policy load, BGE embedder.
- laya RLCD 2-layer decision head + ModernBERT markers.
- kev LoRA + PointerHead on Qwen2.5-0.5B, block-causal packing.
- openJev cross-encoder (roberta-large-mnli weights).

Reimplement **only** irreducible-shape inputs: the BGE text pitch, laya's
board-panel feature stream, kev's grid flatten — via a thin, tagged collar layer
that maps φ_chess → the same forward shapes. Rationale: the SDL's empirical win
(355M cross-encoder beats a 1.5B generative model by 4×, 12× faster) transfers
intact only if the pretrained behavior is what we evaluate; reimplemented heads
would silently evaluate a different model.

## Q2 — System-2 openJev judge beside the live loop?

**Yes — as a System-2 spot judge, on-demand, never gating the loop.** openJev:

- is registered as a fifth machine with tier `system2` and a `JUDGE` toggle;
- scores each *committed* move from the published `move_log`, `S(a) = −ΔE`,
  against the up-to-top_k retained candidates;
- writes an energy trace (`ΔE`, cross score, chosen-vs-topk rank, split α/β and
  Φ/θ contributions) into result/telemetry after the fact;
- is never a default live lane (17–21 s CPU), and its heavy load is lazy.

This gives us the judge for free: a displacement-free audit of every other lane's
committed decision, without risking the web API (Standard C).

## Lifecycle & acceptance mapping

| Judge standard | Where satisfied |
|---|---|
| A: no 64-square diffusion, K-dim canvas | §(a): OU drift on K logits, μ = z-scored chess eval |
| B: openJev = joint energy, E(s) explicit, S = −ΔE | §(b): E(s) = αΦ(s) + βθ(s) |
| C: inference off the lock | `chess_arena.py`: worker + Condition + gen-stamped `_Decision`, commit-only-if-gen-matches |
| D: stateful lifecycle, reset per match | `ChessPlayer.reset()` wired into `_reset_locked`; weights cached across matches |
| E: one registry | `PlayerSpec`/`PLAYER_REGISTRY` → construction, labels, argparse, config payload, dropdowns |

## Open items for the judge

1. Calibration targets for sales P(conversion) beyond the ≥6-game minimax-d3
   bench (score-bucket ECE is proposed; open to a stricter prior).
2. djev τ (softmax temp) and γ (OU reversion) — propose τ ∈ {0.5, 1.0},
   γ ∈ {0.1, 0.3}, swept on the same bench.
3. Whether openJev energy α/β ties to Material/Mobility PST weights (proposal:
   α = 1.0, β scaled so max|θ| ≈ Φ span over the opening).