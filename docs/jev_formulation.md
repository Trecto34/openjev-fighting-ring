# JEV Mathematical Formulation: Five Decision Paradigms

This section gives the formal model of each controller wired into `openjev-tetris` for the
JEV (zero-shot / Jev-inspired) Tetris decision problem. All five agents solve the *same*
abstract task — rank the legal placements of the active tetromino — but they realize the
ranking function with fundamentally different probabilistic machines: sequential token
autoregression, pairwise cross-encoding, masked parallel branch decoding, calibrated
option scoring, and continuous-denoiser diffusion.

---

## 0. Common Problem Setting

Let the game state at step $t$ be

$$
s_t = \left( B_t,\ \pi_t,\ n_t \right)
$$

where $B_t \in \{0,1\}^{10 \times 20}$ is the board bitmask, $\pi_t$ is the active
tetromino type, and $n_t$ is the next piece. From $\pi_t$ and $B_t$ we enumerate the set
of legal placements

$$
\mathcal{A}(s_t) = \left\{ a = (r, c) \right\} \subseteq \{0,1,2,3\} \times \{1,\dots,10\}
$$

with $|\mathcal{A}(s_t)| = K_t$ typically $10 \le K_t \le 34$. Every candidate is dropped
to its terminal lock position with the Super Rotation System, producing a feature vector

$$
f_a = \phi\!\left(\operatorname{sim}(B_t, r, c)\right)
= \left[\, \text{lines},\ \text{landing\_h},\ \Delta\text{holes},\ \text{bumpiness},\ \max_h,\ \dots \,\right]
$$

A prompt builder $\psi$ renders $f_a$ into text $x_a = \operatorname{Premise}(f_a)$ (and, for
NLI, a paired hypothesis $h_a$). The move is selected by the uniform decision rule

$$
\boxed{\; m_t^\star \;=\; \arg\max_{a \in \mathcal{A}(s_t)} S_\theta(s_t, a) \;}
$$

Every approach differs **only** in the scoring operator $S_\theta$, the training
objective that produced $\theta$, and the computational cost of evaluating all $K_t$
candidates. In one table:

| id | Engine | Params | Candidates per decision | Latency (measured) |
|---|---|---|---|---|
| (1) | Qwen2.5-1.5B-Instruct, constrained 1-token decode | 1.5B | $K$ serial decodes | 2.3 s GPU |
| (2) | Qwen3.5-4B NLI cross-encoder (`openjev`) | 4B | $K$ paired forwards, batched | 17–21 s CPU |
| (3) | Qwen2.5-0.5B + LoRA + PointerHead (`kev-0.5b`) | 0.5B | $1$ packed forward (branch-masked) | 1.3 s CPU / 109 ms GPU |
| (4) | ModernBERT-large + RLCD head (`laya`) | 421M | $1$ forward (parallel markers) | 0.8 s CPU / 52 ms GPU |
| (5) | DiffusionGemma System 1 (`djev-spark`) | 26B | $T$ reverse steps over a $K$-vector | 1.2–1.9 ms surrogate |

---

## 1. Autoregressive Token Generation $(N)$

**Model.** A causal decoder defines a next-token distribution via the chain rule over the
prompt $\tau_a$ (system + premise + query) of length $N_\tau$:

$$
p_\theta(\tau_a, y_{1:T}) = \prod_{j=1}^{N_\tau} p_\theta(\tau_j \mid \tau_{<j}) \cdot
\prod_{t=1}^{T} p_\theta(y_t \mid \tau_a, y_{<t})
$$

where each factor is the softmax over the vocabulary $\mathcal{V}$ of a linear readout on
the causal stack:

$$
p_\theta(v \mid \,\cdot\,) = \frac{\exp\!\big( e_v^{\top} h_{\,L}^{(t)} \big)}
{\sum_{u \in \mathcal{V}} \exp\!\big( e_u^{\top} h_{\,L}^{(t)} \big)},
\qquad
h^{(t)} \text{ attends only to } h^{(t-1)}, \dots, h^{(1)} \text{ (causal mask)}
$$

**Decision score.** The player (`QwenMovePlayer`) constrains the first generated token and
never produces prose ($T = 1$). Scoring a candidate is a sum over the *yes/no* token
classes of the vocabulary:

$$
S_{\text{AR}}(a) \;=\; P_\theta\big(\mathcal{V}^{+} \mid \tau_a\big) - P_\theta\big(\mathcal{V}^{-} \mid \tau_a\big)
\;=\; \sum_{v \in \mathcal{V}^{+}} p_\theta(v \mid \tau_a) - \sum_{v \in \mathcal{V}^{-}} p_\theta(v \mid \tau_a)
$$

with $\mathcal{V}^{+} = \{\text{yes},\text{Yes},\text{YES}\}$ and $\mathcal{V}^{-}$ the
corresponding negatives, read from the top-$k$ logprobs.

**Complexity.** Decoding is *sequential in the token position*: the output token cannot
exist until every prefix token has been materialized, and every attention head at step $t$
reads the full prefix. Per candidate the cost is

$$
C^{(1)} = O\!\Big( N_\tau D^2 + \underbrace{N_\tau^2 D}_{\text{prefill self-attn}} +
\underbrace{N_\tau D^2}_{\text{1 output head}} \Big)
\qquad
(\times K\ \text{candidates})
$$

and the wall-clock is dominated by the *memory-bounded sequential* pass, not FLOPs: KV
entries must be read and refreshed for all $N_\tau$ prefix positions. This is the intrinsic
$O(N)$ bottleneck of the JEV problem: **one constrained token already costs a full prefix
decode, and no decision is available until the last token of the prompt is attended to** —
there is no shortcut in a causal architecture.

**Measured.** 61 pieces / 10.7 lines on 150-piece capped games; ≈ 2.3 s per move on Vulkan
(ulama 1.5B Q4). Batched prompts help only through KV reuse (`cache_prompt`), never through
parallelism of the decision itself.

---

## 2. NLI Cross-Encoder: $P(\text{ENT}) - P(\text{CON})$

**Model.** The JEV reference formulation (`OpenJevCrossEncoder` on `Qwen3.5-4B`) treats each
candidate as an inference pair.

$$
x_a = \text{``Premise: } p_a \ \backslash \text{n Hypothesis: } h_a\text{''}
$$

A *bidirectional* (non-causal) backbone encodes the concatenated pair; the pooled state is
the final hidden vector of the last non-pad token, $\ell_a = h^{(L)}_{m_a}$ with
$m_a = \sum_j \mathbb{1}[\text{pad}_j = 0] - 1$:

$$
\mathbf{p}_a = \operatorname{softmax}\!\big( W_s\, \ell_a \big) \in \Delta^2,
\qquad
\mathbf{p}_a = \big[\, \hat p_{\text{CON}},\ \hat p_{\text{ENT}},\ \hat p_{\text{NEU}} \,\big]
$$

**Decision score.** The net-entailment objective:

$$
\boxed{\; S_{\text{NLI}}(a) = P_\theta(\text{entailment} \mid x_a) - P_\theta(\text{contradiction} \mid x_a) \; }
$$

**Training objective.** Cross-entropy over the three NLI classes,

$$
\mathcal{L}_{\text{NLI}} = -\frac{1}{N} \sum_{a} \log \hat p_{y_a},
\qquad
y_a \in \{\text{contradiction},\, \text{entailment},\, \text{neutral}\}
$$

optionally topped by a soft-BCE `LatentMLPHead` on the frozen latent $\ell_a$:

$$
\mathcal{L}_{\text{head}} = \mathbb{E}_{x} \Big[
 w(y)\, \mathrm{BCE}\!\big( \mathrm{sigmoid}(g(\ell_a)),\ y(1-\varepsilon) + (1-y)\varepsilon \big) \Big],
\qquad
w(1) = \tfrac{1-p}{p}
$$

**Complexity.** Each pair is *one* full self-encoding of the concatenation — full attention
allows premise to contradict hypothesis in both directions, which is exactly the signal the
score needs. For $L_a \approx 60$ tokens,

$$
C^{(2)} = O\!\Big( L_a^2 D (\text{layers}) \Big)\ \text{per pair}
\qquad
(\times K,\ \text{batched at } b \le 32)
$$

There is **no sequential generation**: the answer is a projection of one forward pass. The
cost is paid per candidate because the cross-encoder has no way to share representations
across hypotheses — the pairing genuinely changes every attention value.

**Measured.** The single strongest JEV result (185 pieces / 63.3 lines oracle-capped) comes
from *prompt linguistics*, not model scale: the numeric `outcome_entailment` premise scored
34/2.3 (barely above random) because an entailment model cannot compare digits, while the
`qualitative` wording ("It settles low down on the stack…") re-excites lexical semantics.
Latency: ≈ 11 s/candidate, ≈ 17–21 s/move on CPU at 4B.

---

## 3. Block-Causal Branch Masking + PointerHead (`kev-0.5b`)

**Motivation.** (1) and (2) pay $K$ times the model cost per decision. The Jev decision
model instead packs *all* candidates into one sequence and gates attention so every branch
denoises in parallel — a single forward pass.

**Packed record.** A shared state segment (segment 0) is followed by one branch per
question $q$, each `[<q> instr <opt_1> </opt_1> ... <opt_K> <decide>]` in segment $q > 0$.
Each token carries a segment id $\operatorname{seg}_i$. Attention is the *block-causal*
mask

$$
M_{ij} = \begin{cases} 1 & j \le i \ \wedge\ \big(\operatorname{seg}_j = 0 \ \vee\ \operatorname{seg}_j = \operatorname{seg}_i\big) \\ 0 & \text{otherwise} \end{cases}
$$

so each branch attends to the shared state and to itself but **never to sibling branches**;
the state attends causally to itself only. Logits are the masked attention with
$-\infty$ clamping:

$$
h^{(l)}_i = \operatorname{Attn}\!\left( h^{(l-1)}_i;\ \big\{ h^{(l-1)}_j : M_{ij} = 1 \big\} \right),
\qquad
\log M_{ij} \to -\infty \Rightarrow \exp \to 0
$$

**PointerHead.** After the single forward pass, the head reads the hidden state of the
`<decide>` token $d$ and the $K$ `<opt>` tokens, projects to a shared $d_p = 256$ space, and
scores each option with a scaled dot product:

$$
q = W_q^\top h_d + b_q, \qquad
k_i = W_k^\top h_{\text{opt}_i} + b_k,
\qquad
\ell(a_i) = \frac{\langle k_i,\ q \rangle}{\sqrt{d_p}}
$$

**Decision score.** Choice mode applies one softmax over the candidates; NLI mode applies the
same head over the three MNLI criteria options per question and takes the net-entailment gap:

$$
\text{Choice:}\quad
p(a_i) = \frac{\exp\!\big( \ell(a_i) \big)}{\sum_{j=1}^{K} \exp\!\big( \ell(a_j) \big)},
\qquad S^{(3)}_{\text{choice}}(a_i) = p(a_i)
$$

$$
\text{NLI:}\quad
\mathbf{p}_i = \operatorname{softmax}\big( \ell(\cdot) \big)_{\{\text{CON,ENT,NEU}\}},
\qquad S^{(3)}_{\text{nli}}(a_i) = \hat p_{\text{ENT},i} - \hat p_{\text{CON},i}
$$

**Complexity.** The entire decision is **one** autoregressive-in-form but branch-parallel
forward pass of total length $L = L_{\text{state}} + K\,L_{\text{branch}}$:

$$
C^{(3)} = O\!\Big( L_{\text{state}}^2 + K\,L_{\text{branch}}^2 \Big)
\;\approx\; O\!\big( L_{\text{state}}^2 \big) + O\!\big( K L_b^2 \big)
\ \text{vs.}\ C^{(2)} = O(K L^2)
$$

Because each branch sees only its own options (block mask), branch length $L_b$ stays tiny
and the $K$ branches execute as independent parallel work items on the GPU. The model is a
frozen `Qwen2.5-0.5B` with a rank-16 LoRA on all 24 projections, trained as a decision
model (a JEV reconstruction), so FLOPs and memory are ≈ 8× smaller than the 4B cross-encoder.

**Measured.** ≈ 1.3 s/move CPU, 109 ms GPU; README cites ≈ 18× faster than openjev 4B at
comparable play quality (50+ pieces, solid line clears).

---

## 4. ModernBERT-large + RLCD Decision Head (`laya`)

**Model.** A 395M bidirectional encoder (`ModernBERT-large`) with a 2-layer transformer
decision head on top (421M total). Options arrive as inline markers; one parallel forward
encodes the state and every option with full bidirectional attention.

**Calibrated decision.** The head emits a probability vector over the $K$ option markers, an
action probability, and a confidence. The move rule is the argmax with a System-1 gate:

$$
p(a_i \mid s) = \frac{\exp\!\big( z_i / \tau_i \big)}{\sum_{j} \exp\!\big( z_j / \tau_j \big)},
\qquad
m_t^\star = \arg\max_i p(a_i \mid s_t),
\qquad
p_{\text{act}} = \sigma\!\big(\gamma\,(\hat p^{(1)} - \hat p^{(2)} - \delta)\big)
$$

where $\hat p^{(1)} - \hat p^{(2)}$ is the winner-margin; escalation (handoff to a slow
System-2) is chosen when $p_{\text{act}}$ falls below threshold.

**RLCD training objective.** Reinforcement Learning for *Calibrated* Decisions composes a
policy-gradient term with a calibration penalty, so the reported confidence is
trustworthy — critical for the act/escalate gate. With reward $R$ and policy
$\pi_\theta$:

$$
J(\theta) = \mathbb{E}_{a \sim \pi_\theta}\big[ R(s,a) \big],
\qquad
\nabla_\theta J = \mathbb{E}\Big[ R(s,a)\, \nabla_\theta \log \pi_\theta(a \mid s) \Big]
$$

$$
\mathcal{L}_{\text{RLCD}} = -\mathbb{E}\big[ R\, \log \pi_\theta \big]
+ \underbrace{\lambda\, \mathrm{ECE}(p, y)}_{\text{calibration}},
\qquad
\mathrm{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N}\, \big| \operatorname{acc}(B_m) - \operatorname{conf}(B_m) \big|
$$

(binning $B_m$ by predicted confidence; equivalently a Brier term
$\operatorname{Brier} = \frac{1}{N}\sum_i \| p_i - y_i \|^2$).

**Complexity.** Identical profile to (2) on the *candidate dimension* — one bidirectional
pass over state + all markers —

$$
C^{(4)} = O\!\Big( \big( L_{\text{state}} + K L_{\text{opt}} \big)^2 D \Big)\ \text{single forward}
$$

but with a 395M backbone rather than 4B, giving a ~5–20× FLOP reduction over the openjev
backbone and the fastest *learned* decision in this list.

**Measured.** ≈ 800 ms/move CPU, 52 ms GPU (RTX 3060 Ti); 50+ pieces / solid lines on the
150-piece cap.

---

## 5. Continuous Langevin Diffusion (`DiffusionGemma` / `djev-spark`)

**Model.** DiffusionGemma reverses the Gaussian corruption of a *continuous canvas*. The
forward process is a variance-preserving chain,

$$
q(x_t \mid x_0) = \mathcal{N}\!\big( x_t;\ \sqrt{\bar\alpha_t}\, x_0,\ (1 - \bar\alpha_t)\, I \big)
$$

and generation solves the reverse-in-time SDE

$$
\mathrm{d}x = \Big[ f(x,t) - g(t)^2 \nabla_x \log p_t(x) \Big] \mathrm{d}t + g(t)\, \mathrm{d}\bar w
$$

where the score is replaced by a learned denoiser $\hat p_\theta \to \epsilon_\theta(x_t, t)$.
The System-1 adaptation denoises an entire *decision canvas* — one coordinate per candidate —
in parallel, with no vocabulary, no KV cache, and no token order.

**Decision score (parameters $\to$ canvas).** Candidate features $f_a$ and a cheap prior
$\mu_a$ (Dellacherie evaluation $\mu_a = \operatorname{dellacherie}(B_t, a)$, z-scored) define
an initialization and a score target:

$$
x_T \sim \mathcal{N}\!\big( \mu,\ \sigma_0^2 I \big),
\qquad
\nabla_x \log p(x) \approx \gamma\,(\mu - x),\ \gamma = 1.5
$$

**Reverse diffusion (Euler–Maruyama / Langevin step).** With $\Delta t = 1/T$,
$T = 8$, and a decaying noise schedule $\sigma_t = \sqrt{2\, t\, \Delta t}$:

$$
\boxed{\; x_{t-1} \;=\; x_t \;+\; \underbrace{\alpha\,\gamma\,(\mu - x_t)\,\Delta t}_{\text{score drift}\ \eta=0.6}
\;+\; \underbrace{\sigma_t\,\varepsilon_t}_{z \sim \mathcal{N}(0,I),\ \varepsilon \sim \mathcal{N}(0,\beta^2 I)} \;}
$$

The drift is a harmonic-attractor force pulling the logit toward the prior; the injected
noise is annealed to zero only in the last step (simulated annealing / critical
temperature). Energy traces are the residual score norm

$$
E_t = \frac{1}{K} \big\| \mu - x_t \big\|_2^2
$$

**Readout.** The final canvas is rendered with a *calibrated* (low-temperature) softmax
over candidates, and the act/escalate margin logistic mirrors Section 4:

$$
p_i = \frac{\exp\!\big( (x_i - \max_j x_j)/\tau \big)}{\sum_j \exp\!\big( (x_j - \max_j x_j)/\tau \big)},
\qquad \tau = 0.25,
\qquad
p_{\text{act}} = \sigma\!\big( 6\,(\hat p^{(1)} - \hat p^{(2)} - 0.2) \big)
$$

**Complexity.** The **decision gradient** is over a $K$-vector, not a token sequence, so

$$
C^{(5)} = O\big( T \cdot K \big)
$$

the only approach whose per-decision cost is linear in the *number of candidates* with a
constant that does not scale the model depth at all. It sidesteps the $O(N)$ autoregressive
law and the per-pair $O(L^2)$ law simultaneously.

**Measured.** Local surrogate: ≈ 1.2–1.9 ms/move (~466 moves/s); native DGX Spark target is
the 5–20 ms NVFP4 nerve budget. Requires NVIDIA GB10 / NVFP4 tensors for the 26B
weights; on this Ampere workstation it runs only as the surrogate.

---

## 6. Cross-Approach Synthesis

### 6.1 The three scaling laws being compared

Let $K$ = candidates, $L$ = prompt length, $N$ = generated tokens, $D$ = hidden width,
$T$ = diffusion steps.

| Law | (1) AR | (2) NLI | (3) Branch mask | (4) RLCD | (5) Diffusion |
|---|---|---|---|---|---|
| Decision FLOPs | $O(K \cdot N_\tau^2 D)$ | $O(K \cdot L^2 D)$ | $O(L_s^2 + K L_b^2)$ | $O((L_s{+}K L_o)^2 D)$ | $O(T K)$ |
| Sequential chain | token position | none | none (branch-parallel) | none | $T$ steps |
| Attention sharing across candidates | none | none | shared state only | shared context | none needed |
| Measured latency | 2.3 s | 17–21 s | 1.3 s | 0.8 s | 1.9 ms |

(1) and (2) both scale **linearly in $K$ with the *quadratic* sequence factor** — the former
in $N_\tau^2$, the latter in $L^2$ — and the pairwise cross-encoder cannot reuse per-candidate
work. (3) collapses $K$ to a constant-ish factor by packing candidates into one sequence with
immune sibling branches; (4) does the same with a tiny bidirectional backbone plus a
calibration head; (5) replaces language span with a $K$-dimensional vector, reducing the whole
decision to a handful of scalar updates.

### 6.2 What transfers from language, what does not

- **Semantics beats arithmetic.** The single largest score change in the JEV benchmark came
  from re-rendering the premise in words (34 → 185 pieces) — evidence that these models are
  lexically-grounded entailment machines, and the score $P(\text{ENT}) - P(\text{CON})$ is the
  right *decision algebra* for a cross-encoder, but digits are the wrong *alphabet*.
- **One-shot decision heads dominate generation.** (3) and (4) reach play strength with 8–13%
  of the parameters of (2) at ~15–20% of the latency, purely by removing autoregression and
  batching the decision.
- **Diffusion trades fidelity for reaction.** The continuous Langevin form is the only one
  with deterministic bounded latency (fixed $T$), the exact property a System-1 reflex
  requires; its price is that the "semantics" live only in the prior $\mu$, so its quality
  ceiling equals the quality of the heuristic prior.
- **Calibration is not implied.** (4) and (5) explicitly ship calibrated confidences and a
  *margin-gated* act/escalate switch; (1)–(3) return softmax masses that are not calibrated,
  which is why the classic NLI gap score works better than the raw probability magnitude.

### 6.3 JEV recommendation

None of the five dominates on every axis, so the choice is a corner-of-the-triangle decision
over *quality*, *latency*, and *simplicity*:

| Constraint | Choose | Reason |
|---|---|---|
| Max cones play strength, wall-clock irrelevant | (2) NLI cross-encoder | the only formulation demonstrated at 185/63.3 with semantic (non-numeric) premises |
| Learned single-shot, human-move-pace (~1 s) | (3) kev branch-mask or (4) RLCD | one forward pass, 8–13% of the params of (2), same ballpark play quality |
| Hard reflex budget (~ms), prior available | (5) Langevin diffusion | bounded $O(TK)$ latency independent of model depth or sequence length |
| Cheap diagnostic, no vision of a learned semantics | (1) AR 1-token | useful as a baseline that constrains generation to a decision |

The two pareto-frontier formulations for the JEV decision contract are the
block-causal PointerHead family (3) among *learned* models — it buys the parallel-branch
mask without the per-pair $O(L^2)$ tax — and the continuous Langevin canvas (5) when the
latency budget dominates and a strong prior (Dellacherie) can seed the denoiser. The NLI
score $(2)$, $P(\text{ENT}) - P(\text{CON})$, remains the semantic reference target that both
learned heads are measured against; note it can also be replayed inside (3)'s `nli` mode at
0.5B for a $O(L_b^2)$ rather than $O(L_a^2)$ per candidate.

---

## References

- `modeling_openjev.py` — `OpenJevCrossEncoder`, soft-BCE latent head (NLI formulation).
- `tetris_player.py` — `OpenJevPlayer`, `QwenMovePlayer` (autoregressive formulation),
  `TetrisRankHead` (listwise, temperature-softmax targets).
- `kev_player.py` — `PointerHead`, `branch_mask_batch`, packed record encoding.
- `laya_player.py` — RLCD decision agent wrapper.
- `djev_spark_player.py`, `djev_spark_architecture.md` — System 1 Langevin surrogate.
- `tetris_prompts.py` — premise/hypothesis builders (`qualitative`, `qualitative_fine`,
  `outcome_entailment`).