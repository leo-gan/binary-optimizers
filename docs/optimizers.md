# Optimizers (extracted)

Source: `experiments/Benchmarking Voting Optimizer vs STE on CIFAR-10.md`

This document summarizes the optimizer variants defined and discussed in the source note.

## Design goals (as stated in the note)

Across the versions in the note, the optimizer designs are motivated by:

- **Stability for binary weights**: reduce "flip-flapping", oscillations, and sensitivity to noisy per-batch gradients.
- **Less optimizer state**: reduce memory compared to Adam by using fewer auxiliary buffers.
- **Sign- and bit-centric updates**: prefer bounded, vote-like signals over unbounded gradient magnitudes.
- **Toward "logic-only" training**: explore bit-flip rules (stochastic or rank-based) that avoid floating-point-style accumulation in the update rule.

## STE Optimizer (SGD + clamp)

The note uses a "Classic STE" approach:

- **Core idea**: keep **latent float weights**, but use `sign()`-quantized weights during forward.
- **Update rule**: standard SGD update on the latent weights.
- **Stabilizer**: clamp weights to `[-1, 1]` after each step.

In the note this appears as `STEOptimizer(torch.optim.SGD)` which performs `super().step(...)` and then applies `p.data.clamp_(-1, 1)`.

### Why this exists

The note uses STE as a baseline to train binarized (sign) weights while still using standard backprop on float parameters.

### How it works (intuition)

- Forward uses `sign()` quantization (either by mutating weights in early code, or via a non-destructive proxy in later code).
- Backward propagates gradients to the underlying float parameters.
- Clamping keeps latent weights bounded so the training dynamics do not explode.

### Better than its ancestor

Compared to destructive "always sign the stored parameters" approaches, STE keeps trainable latent float weights while still producing sign weights for the forward pass.

### Pros

- Simple to implement (uses standard SGD step).
- Compatible with standard PyTorch training loops.

### Cons

- The note characterizes STE as **noisy** for binary weights; gradient spikes can cause unstable flips.
- Compared to sign/voting approaches, it can require more optimizer state if paired with Adam-like optimizers.

## Voting Optimizer (reference, non-fused)

A Python reference implementation in the note:

- **State**: per-parameter accumulator `A`.
- **Vote**: `batch_consensus = -sign(grad)`.
- **Accumulate**: `A += lr * batch_consensus`, then `A = clamp(A, -1, 1)`.
- **Binary weight**: `p.data = sign(A)`.

This version writes binary weights directly from the accumulator sign.

### Why this exists

The note introduces Voting to stabilize binary training by accumulating "votes" for the sign before flipping the weight, instead of reacting to each batch instantly.

### How it works (intuition)

- Each batch produces a discrete vote (`-sign(grad)`).
- Votes integrate into a bounded accumulator `A`.
- The weight becomes `sign(A)`, so it only flips when the accumulator crosses zero with enough consistency.

### Better than its ancestor

Relative to per-batch sign updates, the accumulator acts as a memory, reducing rapid back-and-forth flips.

### Pros

- The note expects smoother convergence by avoiding catastrophic oscillations.
- Bounded accumulator reduces unbounded drift.

### Cons

- In this reference form it overwrites weights as pure signs; it is not directly compatible with later "keep float latent weights" STE-style forward binarization.

## Voting Optimizer (STE-compatible accumulator + push)

In the note’s corrected CIFAR-10 setup ("v2"), Voting is rewritten to keep float storage compatible with STE-style forward binarization:

- **State**: per-parameter accumulator `A` clamped to `[-1, 1]`.
- **Vote**: `batch_consensus = -sign(grad)`.
- **Target sign**: `w_target = sign(A)`.
- **Push rule**: move the float weight toward `w_target` using `push_rate`:
  - `p.data += push_rate * (w_target - p.data)`
- **Bound**: clamp `p.data` to `[-1, 1]`.

### Why this was implemented

The note’s "v2" explains that earlier binarization logic caused problems. This variant is designed to keep Voting compatible with STE-style, non-destructive forward binarization by retaining float parameters.

### How it works (intuition)

- The accumulator `A` still decides a target sign `w_target = sign(A)`.
- Instead of overwriting weights with `sign(A)`, the optimizer **pushes** the float weight toward that sign by a fraction (`push_rate`).

### Better than its ancestor

Compared to the reference Voting optimizer, this version:

- keeps float weights (so STE-style proxies can be used in forward), and
- avoids hard overwrites that break "latent weight" training.

### Pros

- Maintains Voting’s "consensus" behavior while preserving float storage.
- Compatible with STE-style forward binarization that does not mutate parameters.

### Cons

- Introduces an extra hyperparameter (`push_rate`) that affects how abruptly weights move.

## Voting Optimizer (simple SignSGD)

Later sections introduce a simplified "Voting" optimizer described as "Your Idea / SignSGD":

- **Vote**: `vote = sign(grad)`.
- **Update**: `p.data -= lr * vote`.
- **Note in the source**: no clipping in this initial version (weights can drift).

The note explicitly analyzes two issues with this baseline:

- **Drift problem**: large magnitude shadow weights become hard to flip.
- **No momentum**: per-batch vote noise causes back-and-forth updates.

### Why this exists

The note frames this as a minimal "Voting"/SignSGD idea: ignore gradient magnitude, only keep direction, and reduce optimizer memory compared to Adam.

### How it works (intuition)

- Every parameter receives a discrete direction vote (`sign(grad)`).
- The weight is nudged by a fixed step size each batch.

### Better than its ancestor

Relative to Adam/variance-tracking optimizers, it uses less optimizer state (the note highlights memory savings as a key advantage).

### Pros

- Lower optimizer-state memory than Adam (as emphasized by the note).
- Bounded per-step update magnitude (step size does not scale with gradient magnitude).

### Cons

- Drift: without bounding, large-magnitude latent weights can become "stuck".
- Noise: without momentum, votes can alternate batch-to-batch.

## Momentum Voting Optimizer (Momentum SignSGD / Signum)

The note’s refinement adds a momentum buffer and clipping:

- **State**: `momentum_buffer` per parameter.
- **Momentum update**: `buf = momentum * buf + (1 - momentum) * grad`.
- **Vote**: `vote = sign(buf)`.
- **Update**: `p.data -= lr * vote`.
- **Bound**: `p.data.clamp_(...)` (the note uses bounds like `[-1.2, 1.2]` or `[-1.5, 1.5]`).
- **Optional**: weight decay variant appears in the memory-usage sections.

### Why this was implemented

The note explicitly motivates this as the fix for:

- the **drift** problem (by clipping), and
- the **amnesia/noise** problem (by adding momentum on votes).

### How it works (intuition)

- Momentum accumulates a smoothed vote direction across batches.
- The actual update uses only the sign of that momentum buffer.

### Better than its ancestor

Compared to simple SignSGD:

- momentum stabilizes the direction vote, and
- clipping keeps latent weights responsive so signs can still flip.

### Pros

- Smoother training dynamics than per-batch sign updates.
- Still uses only one auxiliary buffer (momentum), which the note highlights as a memory advantage vs Adam.

### Cons

- Requires careful tuning of momentum, learning rate, and clip bounds.

## Swarm Optimizer (population bit-flipping)

Introduced as a "biological" branch ("Swarm Descent") when weights are represented as a population of bits.

- **Applies to**: swarm parameters shaped like `[out, in, swarm_size]`.
- **Pressure**: `grad_pressure = grad.mean(dim=2)`.
- **Flip probability**: `probs = clamp(abs(grad_pressure) * recruit_rate, 0, 0.5)`.
- **Target**: `target = -sign(grad_pressure)`.
- **Flip rule**: flip bits that differ from target with probability `probs`.
- **Implementation note**: later fixed to handle non-swarm parameters (e.g. BatchNorm) by applying a standard SGD-style update for those.

### Why this exists

In the note, Swarm Descent is introduced as a different evolutionary branch: represent each logical weight as a population of binary agents (e.g. 32 bits) and learn by flipping bits, motivated by an "equal memory / equal information volume" comparison.

### How it works (intuition)

- Compute a per-weight "pressure" by averaging gradients over the swarm dimension.
- Convert pressure to a flip probability.
- Flip the subset of bits that disagree with the target direction, stochastically.

### Better than its ancestor

Compared to sign updates on a single float weight, the swarm representation provides an internal population that can change gradually via partial flips.

### Pros

- Directly operates on discrete bit populations.
- Naturally supports "partial" changes (only some agents flip) instead of all-or-nothing sign flips.

### Cons

- Requires parameter-shape-specific logic (the note shows failure when treating BatchNorm params as if they were swarm tensors).
- Higher raw parameter tensor size due to the extra swarm dimension.

## BitLogic Optimizer (stochastic comparator / "logic gate")

Used in the note’s "bit-physics" simulation framing:

- **Applies to**: swarm parameters (3D tensors).
- **Noise**: compare a random tensor against a flip probability.
- **Flip probability**: `abs(grad_pressure) * sensitivity`.
- **Flip rule**: flip bits that are not equal to `target_sign = -sign(grad_pressure)` when `noise < flip_probability`.
- **Non-swarm parameters**: updated with a simple sign-based step for thresholds.

### Why this exists

The note reframes optimization as a "logic gate" / comparator: use randomness versus an error signal to decide flips, aiming to avoid floating-point-style arithmetic in the update rule.

### How it works (intuition)

- Convert gradient pressure into a flip probability.
- Compare a random "noise" value to that probability.
- Flip bits when the comparator triggers and the bit disagrees with the desired direction.

### Better than its ancestor

This is a more explicitly "digital" restatement of stochastic flipping: the decision rule is cast as comparator logic instead of arithmetic updates.

### Pros

- Directly aligned with the note’s "stochastic comparator" framing.
- Simple conceptual rule: compare noise to pressure.

### Cons

- Still requires scaling/sensitivity choices; if gradients are tiny, flips can become vanishingly rare.

## RankBasedBitOptimizer (deterministic top-k flipping)

The note’s deterministic alternative:

- **Wrongness score**: `wrongness_score = grad * p.data`.
- **Rank and flip**: select top-`k` entries (based on `flip_rate`) and flip those bits.
- **Non-swarm parameters**: updated with a sign-based step.

The note motivates this as ensuring updates happen even when gradient magnitudes are tiny.

### Why this was implemented

The note introduces rank-based flipping to avoid "dead zones" where stochastic probabilities become effectively zero for small gradients.

### How it works (intuition)

- Compute a wrongness score per bit.
- Flip the top fraction of most-wrong bits every step.

### Better than its ancestor

Compared to probability-based flipping, it guarantees a minimum amount of adaptation each step.

### Pros

- Guaranteed updates even under small gradients.
- Deterministic and easier to reason about than stochastic flips.

### Cons

- Can introduce "thermal noise" if flip rate stays high (the note later discusses the need for annealing).

## AdaptiveBitOptimizer (layer-normalized stochastic flips)

A stochastic bit-flip optimizer that normalizes gradients per layer:

- **Pressure**: `grad_pressure = grad.mean(dim=2)`.
- **Layer scale**: `mean(abs(grad_pressure)) + eps`.
- **Normalized pressure**: `norm_pressure = grad_pressure / layer_scale`.
- **Flip probability**: `clamp(abs(norm_pressure) * base_flip_rate, 0, 1)`.
- **Target**: `target_sign = -sign(grad_pressure)`.
- **Flip rule**: stochastic flips where bit differs from target and `rand < flip_prob`.

### Why this was implemented

The note identifies issues with:

- extreme discontinuities from an all-or-nothing sign forward pass, and
- sensitivity becoming effectively zero when gradients are tiny.

This variant normalizes pressure per layer so flip probabilities remain meaningful.

### How it works (intuition)

- Normalize gradient pressure by a per-layer scale.
- Use the normalized magnitude to drive flip probabilities.

### Better than its ancestor

Compared to fixed sensitivity, this makes deep layers (with tiny gradients) able to flip bits.

### Pros

- Automatically rescales across layers.
- Reduces the need to hand-tune a global sensitivity.

### Cons

- Still stochastic; training can remain unstable depending on base flip rate and the broader architecture.

## MomentumRankOptimizer (momentum on wrongness + ranking) + annealing

A later refinement adds momentum to the wrongness score and schedules the flip rate:

- **State**: per-bit `wrongness_buffer`.
- **Current wrongness**: `current_wrongness = grad * p.data`.
- **Buffer update**: exponential moving average with `momentum`.
- **Rank and flip**: flip the top-`k` consistently-wrong bits.
- **Annealing**: the note includes an explicit exponential schedule for `flip_rate` from a high start rate to a lower end rate.

### Why this was implemented

The note frames this as the fix for "thermal noise": flipping too many bits forever prevents convergence.

### How it works (intuition)

- Momentum buffer accumulates "consistent wrongness".
- Ranking flips only the most consistently-wrong bits.
- Annealing reduces flip rate over epochs to transition from exploration to refinement.

### Better than its ancestor

Compared to plain rank-based flipping:

- momentum reduces reaction to one-off noisy batches, and
- annealing reduces destructive late-stage flipping.

### Pros

- Explicit convergence mechanism (cooling schedule).
- More stable than constant-rate top-k flipping.

### Cons

- More moving parts: momentum, flip-rate schedule, and associated hyperparameters.

## IntegrateFireOptimizer (accumulator threshold)

A "refuse to flip unless consistently wrong" strategy:

- **State**: per-bit accumulator.
- **Pressure**: `pressure = grad * p.data * 100.0` (as written in the note).
- **Integrate**: `acc = decay * acc + pressure`.
- **Fire**: flip when `acc > threshold`, then reset accumulator.
- **Non-swarm parameters**: updated via clamped SGD.

### Why this was implemented

The note introduces integrate-and-fire to prevent thrashing: bits only flip when wrongness accumulates past a threshold, analogous to a neuron firing after integrating input.

### How it works (intuition)

- Accumulator integrates pressure with leak (decay).
- When accumulator crosses threshold, the bit flips and the accumulator resets.

### Better than its ancestor

Compared to immediate flipping, it enforces persistence: a single noisy batch is less likely to trigger a flip.

### Pros

- Built-in hysteresis through the accumulator.
- Natural protection against rapid oscillations.

### Cons

- The note identifies a "whispering gradient" failure mode: if pressure is too small relative to threshold and leak, flips never happen.

## AdaptiveIntegrateFireOptimizer (self-tuning threshold)

Adds threshold adaptation based on observed flip rate:

- **Pressure**: computed as `grad * p.data` and then boosted (the note uses `* 1000.0`).
- **Integrate/fire**: as above.
- **Adaptive threshold**: per parameter-group adjustment based on whether flip rate is below/above a target range.

The note also discusses a failure mode where the adaptive threshold can "run away" after a spike.

### Why this was implemented

It aims to remove manual tuning of the firing threshold by adjusting thresholds to maintain a target flip rate.

### How it works (intuition)

- Measure flip rate per parameter group.
- Lower threshold when flips are too rare; raise threshold when flips are too frequent.

### Better than its ancestor

Compared to fixed-threshold integrate-and-fire, it adapts to different gradient scales automatically.

### Pros

- Auto-tuning behavior (no single fixed threshold for all conditions).

### Cons

- The note shows an instability where a spike can cause extreme threshold growth, effectively freezing learning.

## BoundedVoteOptimizer (bounded accumulator votes)

A later fix to avoid unbounded pressure from gradient magnitudes:

- **Vote**: `vote = sign(grad) * sign(p.data)` (bounded to `{-1, 0, +1}`).
- **Integrate**: `acc = decay * acc + vote`.
- **Fire**: flip if `acc > vote_threshold`.
- **Refractory reset**: accumulator set to `-threshold * 0.5` for fired bits (as written in the note), to reduce immediate re-flipping.

### Why this was implemented

The note’s "final fix" motivation is to prevent runaway dynamics caused by unbounded gradient magnitudes in the accumulator/threshold feedback loop.

### How it works (intuition)

- Replace pressure magnitude with a bounded democratic vote.
- Use a fixed threshold with a refractory reset to reduce immediate flip-back.

### Better than its ancestor

Compared to magnitude-based pressure (e.g. `grad * weight * 1000`), bounded votes prevent accumulator explosions and make thresholds physically interpretable as "number of net bad votes".

### Pros

- Bounded inputs to the accumulator reduce instability.
- Refractory reset reduces oscillatory flip-back.

### Cons

- Discards gradient magnitude information entirely; tuning the vote threshold and decay still matters.

