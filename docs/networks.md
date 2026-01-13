# Neural networks and layers (extracted)

Source: `experiments/Benchmarking Voting Optimizer vs STE on CIFAR-10.md`

This document summarizes the neural network architectures and building-block layers defined and discussed in the source note.

## Design goals (as stated in the note)

Across the versions in the note, the binary-network designs are motivated by:

- **Binary forward path**: use `sign()` or swarm-majority voting so effective weights are 1-bit.
- **Trainability**: preserve a gradient path (STE-style proxies or normalized swarm sums) so learning can progress.
- **Stability**: avoid destructive in-place binarization and reduce discontinuities that cause chaotic gradients.
- **Reduced reliance on floating-point ops**: explore replacing BatchNorm and float-style optimizer arithmetic with simpler thresholding/centering and bit-centric mechanisms.

## CIFAR-10: `SmallConvNet` (initial version)

The note defines a small convnet for CIFAR-10 with:

- `conv1`: `Conv2d(3, 32, 3, padding=1, bias=False)`
- `conv2`: `Conv2d(32, 64, 3, padding=1, bias=False)`
- `fc1`: `Linear(64*8*8, 256, bias=False)`
- `fc2`: `Linear(256, 10, bias=False)`
- Activations: `ReLU`
- Pooling: `max_pool2d(..., 2)` after each conv

Binarization behavior in this initial version:

- When `binary=True`, it calls `_binarize()` which **mutates** parameters in-place using `p.data.sign()`.

### Why this exists

This is the minimal CIFAR-10 baseline in the note: a tiny convnet with an option to binarize weights.

### How it works (intuition)

- Forward: uses standard convolution / pooling / ReLU layers.
- If `binary=True`, weights are forced to `{-1, +1}` before the forward computations by overwriting `p.data`.

### Better than its ancestor

As a starting point, it provides an easy way to test the effect of sign weights on a small model.

### Pros

- Very simple and explicit.
- Makes it obvious whether the network is using sign weights.

### Cons

- In-place mutation of `p.data` breaks the notion of latent float weights.
- Can interfere with optimizers expecting float-valued parameters and can destabilize training.

## CIFAR-10: `SmallConvNet` (STE-style non-destructive binarization)

In the note’s corrected version ("v2"), `SmallConvNet` is rewritten so that binarization is applied in forward without mutating parameters.

### `ste_binarize(w)` helper

The note defines:

- `w_b = w + (w.sign() - w).detach()`

So forward sees `sign(w)` but gradients flow to the underlying float `w`.

### Functional conv/linear forward

Instead of using module calls that read `module.weight` implicitly, the forward path uses `F.conv2d` / `F.linear` so it can explicitly provide either:

- binarized weights (`ste_binarize(self.<layer>.weight)`) when `binary=True`, or
- raw float weights when `binary=False`.

The architecture stays the same shape (two conv + two FC), but avoids in-place `p.data` mutation.

### Why this was implemented

The note states that "problems with binarization" were fixed by switching to a non-destructive STE-style binarization in forward. This is intended to keep gradients flowing to float weights while still performing binary inference.

### How it works (intuition)

- Compute binarized weights for forward only.
- Use the STE trick so backprop treats the binarization as identity for gradients.
- Use functional layers (`F.conv2d`, `F.linear`) to supply the chosen weights explicitly.

### Better than its ancestor

Compared to the initial `SmallConvNet`:

- avoids overwriting `p.data` with signs, and
- makes the forward binarization compatible with optimizers that update float weights.

### Pros

- Much better optimizer compatibility (float weights remain float).
- Clear separation between "latent storage" and "binary forward".

### Cons

- Slightly more complex forward path (functional ops + explicit weight selection).

## MNIST / MLP: `BitLinearSTE` + `create_model()`

The note introduces a 1-bit linear layer (STE) and builds a simple MNIST classifier.

### `BitLinearSTE` layer

- Stores float latent weights: `self.weight = nn.Parameter(...)`.
- Forward:
  - `w_q = sign(self.weight)`
  - `weight_proxy = self.weight + (w_q - self.weight).detach()`
  - `linear(x, weight_proxy)`

### Model (`create_model()`)

A sequential model:

- `Flatten()`
- `BitLinearSTE(28*28, hidden)`
- `BatchNorm1d(hidden)` (the note states BatchNorm is critical for convergence in this setup)
- `ReLU()`
- `BitLinearSTE(hidden, 10)`

Several sections use this same pattern with different hidden sizes (e.g. 128, 512).

### Why this exists

The note uses this as a fast-to-run reference setting (MNIST) to validate optimizer ideas and training stability for 1-bit linear layers.

### How it works (intuition)

- Each `BitLinearSTE` stores float weights but uses their sign in the forward pass.
- The STE proxy ensures gradients update the latent float weights.
- BatchNorm is described in the note as "critical" to make the 1-bit model converge.

### Better than its ancestor

Compared to using `sign(weight)` directly (with no STE proxy), the STE proxy preserves a gradient path.

### Pros

- Simple, small architecture useful for debugging optimizer behavior.
- Demonstrates the STE pattern cleanly.

### Cons

- Relies on floating-point BatchNorm (the note later attempts to remove this "crutch").

### Larger model (`create_large_model()`)

Used in the note’s memory measurement:

- `Flatten()`
- `BitLinearSTE(28*28, 1024)`
- `BatchNorm1d(1024)`
- `ReLU()`
- `BitLinearSTE(1024, 1024)`
- `BatchNorm1d(1024)`
- `ReLU()`
- `BitLinearSTE(1024, 10)`

### Why this exists

The note uses a larger model to make optimizer memory-state differences more measurable.

### Pros

- Makes memory/overhead differences easier to observe.

### Cons

- Less of a minimal debug setting; slower and heavier.

## Swarm-weight layers ("32 agents per weight")

The note introduces a swarm representation where a single logical weight is represented by a population of bits with `swarm_size=32`.

### Why this exists

The note motivates this as an "equal information volume" comparison:

- Standard: 1 float32 weight (32 bits) per connection.
- Swarm: 32 binary agents (32 bits) per connection.

The swarm representation is then used to explore bit-flip style learning.

### Pros

- Makes the bit-population concept explicit in the parameterization.

### Cons

- Adds an extra dimension to parameters, complicating both layers and optimizers.

## `BitSwarmLinear`

Defined for the "equal memory" comparison:

- Parameter `population` shaped `[out_features, in_features, swarm_size]`.
- Forward:
  - `swarm_sum = population.sum(dim=2)` (ranges `[-32, 32]`)
  - `w_eff = sign(swarm_sum)` with ties forced to `+1`
  - `weight_proxy = swarm_sum + (w_eff - swarm_sum).detach()`
  - `linear(x, weight_proxy)`

This yields an effective 1-bit weight (`w_eff`) but keeps a sum-based proxy for gradient flow.

### How it works (intuition)

- The population vote produces a majority-sign weight.
- The proxy uses a sum-like path so gradients can influence the underlying population.

### Pros

- Majority vote provides a built-in nonlinearity/discreteness.

### Cons

- Tie handling (`0 -> 1`) is an explicit design choice and can bias behavior.

## `BitSwarmLogicLinear` (bit-physics / logic framing)

Multiple variants appear throughout the note:

- A version that stores `population` as `int8` (values `{-1, 1}`) to simulate "raw registers" but notes PyTorch restrictions around gradients for non-float tensors.
- Variants that compute `swarm_sum` and then either:
  - use a **majority vote proxy** (STE-style: `w_norm + (w_eff - w_norm).detach()`), or
  - use an **analog/normalized output** `w_normalized = swarm_sum / swarm_size` directly ("Analog Sum, Binary Storage"), producing 33 discrete output levels.

### Why these variants appear

The note discusses failure modes when the forward path is too discontinuous:

- When the output is strictly `sign(swarm_sum)`, a single bit flip can change the effective output dramatically.

To address this, the note introduces an "analog sum" forward path where each bit flip changes the output by `1/swarm_size`.

### Pros

- Analog/normalized outputs provide smoother signal changes with still-binary storage.

### Cons

- Some variants attempt to store int8 populations, but the note hits framework constraints around gradients for non-float tensors.

## Normalization / homeostasis modules

The note replaces or complements BatchNorm with simpler thresholding/centering mechanisms in several sections.

### Why these exist

The note explicitly tries to remove BatchNorm (division/sqrt) and replace it with simpler counter-like mechanisms that keep neuron activity in a useful range.

## `HomeostaticThreshold`

A module that maintains a learnable `threshold` and a running statistic:

- Forward subtracts a per-feature threshold.
- During training, updates internal statistics and nudges threshold values to control activity.

The note presents variants including:

- activity-based (targeting a firing rate around 50%), and
- mean-matching behavior using a running mean.

### Pros

- Provides an adaptive offset to keep activations centered or maintain firing rates.

### Cons

- Still introduces stateful dynamics and hyperparameters; behavior depends on update rules.

## `SimpleCentering`

A minimal centering layer:

- Learnable `bias`.
- Forward: `x - bias`.

Presented as a simpler alternative to BatchNorm / homeostasis.

### Pros

- Extremely simple.

### Cons

- Lacks variance normalization; may be insufficient for stability compared to BatchNorm.

## `HomeostaticScaleShift`

A scale-and-shift normalization module:

- Parameters: `threshold` and `gain`.
- Uses `gain` initialized using `1/sqrt(fan_in)` (as written in the note).
- Forward: `out = (x - threshold) * gain`.
- During training, updates `threshold` based on batch mean with a configurable learning rate.

### Why this exists

This layer adds scaling (gain) to complement thresholding so the network can keep signals in a usable range without full BatchNorm.

### Pros

- More expressive than a pure centering layer.

### Cons

- Still uses float-valued parameters and floating-point arithmetic.

## Example "equal information volume" architectures

The note includes an experiment runner that uses the same layer sizes for:

- a float/STE model (using `BitLinearSTE`), and
- a swarm model (using `BitSwarmLinear` with `swarm_size=32`),

motivated by:

- Standard model: 1 FP32 weight (32 bits) per connection.
- Swarm model: 32 binary agents (32 bits) per connection.

### Pros

- Provides a controlled comparison between representations under a matched "bits per connection" framing.

### Cons

- Even if information volume matches, runtime/memory layout and training dynamics differ substantially between float and swarm representations.
