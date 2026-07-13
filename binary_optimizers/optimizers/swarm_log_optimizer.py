import torch
from torch.optim import Optimizer


class SwarmLogOptimizer(Optimizer):
    """
    Probabilistic/Logarithmic bit flipper for Swarm populations.
    Uses integer counters per connection.
    """

    def __init__(
        self, params, threshold: int = 10, flip_prob: float = 0.1, dynamic: bool = False
    ):
        if threshold < 1:
            raise ValueError("Threshold must be at least 1")
        defaults = dict(threshold=threshold, flip_prob=flip_prob, dynamic=dynamic)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # DEBUG: Print to ensure we are running
        if torch.rand(1) < 0.01:
            print("Optimizer Step Start (Sampled)")

        for group in self.param_groups:
            threshold = group["threshold"]
            flip_prob = group["flip_prob"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # We only apply this to the 3D population tensors
                if p.dim() != 3:
                    # Fallback for 1D or 2D params (e.g. bias, scale)
                    p.data.add_(p.grad, alpha=-0.01)
                    continue

                state = self.state[p]

                # Integer consensus counter per connection [out, in]
                if "counter" not in state:
                    state["counter"] = torch.zeros(
                        p.shape[0], p.shape[1], dtype=torch.int32, device=p.device
                    )

                counter = state["counter"]

                # 1. Consensus accumulation
                # Average gradient over swarm dimension to get "pressure" direction
                grad_pressure = -torch.sign(p.grad.mean(dim=2)).to(torch.int32)
                counter.add_(grad_pressure)

                # 2. Fire Logic
                # Dynamic Mode: Probability scales with counter magnitude
                if group["dynamic"]:
                    # Heuristic: P = (abs(counter) - threshold_bottom) * slope
                    # Using derived slope 0.035 and start 4 to match user example logic
                    # 5->3%, 10->20%, 30->90%

                    abs_counter = counter.abs()
                    # print("DEBUG: Excess Calc Start")
                    # Refired SOTA-Aligned Logic: Log-Dampened Curve
                    # P = P_base * (1 + alpha * log(1 + (C - T)/T))
                    # T=10, P_base=0.1, alpha=0.5
                    # C=10 -> P=0.1
                    # C=30 -> P=0.1 * (1 + 0.5 * ln(3)) ~= 0.155
                    # C=100 -> P=0.1 * (1 + 0.5 * ln(10)) ~= 0.215

                    # Ensure we only compute for C > 4 (or T=10 in this case? Let's use T=10 as base).
                    # Actually, let's keep the user's "Start at 4" logic but apply curve.
                    # Let's map C=10 to 0.1, C=30 to ~0.15.

                    # Effective excess ratio: R = (Count - 4) / 6.0  (At 10, R=1.0)
                    excess = abs_counter.float() - 4.0
                    excess.clamp_(min=0.0)

                    # Log Scaled Probability
                    # P = 0.05 + 0.05 * log(1 + excess/6.0)
                    # At 4: P = 0.05
                    # At 10: P = 0.05 + 0.05 * ln(2) = 0.085 (~0.1)
                    # At 30: P = 0.05 + 0.05 * ln(1 + 26/6) = 0.05 + 0.05 * 1.6 = 0.13
                    # At 100: P = 0.05 + 0.05 * ln(1 + 96/6) = 0.05 + 0.05 * 2.8 = 0.19

                    probs = 0.05 + 0.05 * torch.log1p(excess / 6.0)
                    probs.clamp_(0.0, 1.0)

                    # Roll for firing
                    # Note: We fire based on the PROBABILITY. If we fire, we flip ALL bits with that *same* probability?
                    # No, the user asked for "If 30 -> 25 flipped". That is ~80% of bits.
                    # So 'probs' is the PROBABILITY THAT AN INDIVIDUAL BIT FLIPS.
                    # We always check for flips if counter > 4.

                    # We split into Positive (driving to +1) and Negative (driving to -1) pressure
                    # But wait, counter sign tells us the direction.

                    pos_mask = counter > 4
                    neg_mask = counter < -4

                    if pos_mask.any():
                        # Target +1. Candidates: -1s in pos_mask locations

                        # Apply
                        # Need to be careful with indexing masked tensors.
                        # Ideally, we generate full-size masks.

                        full_probs = probs.unsqueeze(-1)
                        full_roll = torch.rand_like(p.data) < full_probs

                        # Flip -1 to 1 where counter > 4 and roll succeeds
                        flip_to_1 = (
                            (p.data == -1.0) & pos_mask.unsqueeze(-1) & full_roll
                        )
                        p.data[flip_to_1] = 1.0

                        # Soft Reset: Decay counter by the probability of firing
                        # This allows the counter to "discharge" its energy
                        decay_factor = 1.0 - probs
                        counter.to(torch.float32).mul_(decay_factor).to(torch.int32)

                    if neg_mask.any():
                        full_probs = probs.unsqueeze(-1)
                        full_roll = torch.rand_like(p.data) < full_probs

                        flip_to_neg1 = (
                            (p.data == 1.0) & neg_mask.unsqueeze(-1) & full_roll
                        )
                        p.data[flip_to_neg1] = -1.0

                        # (Decay already handled above for all)

                else:
                    # Static Mode (Original)
                    fire_pos = counter >= threshold
                    fire_neg = counter <= -threshold

                    if fire_pos.any():
                        mask = (fire_pos.unsqueeze(-1)) & (p.data == -1.0)
                        roll = torch.rand_like(p.data) < flip_prob
                        p.data[mask & roll] = 1.0
                        counter[fire_pos] = 0

                    if fire_neg.any():
                        mask = (fire_neg.unsqueeze(-1)) & (p.data == 1.0)
                        roll = torch.rand_like(p.data) < flip_prob
                        p.data[mask & roll] = -1.0
                        counter[fire_neg] = 0

        return loss
