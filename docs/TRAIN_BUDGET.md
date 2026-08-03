# Train budget design (wall clock + epochs)

**Status:** active protocol `wall_epoch_budget_v1`  
**Code:** `binary_optimizers/training/budget.py`  
**Versioning:** re-runs use patch ids (`v0_2_1`, …) — see `docs/EXPERIMENT_VERSIONS.md`

This note records **why** we moved off epoch-only early stop, what defaults we
chose, and what we deliberately did **not** optimize for.

---

## 1. Problem

Atlas and comparison trains use **one fixed network and data recipe**, but
**cost per epoch varies a lot** with representation:

| Kind of run (observed, this machine) | ~sec / epoch | Notes |
|--------------------------------------|-------------:|-------|
| Small register / thin encodings | 6–25 s | e.g. n=3–16 fixed, exp_mant |
| Wide register (n=32–62) | ~10–20 s | more state, still moderate |
| Unary S=64–128 | ~30–60 s | scales with S |
| Unary S=256 | ~110–150 s | baseline in WP2 |
| Unary S=512–1024 | ~220–420 s | large population |

### Epoch-only budget is unfair

**A. Stall patience in epochs** (e.g. “stop after 5 epochs without gain”)

- Fast cell: 5 × 10 s ≈ **50 s** of no improvement → stop.  
- Slow cell: 5 × 120 s ≈ **10 min** of no improvement → keep training.  

So **slow representations get more wall compute after the best epoch** before
early stop. That confounds “encoding quality” with “how long we waited.”

**B. Hard max epochs only** (e.g. 80)

- Fast cell finishes 80 epochs in ~15–30 min.  
- Unary S=1024 would need ~80 × 400 s ≈ **9 hours**.  

Comparing “best after ≤80 epochs” mixes **different wall-time investments**.

**C. What we care about in this project**

Research stance (Plan): rank **representations**, not shave 0.3% with longer
polish. Fair ranking needs a **shared resource** that is comparable across cells.
On one machine, **wall-clock** is the practical shared resource (and a proxy for
compute when batch size and hardware are fixed).

---

## 2. Design goals

1. **Fairness across width/encoding cells** on the same host: same max wall
   (and same patience *time*) by default.  
2. **Still bound epoch count** so a buggy fast loop cannot spin forever, and
   so “epoch” remains a readable log unit.  
3. **Patience scales with the budget**, not a magic absolute that was tuned for
   one epoch length (old atlas `patience=5` was that kind of magic number).  
4. **Any strict test improvement** resets stall (`min_delta=0`) so we do not
   stop while `test > best` by a hair (see early-stop bug with `5e-4`).  
5. **Protocol change is versioned** (`v0_N` → `v0_N_1`) so legacy epoch-only
   numbers stay in the DB under parent ids.

Non-goals:

- Exact FLOP parity across devices.  
- Multi-seed statistical fairness.  
- Matching published “80 epoch” recipes for absolute SOTA.

---

## 3. Decisions (defaults)

Implemented in `TrainBudget` / CLI (`add_budget_args`):

| Knob | Default | Decision |
|------|---------|----------|
| `max_epochs` | **80** | Keep historical upper bound; safety + log readability. Rarely the binding limit for *slow* cells under the wall budget. |
| `max_wall_sec` | **1200** (20 min) | Primary fairness cap per **run** (one ln_mode / one width / one encoding cell). |
| `patience_frac` | **0.125** (12.5%) | Single fraction applied to **both** budgets. |
| → `patience_epochs` | **10** | `round(0.125 × 80)`. |
| → `patience_wall_sec` | **150 s** | `0.125 × 1200`. |
| `min_delta` | **0** | Any strict `test_acc > best` updates best and resets stall. |
| Stop rule | first of | `max_wall_sec` · `max_epochs` · epoch patience · wall patience |

### 3.1 Why 20 minutes wall (`max_wall_sec=1200`)?

Grounded in runs we already did (MNIST MLP, hidden=128, this hardware):

- Most **register / encoding** cells already peaked and stalled within **~2–12
  minutes** under epoch-only patience (often well under 1200 s).  
- **Unary S=256** baselines took **~20–30+ minutes** for a full early-stopped
  trajectory under epoch patience; 1200 s still allows ~8–10 epochs at ~120 s/ep,
  enough to reach the soft plateau region we care about for ranking, not full
  polish.  
- A full atlas has **many cells** (e.g. WP2 primary ≈ 11 cells). At 20 min
  each, worst case ~3–4 hours wall for the grid—usable for iteration without
  multi-day sweeps.  
- **1200 s is a policy choice, not a law.** Override with `--max-wall-sec` for
  polish or larger models (CIFAR / deeper nets will need higher budgets).

If 1200 s is too tight for a slow cell, accuracy may be **under-trained relative
to a long unary run in the legacy parent id**—that is acceptable for atlas
ranking under a declared protocol; document overrides in NOTES when used.

### 3.2 Why patience as a **fraction** of the budget?

Absolute patience (5 epochs, or “always 3 minutes”) couples to one regime:

- 5 epochs was OK for ~15–30 s/ep atlas work; it is **too short in wall time**
  for fast cells and **too long in wall time** for huge S if you only count
  epochs.  
- Fraction of **max_epochs** keeps “about an eighth of the allowed schedule”
  if wall is disabled (`--max-wall-sec 0`).  
- Fraction of **max_wall_sec** keeps “about an eighth of the allowed wall”
  without improvement—**same wall stall** for fast and slow epochs.

**Why 12.5% specifically?**

- Matches a readable pair: **10 / 80 epochs** and **150 / 1200 s**.  
- Stricter than old “patience 10 of 80” only in that wall stall is *also*
  enforced; looser than atlas “patience 5” on the epoch axis so small noise
  plateaus get a bit more room under the new dual rule.  
- Easy to change: one knob `--patience-frac 0.1` → 8 ep / 120 s wall.

Absolute epoch override remains: `--patience 15` sets epoch patience only;
wall patience still follows `patience_frac × max_wall_sec` unless we later add
a separate flag (not needed yet).

### 3.3 Why keep both wall patience **and** epoch patience?

- **Wall patience:** fairness across epoch lengths (primary fix).  
- **Epoch patience:** if wall limit is disabled, behavior stays defined; also
  stops a fast cell that is oscillating without any long wall stall if we only
  had a large wall patience (edge case).  
- Either can fire first; logs print `Stop: <reason>` (`patience_wall`,
  `patience_epochs`, `max_wall_sec`, `max_epochs`).

### 3.4 Why `min_delta=0`?

With `min_delta=5e-4`, a run could show `test=0.9349 > best=0.9347` and still
increment stall (and stop). That is wrong for “best so far” tracking. Noise
filtering belongs in multi-seed analysis later, not in discarding strict gains
on a single seed.

### 3.5 What “fair” means here (and what it does not)

| Fair under this protocol | Not claimed |
|--------------------------|-------------|
| Same wall budget and wall stall rule per cell | Same FLOPs on GPU vs CPU |
| Same epoch cap | Optimal HPO per encoding |
| Documented version id for re-runs | Continuity of absolute best vs parent id |

Parent runs (`v0_5_width_*`, `v0_6_encoding`, …) remain historical under old
ids; **do not** mix them into the same leaderboard without labeling protocol.

---

## 4. Worked examples (defaults)

Assume `max_wall_sec=1200`, `patience_wall=150`, `patience_epochs=10`,
`max_epochs=80`.

| Cell | ~s/ep | Epochs fit in 20 min | Wall stall ≈ | Epoch stall ≈ |
|------|------:|---------------------:|-------------:|--------------:|
| Fixed n=8 | 20 | ~60 | 150 s ≈ 7–8 ep | 10 ep ≈ 200 s |
| Encoding exp_mant | 15 | ~80 (epoch cap may hit first) | 150 s ≈ 10 ep | 10 ep |
| Unary S=256 | 130 | ~9 | 150 s ≈ 1 ep | 10 ep would be ~22 min (**wall stall usually binds first**) |
| Unary S=1024 | 400 | ~3 | 150 s &lt; 1 ep | Epoch patience almost irrelevant under wall |

Implication: under defaults, **large unary** is wall-limited; **thin register**
is often patience- or epoch-limited. That is intentional: both get ~20 min max,
and both stop after ~2.5 min without improvement on the wall axis.

---

## 5. Overrides (when to change defaults)

| Situation | Suggested override |
|-----------|-------------------|
| Polish one winning cell | `--max-wall-sec 3600 --patience-frac 0.2` or `--patience 20` |
| Smoke / CI | `--epochs 3 --max-wall-sec 120 --patience-frac 0.5` |
| CPU-only laptop, slower epochs | raise `--max-wall-sec` (e.g. 2400) rather than only epochs |
| Disable wall (epoch-only, legacy-like) | `--max-wall-sec 0` (patience_wall off; only epoch patience + max epochs) |
| CIFAR / deeper net (future WP3) | Revisit defaults; 1200 s will likely be too small—set in that experiment’s PROTOCOL |

---

## 6. Interaction with experiment versioning

Changing the train budget **is** a protocol change:

- New runs → **`v0_N_1`** (or compound `v0_5_1_width_*`, `v0_6_1_encoding`).  
- DuckDB: `runs.experiment` = new id; `notes` + `config` store parent and
  `train_protocol=wall_epoch_budget_v1`.  
- Legacy results stay under parent ids for reference only.

---

## 7. Open risks / follow-ups

1. **1200 s may under-train unary S≥512** relative to legacy long runs—rank
   unary baselines with that caveat, or raise wall for unary-only re-runs.  
2. **Machine speed** changes wall meaning (desktop GPU vs CI CPU). Prefer
   documenting host class in NOTES when publishing curves.  
3. Optional later: FLOP or step budgets if multi-device comparisons matter.  
4. Optional: separate `--patience-wall-sec` if fraction of wall and fraction of
   epochs should diverge.

---

## 8. Summary decision record

| Decision | Choice | Primary reason |
|----------|--------|----------------|
| Shared resource | Wall-clock | Epoch cost differs by representation |
| Default wall | 20 min / run | Fits MNIST atlas grids; covers most register peaks; bounds unary |
| Default epochs | 80 | Historical safety cap |
| Patience | 12.5% of wall **and** epochs | One knob; fair stall for fast/slow epochs |
| min_delta | 0 | Strict improvements always count |
| Versioning | Patch id (`_1`) | Isolate protocol from legacy numbers |

When in doubt: **same wall budget, same wall patience, declare the protocol id.**
