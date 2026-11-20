# Reproduction Guide

## Three Hard Conditions (All Mandatory)

1. **The same ckpt file** — not "the same seed", not "the same configuration".
   We measured: retraining once with the same seed / same data / same number of steps gives a routing state whose L1 distance from the previous one is **0.10**,
   the same order as the intervention effect we want to measure (0.06), or larger.
2. **The same window length** — not "roughly the same".
   A 96-step run and an 85-step run **cannot be compared in the same coordinate system**.
3. **The intervention magnitude is computed on the fly from the current baseline's own load** — not reused from another baseline's value.

> We distilled these three after stepping in the pits ourselves. Omit them and reproduction will fail.

---

## Minimal Reproduction Steps

### Step 0: Environment

```bash
pip install torch numpy matplotlib
```

> Code blocks below are translated for readability; the actual scripts in this
> repository have Chinese comments.

### Step 1: Make a baseline ckpt

Train N steps with any MoE training code (we used 130 steps) and save the ckpt.
**Record its `expert_load` (the load shares of the 8 experts).**

### Step 2: Copy it into two

```bash
cp -r base_ckpt A_ckpt
cp -r base_ckpt B_ckpt
```

### Step 3: Compute the intervention magnitude (A only)

```python
import torch

E = 8
gamma = 0.2                                   # intervention strength
frac = torch.tensor([...])                     # the 8-dim load recorded in step 1, summing to 1
delta = gamma * torch.sign(1.0 / E - frac)     # one number per expert
# write into A's router bias
```

**Key: `delta` must be computed from this baseline's own `frac`.** Using another baseline's `delta` is fitting someone else's solution to your problem.

> **Source of the method**: this `sign(1/E - frac)` update is DeepSeek-V3's aux-loss-free load balancing
> (arXiv:2412.19437). This document only measures the metric; it does not propose the method.

### Step 4: Train both for the same number of steps

```
A: start from A_ckpt, train M steps (we used 96~128 steps)
B: start from B_ckpt, train M steps
```

⚠️ **Both must use the same M.** With different windows the results are not comparable.

### Step 5: Compute the metrics

```bash
python metrics.py --cards ./cards
```

or directly:

```python
from metrics import cv, l1

cv_A  = cv(load_A)          # coefficient of variation
dist  = l1(load_A, load_B)  # share distance, 0~2
```

### Step 6: Read the result

| Symptom | Meaning |
|---|---|
| `l1` about 0.06, `cv` differs by only 1.5% | **matches our observation**: cv barely sees it, L1 does |
| `l1` about 1e-4 | the intervention did not take effect (check whether the bias really got written into the model) |
| `l1` about 0.10 but `cv` also differs a lot | check whether the two groups' ckpts are the same file |

---

## How to Confirm That "the Intervention Really Took Effect"

**Do not trust "I set it". Read it back.**

```python
print("bias_abs_mean =", model.router.bias.abs().mean().item())
# expected: A is approximately gamma, B is 0
```

We stepped in this twice:
1. the forward pass with `router_mode=none` **does not read the bias at all**, and `set_router_mode` resets it
2. when restoring from a ckpt, `load_state_dict` **overwrites the bias back to 0**

**In both cases "the log showed the setting succeeded, but in the actual model it was 0".**

---

## Plotting

```bash
python make_figures.py --data data_for_figures.json --out figures
```

The structure of `data_for_figures.json` is described in the comments inside the file.


---

## Automatic Check: check_pair.py

**Run this before any paired experiment.**

```bash
python check_pair.py --csv data/summary.csv      # check grouped by group
python check_pair.py --a card_A.json --b card_B.json   # check a single pair
```

It checks six items:

| # | Check | Why |
|---|---|---|
| 1 | the two groups share `start_step` | otherwise they are not the same baseline ckpt |
| 2 | the two groups share `window_steps` | metrics from different windows are not comparable |
| 3 | the two groups share `seed` | rules out differences in data order |
| 4 | the intervention group has `bias_mean > 0`, and a control with `bias_mean = 0` exists | otherwise either the "intervention" or the "control" is missing |
| 5 | `L1` is not 0 | if it is 0, the two groups are actually identical ⇒ the intervention did not take effect |
| 6 | the intervention group's `bias_mean` is not a tiny value (>1e-6) | non-zero but tiny ⇒ possibly zeroed out / not actually written |

### ⚠️ Run on historical data and it will report many FAILs — this is **expected**

```
checked 43 pairs, 25 failed.
```

**Most of them are not script bugs; those data simply never satisfied the pairing conditions.**
(An early version reported 32 pairs, of which 7 were **false positives by the tool itself** — e.g. it still checked L1=0 for groups that had no intervention to begin with.
Fixed; see `PITFALLS.md` item 11.)

Our summary table spans a week of exploration and mixes:
- cross-day A/B (different windows)
- stage comparisons from different starting points (their start_step differs to begin with)
- control groups where the injection failed (the bias was zeroed)

**This is exactly why we open-sourced this tool**:
without a check, you cannot tell "which data are comparable and which are not" —
and these FAILs are the list of "not comparable".

> **Documents are read by people; scripts are executed by machines.**
> Writing a pitfall down as a rule is far more useful than writing it as a paragraph.
