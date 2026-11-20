# cv, entropy, top1, L1: what each of them sees

> Supplementary material: it explains what `cv`, entropy, `top1` and `L1` each measure, and what they are blind to.
> **This document does not change the main conclusions of the README; it only adds "which metric to use, and when".**

---

## 1. One Set of Changes, Four Metrics, Four Different Stories

| Metric | base A | control run | change | base B | control run | change |
|---|---|---|---|---|---|---|
| `cv` | 0.337 | 0.204 | **−40%** | 0.356 | 0.367 | +3% |
| entropy (ln) | 2.028 | 2.062 | +1.7% | 2.028 | 2.024 | −0.2% |
| `top1` share | 0.187 | 0.179 | −4% | 0.214 | 0.214 | 0% |
| **minimum share `min`** | **0.058** | **0.107** | **+85%** | **0.088** | **0.089** | **+1%** |

---

## 2. Three Sentences

### 2.1 `cv` is almost entirely determined by the tail

```
A: cv −40%  ←→  min +85%
B: cv  +3%  ←→  min  +1%
```

**It is not that `cv` is oversensitive; it is that the tail is exactly what it measures.**

### 2.2 Entropy is already close to its upper bound in this range

```
perfectly balanced  entropy = ln(8) = 2.079
measured            2.024 ~ 2.062
normalised          97.4% ~ 99.2%
```

**It is not that entropy is inaccurate; it is that at 8 experts and close to balance it has nearly hit the ceiling, so its resolution is low.**

### 2.3 `top1` sees only the head, not the tail

A's `top1` drops by only 4%, yet the tail rises by 85% — `top1` does not see this change at all.

---

## 3. Two Independent Corroborations (**independent of any intervention — pure statistical relations**)

### Corroboration A: `cv` almost identical, yet the allocation differs by 81% of a dose effect

| Arm | γ | `cv` | entropy | `min` | `max` |
|---|---|---|---|---|---|
| E1 | 0.1 | **0.2645** | 2.047 | 0.071 | 0.161 |
| E2 | 0.2 | **0.2621** | 2.048 | 0.071 | 0.179 |

```
cv differs by only 0.9%
L1(E1, E2) = 0.1155
L1(E0, E1) = 0.1426 (the dose effect itself)
=> L1(E1,E2) is 81% of the dose effect
```

> **The `cv` of the two configurations is almost identical, yet their allocations differ by 81% of the dose-effect magnitude.**
> **This needs no intervention to explain it; they are simply two freely chosen configurations: `cv` says "the same", `L1` says "very different".**

### Corroboration B: three bases with similar `cv` but completely different dose-response shapes

| base | `cv` | `min` | shape of the dose response |
|---|---|---|---|
| A | 0.3383 | 0.058 | single peak at γ=0.1 |
| B | 0.3557 | 0.088 | single peak at γ=0.2 |
| C | 0.3238 | 0.054 | plateau over 0.05~0.2, no peak |

**The same `cv` magnitude, completely different shapes ⇒ `cv` is not enough to describe a baseline state.**

---

## 4. The Purest Demonstration: Without Restoring, L1 Is Exactly 0

**`E05` (γ=0.05) and `E1` (γ=0.1):**

```
perm E05 [5,7,0,1,3,2,6,4]
perm E1  [1,5,0,7,3,2,6,4]

L1 not restored = 0.0000   <- the post-sort arrays are identical position by position
L1 restored     = 0.0356
```

**And `cv` / entropy / `min` / `max` are all identical.**

> **The only difference is "which experts received these shares".**

This is the purest demonstration of `L1` as an "allocation metric":

```
the same load distribution, carried by a different set of experts
every scalar metric (cv / entropy / top1 / min / max) is blind to it
the information "which expert" is completely lost
only the restored L1 can see it
```

**⇒ And this was not constructed on purpose — it was run into naturally while scanning the dose; the two γ differ by only 0.05.**

---

## 5. Conclusions

| Metric | Sees | Does not see |
|---|---|---|
| `cv` | **the tail** (the fluctuation of the minimum share) | the spread of the head; who received what |
| entropy | the overall shape (low resolution near the ceiling) | tail detail; who received what |
| `top1` | head concentration | **the tail is entirely invisible** |
| **`L1`** | **who received what, and how much it changed** | **it does not tell you whether the load is balanced** |

> **No single scalar can describe "whether the load allocation is reasonable". This is not to say the scalars are bad; it is to say they answer different questions.**

---

## 6. Scope Restriction (Important — Otherwise This Will Be Taken as a General Law)

**"A fall in `cv` does not mean the balance improved" holds only when entropy is close to its upper bound.**

| Setting | Conclusion |
|---|---|
| **8 experts, close to balance (this document)** | a fall in `cv` may only mean the tail was lifted; entropy barely moves |
| **64 experts, `cv` = 0.5 (far from balance)** | a fall in `cv` **most likely does correspond to a rise in entropy** |

> **Every conclusion in this document is restricted to "8 experts, `cv` in the 0.2~0.4 range".**
> **It must not be generalised to "`cv` is generally unreliable".**

---

## 7. Tool Selection Guide

| What you want to know | What to use |
|---|---|
| whether any expert is starving | **`cv`** (or look at `min` directly) |
| how spread out the load is overall | entropy (resolves only away from the ceiling) |
| whether one expert is eating everything | `top1` / `max` |
| **who the load went to, and how much it changed since last time** | **`L1`** |
