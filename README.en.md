# moe-load-metrics

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24404735.svg)](https://doi.org/10.5281/zenodo.24404735)

> Chinese version: [README.md](README.md)

> **Cite this version (v0.2.0)**: DOI `10.5281/zenodo.24404735`; **all versions**: `10.5281/zenodo.24404734`.
> Use the version DOI to cite this reproducibility snapshot; use the concept DOI to point at the project / latest version.

> **v0.2.0 (2026-11-21) — main changes:**
>
> 1. **Withdrawn**: the Baidu dose scan (2026-11-18) has **no archived logs**, so the injection vector and statistical
>    scope used at the time cannot be verified. **This document therefore does not cite any of its dose points**
>    (earlier claims derived from it — "Baidu minimum at 0.4", "20.8% overshoot at 0.8" — are withdrawn as well).
>    **The dose curve in §2.2 now comes from Ascend only; the Baidu part of §4 keeps only the baseline cv and the `ce_ema` of 6 runs.**
> 2. **Added**: the cross-seed difference (5 seeds / 10 pairs, median 0.3620);
>    the plateau (**after about 60~80 steps the load distribution no longer moves beyond the noise**);
>    a two-way check of the cumulative-scope formula.
> 3. `PITFALLS` now has **24 items**.
>
> **The item below was corrected on 2026-11-20 and is carried forward:**
> the `L1` intervention effect **0.0599 → 0.1077**, and the `L1` ckpt-switch difference **0.1025 → 0.2493**.
> Reason: the old values were computed **without undoing `reorder_perm`**; every `L1` in this document is now
> computed after restoring the original expert order. The conclusions point in the same direction and are now stronger
> (see notes ③ and ⑤, and `PITFALLS` item 17).

## MoE Expert-Load Metrics: A Precision Comparison of cv and L1

> A small experiment about **metric choice**.
> No methods on offer — it answers one question: **is the ruler you use to judge MoE routing any good?**

---

## Abstract

MoE training papers commonly argue that load balancing is working well by citing the
**coefficient of variation (cv) of expert load**.
We ran a set of controlled experiments on a 1.2B / 8-expert / top_k=2 model (paired / strength scan / long training / noise ruler / plateau) and observed:

1. **Under a single intervention, the change in cv is not proportional to the change in L1** — in the paired experiment cv differs by only 1.4%
   (~19× its own noise), while L1 differs by 0.1077 (~215× the difference between two repeats);
   in the long run, cv changes by 11% yet corresponds to L1 0.30, and a 1% change corresponds to L1 0.016 — **there is no stable ratio**
2. **cv is a scalar: it compresses away the shape of the distribution** — `abN3_r2_s1` has cv **0.3555** and a load range of **2.50×**;
   `ag_dose_g02` has cv **0.2833** and a range of **3.11×**.
   **The one with the lower cv has the larger range — the two scalars describe different aspects, and they do not rank the same way.**
3. **The cross-run difference is far larger than the intervention effect** — this is not "training is stochastic", it is "every run starts from a different initialisation":
   - **switching to another seed** (5 seeds, 10 pairwise): `L1` spans **0.2752 ~ 0.4736**, median **0.3620**
   - **switching to another ckpt**: 0.2493 (two different 130-step baselines, each continued) / 0.5638 (two independent trainings)
   - while the **intervention effect is 0.1077** (same-ckpt noise on the order of 1e-4)
   **=> The lower bound of the cross-run difference (0.2752) is already 2.6× the intervention effect; the median is 3.4×.**
4. **After about 60~80 steps the load distribution stops moving beyond the noise** — from step 81 on, `cv` stays at 0.2753 all the way to step 800;
   two window distributions 400 steps apart differ by `L1` of only **0.0005** (0.14% of the seed-switch median 0.3620).
   **=> "No measurable effect" is not because the window is too short; the distribution was stable long before.**

**Conclusion: comparing routing metrics across runs is unreliable, and mainly not because "training is stochastic",
but because "every training run starts from a different initialisation". And once the ckpt is fixed, cv still cannot reflect who carries the load — **it is completely invariant to reordering the experts**.**

---

## Repository Layout

```
README.md              this document
REPRODUCE.md           reproduction guide (three hard conditions + minimal steps)
PITFALLS.md            the pitfalls we ran into (24 items)
metrics.py             minimal implementation of cv / L1, runnable as is
make_figures.py        reproduces the three figures
check_pair.py          pre/post checks for the paired experiment (the package's main tool)
README.en.md           English version
data/                  experimental data (session cards + summary tables)
figures/               the three figures
LICENSE.md             CC BY 4.0 (data/text) + MIT (code)
AI-COLLABORATION.md    AI-collaboration statement (who did what, which mistakes the AI made)
PITFALLS.en.md         English version of PITFALLS.md
REPRODUCE.en.md        English version of REPRODUCE.md
REFERENCES.md          references and citation format
REFERENCES.en.md       references and citation format (English)
AI-COLLABORATION.en.md English version of the AI-collaboration statement
MANIFEST.sha256        content hashes of every file in the package (for verification)
METRICS-SCOPE.md       supplement: what cv / entropy / top1 / L1 each do and do not see (Chinese)
METRICS-SCOPE.en.md    the same supplement (English)
```

> The tree's annotations are translated here; file and directory names are verbatim.

---

## 1. The Two Metrics

```python
cv(x)    = std(x) / mean(x)                      # degree of dispersion, dimensionless
L1(a, b) = sum(|a[i] - b[i]| for i in range(E))  # distance between two share distributions
                                                 # precondition: a and b are share vectors summing to 1, range 0~2
```

> The actual `metrics.py` in this repository has Chinese comments; the block above is translated for readability.

| Metric | Under permutation | What it reflects | Units |
|---|---|---|---|
| **cv** | **permutation-invariant (insensitive)** | the **degree** of imbalance | dimensionless |
| **L1** | **sensitive** | **who** carries the load | share (0~2) |

**"Expert 0 takes 40% / expert 7 takes 5%" and "expert 0 takes 5% / expert 7 takes 40%" have exactly the same cv.**

---

## 2. Three Pieces of Evidence

### Platforms

| Platform | Accelerator | Software stack | Platform characteristics relevant here |
|---|---|---|---|
| AtomGit | Ascend 910B4 (32 GB HBM) | PyTorch 2.9.0 + torch_npu 2.9.0, CANN 8.5.0 | 1e-4~6e-4 run-to-run non-determinism |
| Baidu AI Studio | Iluvatar BI-V150S (32 GB) | PyTorch 2.4.1 + corex, IX-ML 4.3.8 | **bit-deterministic**; fp64 statistics are 0 (see note ①) |

> These two platform properties directly support two claims in this document:
> ① **the noise floor is a property of the platform** → signal-to-noise ratios are not comparable across platforms (see §3);
> ② fp64 statistics on Baidu are 0 → `RouteStats` now uses int64/fp32 counters (see note ① and PITFALLS item 7).

> **Note ① (a record of what we observed, not a claim about the platform)**
> Under the platform configuration given in this repository, when we computed load statistics with `torch.float64`
> the results were always 0 and no exception was raised. **We have not located the root cause** —
> it may come from the environment, the driver, a library version, or from how this repository uses them.
> **We make no assertion about the platform on this basis; please open an issue if you need reproduction details or a minimal case.**
>
> **This does not affect any conclusion in this document**: `RouteStats` has been switched to `int64`/`float32` counters,
> so all metrics **bypass the fp64 path** (see PITFALLS item 7).

Measured throughput (2048×2048 matrix multiply, not nominal values): Ascend bf16 207.9 TFLOPS; BI-V150S bf16 92.6 TFLOPS.

### 2.1 Paired: one intervention moves cv by 1.4% but moves L1 by 215× the noise

> **Data date**: 2026-11-17 (Ascend) | **Conditions**: fixed ckpt (one file, copied twice) | single platform.
> Raw cards in `data/session_cards/card_pair2*`.

We copied the same ckpt twice, added 0.2 to the router bias in one copy and left the other untouched, and trained each for 129 steps (`pair2A/2B` both go 127→256).

| | L1 | cv |
|---|---|---|
| intervention vs no intervention | **0.1077** | 0.3899 vs 0.3955 (difference 0.0056, i.e. 1.4% relative) |
| **this group's own noise** (A1 vs A2, two independent runs) | **0.0005** | **0.0003** |
| reference: another group, 5 repeats | 0.0001 ~ 0.0006 | range 0.0004 (that group's cv≈0.205) |
| **signal-to-noise ratio** (effect ÷ this group's own noise) | **215×** (0.1077 ÷ 0.0005) | **~19×** (0.0056 ÷ 0.0003) |

> **Note the denominator**: the "within-group noise" of both L1 and cv is taken as **this group's own difference between two repeats**
> (A1 vs A2); we do not mix in the 0.0004 of the other group (the one with cv≈0.205) — the two differ in cv magnitude, and mixing them would give the wrong ratio.


![Paired experiment](figures/fig2_pair.png)

**This is the hardest piece of evidence in this document**: the two groups start from an identical point, the only difference is those 8 bias numbers,
and yet L1 amplifies the difference 215×, while cv amplifies it only 19× (a change of 0.0056, i.e. 1.4% relative).

### 2.2 Intervention-strength scan: how cv and L1 respond

> **Data date**: Ascend 2026-11-17; Baidu 2026-11-18 | **Conditions**: fixed ckpt |
> **the dose curve is Ascend only; Baidu contributes the baseline value only** (its dose points have unverified provenance, see below).
> Raw cards in `data/session_cards/card_dose_*` and `card_baidu_d*`.

We scanned gamma on the same ckpt (96 steps each):

| Platform | Control cv (same window) | γ=0.1 | γ=0.2 | γ=0.4 | γ=0.8 | Shape |
|---|---|---|---|---|---|---|
| Ascend | 0.1978 | **0.1125** | 0.1287 | 0.1639 | — | non-monotonic (U-shaped, minimum at 0.1) |
| Baidu | **0.2674** (baseline) | — | — | — | — | **dose points unverified, no conclusion drawn** (see below) |

> ⚠️ **The Baidu dose scan (2026-11-18) has no archived logs, so the injection vector and
> statistical scope used at that time cannot be verified. This document therefore does not cite its dose points.**
> A controlled comparison on 2026-11-20 (**source cards `card_bd_M0` / `MC` / `M2` / `M4`**, same base, same 96-step window)
> showed that on a different base the same γ can move cv in the opposite direction (`cv` rises from 0.2753 to 0.3403 at γ=0.2). **This further indicates that "optimal dose"
> cannot be extrapolated across bases.**

⇒ **On Ascend, overshoot shows up as "a smaller improvement"**: `cv` climbs back from `0.1125` (γ=0.1, the minimum) to `0.1287` (0.2) and `0.1639` (0.4), **but stays below the baseline `0.1978`**.
**This document has no verified data point where the intervention pushed `cv` above the baseline** — the Baidu γ=0.8 point once observed has unverified provenance (see above) and is not used.

> ⚠️ **On Ascend this response is non-monotonic (0.1978 -> 0.1125 -> 0.1287 -> 0.1639). The Baidu dose points are unverified and no conclusion is drawn from them.**
> **Nothing about any "platform regularity" can be concluded** — the baselines behind the two controls are in different states, see Section 5.
> The two sides also differ in **baseline imbalance** (cv 0.19 vs 0.27), and **this experiment does not disentangle platform from control**.

### 2.3 Long training: cv and L1 move in the same direction, but not proportionally

> **Data date**: 2026-11-16 (Ascend, 800 steps) | **Conditions**: two sessions of a single run (same ckpt chain) | single platform.
> Raw cards in `card_ag_long_s1/s2`.

> **⚠️ Statistical scope**: `s1` in this section is `0→408` (trained from scratch, no history); `s2` is a continuation.
> **No cross-batch comparison in this repository mixes the two scopes**: continuation cards before 2026-11-19 are "cumulative",
> those after are "this session only", and the two are not directly comparable.

> **Note ② (data provenance)**: the 8-dimensional expert-load array for `step0` in this section (used to compute L1 = 0.3009)
> survives only in `data_for_figures.json`; **there is no corresponding raw session card**, so it is summary-layer data.
> Please check against that file; every other datum has a card. **This is the only place in this document where the first-level provenance is missing.**

**Experimental conditions (required for reproduction)**: `router_mode=none`.
The `none` mode does not call `reorder_experts`, so `expert_load` keeps the **original index order**.
Under the `sber` mode, the reorder at the end of the session permutes the array in descending load order — **the phenomenon then cannot be observed directly**.

800 training steps (this run consumed 53.54 M tokens):

| Interval | cv change | L1 |
|---|---|---|
| first 400 steps (step0 → s1) | 0.3093 → 0.3432 (**+11%**) | **0.3009** |
| last 400 steps (s1 → s2) | 0.3432 → 0.3465 (**+1%**) | **0.0164**<br>(card value 0.0082, see note ③) |

> **Note ③ (the statistical scope of `s2`)**: `s2` was continued from `s1`, so the `expert_load` in its card is
> **cumulative over (history + this session)** (408 steps of `s1` + 409 steps of this session), not this session alone.
> Back-solving with `sess/(base+sess) = 409/817 = 0.501` gives a **true L1 for this session of about 0.0164** (the card value 0.0082 × 2).
> `s1` was trained from scratch and **has no history**, so its `cv` and `L1` are unaffected.
>
> **⇒ The table in this section uses 0.0164 and notes the card's original value. This is the only number in this document that needs a scope correction.**

![Rotation](figures/fig3_rotation.png)

**An 11% change in cv corresponds to L1 0.30, and a 1% change to L1 0.016 — there is no stable proportional relationship between them.**
(Dividing the **absolute change in cv** by the repeat range of cv, `4e-4`: the 11% stretch is `0.0339/4e-4` = **85×**, the 1% stretch is `0.0033/4e-4` = **8×**; both stretches exceed the noise.
**These are multiples in cv; the L1 values for the same two stretches are 0.3009 and 0.0164, a different quantity.**)

## 3. Precision: The Noise Ruler

> **Data date**: Ascend 2026-11-17 (5 repeats); Baidu 2026-11-18 (2 repeats).
> **Note**: an earlier measurement (2026-11-15) on a different configuration gave a cv fluctuation of ±20%;
> that batch **did not fix the ckpt**, so it should not be placed alongside this table — it shows that
> "runs are not comparable across runs", not "noise within the same ckpt".

| Setting | Conditions | L1 | cv |
|---|---|---|---|
| within one ckpt, Ascend 5 repeats | **fixed ckpt** / single platform | 0.0001 ~ 0.0006 | range 0.0004 |
| within one ckpt, Baidu 2 repeats | **fixed ckpt** / single platform | **0.0000 (bit-identical)** | identical |
| **switch to another seed** (different init) | across runs / **Baidu only** | **0.2752 ~ 0.4736**<br>(5 seeds, 10 pairwise, median 0.3620) | cv spread 0.12 |
| **switch to another ckpt** (two different 130-step baselines, each continued) | across runs / single platform | **0.2493 (N=1)** | — |
| **switch to another ckpt** (two independent trainings) | across runs / single platform | **0.5638 (N=1)** | — |
| ~~cv ±20% (2026-11-15)~~ | **across runs** / ckpt not fixed | — | **±20%** (not to be placed alongside this table) |

![Noise ruler](figures/fig1_noise_scale.png)

> **Note ④ (the provenance of `0.2493`, which must be stated)**:
it comes from `L1(rep5_rep1[130 steps], phase_pt130_a[128 steps])` — **the two windows differ by 2 steps**,
so strictly speaking it does not satisfy hard condition 2 of REPRODUCE in this package. **The contribution of the window mismatch is estimated from measurements inside the package**: `card_ceTime` (84 steps) / `ce85` (85 steps) / `ce96` (95 steps)
are three runs on the same baseline with the same seed (all status ok); measured, **a 1-step difference gives L1 ≈ 0.0005 and 10 steps ≈ 0.0024**
(~2.5e-4/step) ⇒ **2 steps is about 1e-3, i.e. about 1% of 0.2493**.
Estimating more conservatively from the slope over the first 408 steps (0.3009/408 ≈ 7.4e-4/step): 2 steps ≈ **1.5e-3 (1.5%)**.

**⇒ The conclusion is unaffected, but this number should be read as "an estimate containing at most ~1.5% window contamination", and we could not find a same-window pair.**

> **Note ⑤ (where the second number, 0.5638, comes from)**:
> `L1(card_1118_base_a, card_1118_base_b) = 0.5638`.
> Both runs are `router_mode=none`, `0→130` steps, same seed and configuration,
> with `reorder_perm=null` (`none` mode does not reorder, so **no permutation undo is needed**).
> Both cards are in `data/session_cards/` and can be checked directly.
>
> **⚠️ This number says one thing only: two identically configured runs trained from scratch for 130 steps
> can differ by 0.5638 in load distribution.**
> **It is an observation about one pair of runs, not a "typical cross-run value".**
> (Their `cv` values are 0.4671 / 0.3516; 0.4671 is higher than every later base's `cv`,
> so it should be read as one sample from the tail.)
>
> **Note ⑥ (the 10 seed pairs, added 2026-11-20)**:
> On Baidu, 5 seeds (101~105) were each trained from scratch for 130 steps; comparing all pairs gives 10 `L1` values:
> `0.2752 / 0.2855 / 0.3266 / 0.3451 / 0.3607 / 0.3634 / 0.3851 / 0.4303 / 0.4705 / 0.4736`,
> **median 0.3620, mean 0.3716**. All five cards are in `data/session_cards/` and all have `perm = null`.
>
> **⇒ The load movement caused by "switching to another seed" has a median `3.4×` the intervention effect (`0.1077`);**
> **even the smallest pair (`0.2752`) is `2.6×` it.**
>
> ⚠️ **Caveat 1**: 4 of the 10 pairs use the same `seed101` as reference and are therefore **not independent**;
> the independent sample is **5 seeds**, not 10 distances. Reporting "5 seeds, 10 pairwise distances" is more honest than "`N=10`".
>
> ⚠️ **Caveat 2**: the `seed101` card came from the **old statistics scope** (before 2026-11-19, where resuming added
> historical counts); the other four seeds came from the **new scope** (with `FIX 2026-11-19`, zeroing on resume).
> **These runs are from scratch, and the two scopes should not differ there** — and this was verified: the old `seed101`
> card and a new same-seed card are **bit-identical** (`L1 = 0.0000`, `cv 0.2626`). So the 10 pairs are comparable,
> but a reader checking this themselves should know that layer exists.


## 3a. Plateau: after about 60~80 steps the load distribution no longer moves beyond the noise

> **This section addresses an obvious objection: are your negative conclusions only because training was too short?**

**The same base (`seed=101`, same configuration), trained from scratch for three lengths, looking at the per-segment window `cv`:**

| Length | `cv_every` | per-segment window `cv` |
|---|---|---|
| 130 steps | **20** | `1~20`: 0.2427 -> `21~40`: **0.3286 (+35.4%)** -> `41~60`: 0.2800 -> `61~80`: **0.2754** -> `81~100`: 0.2753 -> `101~120`: 0.2753 -> `111~130`: 0.2753 ⚠️ |
| 400 steps | 50 | `1~50`: 0.2542 -> `51~100`: 0.2754 -> from `101~150` on: all 0.2753 |
| 800 steps | 100 | `1~100`: 0.2599 -> from `101~200` on: all 0.2753 |

**⚠️ With the three side by side, the first has the finest resolution (20 steps), and it is the one that shows what actually happened early on:**

> **The real shape is "rise -> overshoot -> fall back -> stable":**
> `cv` reaches **`0.3286` in the `21~40` segment (19% above the final value)**, falls back to `0.2800`, is `0.2754` at `61~80`, and is constant at `0.2753` from `81~100` on (**the `0.2754` vs `0.2753` difference is within the noise**).
>
> **The first 60~80 steps are not a "slow convergence" — they are an overshoot followed by a fall-back. A 50-step window averages that overshoot away —**
> **looking only at 50/100-step windows suggests the first 50 steps sat smoothly at `0.2542`, which is not true.**

**⚠️ About the last two points**: the final `cv_trace` entry is aligned to `end_step` (130), and 130 is not a multiple
of `cv_every=20`, so `111~130` **overlaps `101~120` by 10 steps** — both read 0.2753, so the conclusion is unaffected,
but a reader reconstructing the windows will see the overlap.

**（This is exactly what `PITFALLS` item 24 warns about: the same `step` index with a different `cv_every` is a different window.**
**This section originally described, with 50/100-step windows, a process that only a 20-step resolution reveals; it was corrected after review.)**

**Stronger evidence: back-solve two window distributions 400 steps apart and compare them directly.**

```
window 131~400 (back-solved): 0.1428 0.1071 0.0892 0.0714 0.1250 0.1429 0.1428 0.1786
window 400~800 (back-solved): 0.1428 0.1071 0.0893 0.0714 0.1249 0.1428 0.1429 0.1785
L1 = 0.0005
```

> **400 steps apart, and the shares of all 8 experts are essentially unchanged one by one (`L1 = 0.0005`).**
> **The median for "switching to another seed" is `0.3620` — the movement caused by 400 steps of training
> is `0.14%` of that (about `1/700`).**

**Over the same interval, task quality improves:**

```
ce_ema:  step 130: 9.0245  ->  step 400: 8.9512  ->  step 800: 8.9088   (down 0.116)
```

> **=> The model is learning; the load distribution does not follow.**
> **In these runs, what determines the load distribution is the initial condition and the first few tens of steps, not the training length.**

**Source cards**: `card_bd_M0` (`cv_every=20`), `card_bd_long400` (`cv_every=50`), `card_bd_long800` (`cv_every=100`),
all under `data/session_cards/`. **Back-solve method**: invert `(n1·L1 + n2·L2)/(n1+n2)` on the two cumulative distributions
to recover the later window, then cross-check against that card's `cv_trace` (the recovered window's `cv` agrees with
the `cv_trace` reading to within 2.5e-4).

**(`cv_trace` and `expert_load` have different scopes: `cv_trace` is per-segment windows, `expert_load` is the
cumulative average over the whole from-scratch run. Every window `cv` in this section comes from `cv_trace`;
we do not mix the two scopes here.)**

---

**⚠️ Two further points must be stated:**
1. **The level of reproducibility is a property of the platform**: Ascend has 1e-4~6e-4 of run-to-run noise,
   whereas the same code on Baidu is **bit-deterministic**. The "signal-to-noise ratios" in this document therefore hold only within their own platform and are **not comparable across platforms**.
2. **"A ckpt switch costs 0.25" was measured on only one pair** (same day, two different 130-step baselines, each continued); the direction is consistent but **N=1**.

---

## 4. A Side Result: the Bias Changes the Load Without Changing Task Quality

> **Data date**: Ascend 2026-11-17; Baidu 2026-11-18 | **Conditions**: fixed ckpt |
> **the dose curve is Ascend only; Baidu contributes the baseline value only** (its dose points have unverified provenance, see below).

| Platform | Intervention | cv | ce_ema |
|---|---|---|---|
| Ascend | 5 dose levels (0~0.4) | 0.1978 → 0.1125 (**-43%**) | spread **0.0002** (range across the five groups) |
| Baidu | (dose points unverified, see §2.2) | — | **`ce_ema` spread across 6 groups: `9e-4`**<br>(`c1/c2/d01/t1/d04/d08`: 8.9491 / 8.9491 / 8.9487 / 8.9491 / 8.9486 / 8.9482) |

**Across both platforms' records there is no measurable association between "load balance" and "language-modelling quality".**

> **Why this does not depend on the Baidu dose points**: it uses the `cv` and `ce_ema` **recorded** for each run,
> not the "intervention strength". Even if the injection vector of some batch has unverified provenance,
> **its `cv` and `ce` are still real recorded values** — across the 6 runs `cv` ranges from 0.1456 to 0.3229 (a spread of 0.177),
> while `ce_ema` moves only within 9e-4.

---

## 5. Scope: this document does not claim that "a universal intervention strength exists"

**This document answers one question only: which of `cv` and `L1` sees the allocation. It does not answer "what intervention strength should be used" - that is separate work, outside this package scope.**

So this document does **not** claim that:

- some optimal intervention strength exists;
- that strength can be extrapolated across bases or platforms.

**The one strength-related fact**: on this one Ascend base, `γ` in the range `0.1~0.4` all produced a **measurable** load change
(`cv` 0.1125 / 0.1287 / 0.1639, baseline 0.1978).
**That is enough to support the experimental design - "we can produce a measurable change in allocation" - but not enough to support any conclusion about an "optimal value".**

> ⚠️ **One observation that must be stated**: on another `base`, the same `γ` can move things in the opposite direction
> (source cards `card_bd_M0` / `MC` / `M2` / `M4`, same base, same 96 steps; at `γ=0.2`, `cv` rises from 0.2753 to 0.3403).
> **This is the empirical basis for not claiming a universal strength**, but it belongs to a separate line of work
> ("how to change the load") and this document does not pursue it — **it is noted here only as a pointer for later versions.**

---

## 6. Related Work

### 6.0 Source of the Method (Required Reading)

The practice used here — "add a load-dependent bias to the router and push the load flat during training" —
is **aux-loss-free load balancing**, from the **DeepSeek-V3 technical report [3]** (arXiv:2412.19437)
https://arxiv.org/abs/2412.19437

> See step 3 of REPRODUCE.md. **This document does not propose the method; it measures "which metric should be used to evaluate it".**


### 6.1 Independent evidence that routing is sensitive to training conditions

**[1] MoE Pretraining: Parallelism Layout Changes the Model, Not Quality** (arXiv:2611.15271, 2026-11-11)
https://arxiv.org/abs/2611.15271

That paper runs a set of experiments complementary to ours:
- **changing the parallelism layout** mid-training leaves **model quality unchanged** but **changes the model itself**
  (the final prediction difference is about **80%** of that from changing the seed, with **>90% of routing decisions flipped**);
- the flips concentrate in **near-tie tokens** (boundary samples);
- the **timing of the switch** matters: switching at 84% of pretraining has almost no effect.

**Relation to this document**: they show "change the parallelism layout → routing changes a lot, quality does not";
we show "**change the ckpt (same seed / same configuration / same data) → the load distribution differs by **0.25 (two continuations on the same day) to 0.56 (two independent trainings)** — larger than the intervention effect of 0.11 we want to measure**".
Both point to the same thing: **routing state is extremely sensitive to training conditions, while task quality is not.**

> This also explains why we insist that "you must lock onto that one specific ckpt file" —
> changing the parallelism layout and running training once more are, in effect, disturbances of the same kind.

### 6.2 Irreproducibility of Routing Itself

**[2] Even under identical conditions, repeated forward passes can choose different experts** (arXiv:2510.11370)
https://arxiv.org/abs/2510.11370

That paper suggests recording the routing distribution and replaying it, rather than assuming the forward pass is deterministic.

**Relation to this document**: what we measured are **within one ckpt** differences that are extremely small
(Ascend 1e-4~6e-4; Baidu 0, bit-identical),
consistent with "even identical conditions can diverge" — **and we provide the magnitude**.

### 6.3 Claims We Did **Not** Adopt

This document does **not** claim that "some universal intervention strength exists". See the caveats in Section 5.

---

## 7. Reproduction Conditions (Three, All Mandatory)

1. **The same ckpt file** (not "the same seed")
2. **The same window length** (not "roughly the same")
3. **The intervention magnitude is computed on the fly from the current baseline's own load** (not reused from another baseline's value)

> We distilled these three after stepping in the pits ourselves. Omit them and reproduction will fail.

---

## 8. Limitations

| Item | Note |
|---|---|
| Scale | 1.2B / 8 experts / top_k=2 |
| **Method-version coverage** | We measure the **pure** aux-loss-free of the V3 era. From V4 on it became "aux-loss-free + a small amount of sequence-wise balance loss" (arXiv:2606.19348), and V4.1 further splits the bias per modality (arXiv:2609.19969) — **neither of these two variants was measured** |
| Data | A single corpus (56.35 M tokens; the original record reads 5635 in units of 10⁴); the long-training run actually consumed **53.54 M tokens** (original record 5354 in units of 10⁴, the two sessions added together) |
| Model extrapolation | **larger models / more experts not verified** |
| L1 of a ckpt switch | two observations, each **N=1**: 0.2493 (same day, **two different 130-step baselines**, each continued, windows 130 vs 128 steps) and 0.5638 (two independent trainings)<br>**Two observations are not enough for a "typical cross-run value"** |
| L1 of a seed switch | **5 seeds, 10 pairs (0.2752~0.4736)** — the largest sample in this document on "cross-run differences" (see note ⑥) |
| **Repeats on the same base** | **Done once, within a single instance**: group A of §2.1 is one ckpt copied into two arms (`A1` / `A2`, `L1 = 0.0005`).<br>**What was not done is a cross-instance repeat of the same base** — rerunning the same configuration on another machine or session.<br>Reason: the ckpt that produced `0.1077` was destroyed with the instance. **This is an inherent cost of the "ckpts are not persisted" experimental design**, not an oversight |
| Mechanism of the optimum | **undetermined** (**partly ruled out**: 0.79 vs 0.71, not fully attributed) |
| Comparability of SNR | Baidu noise 0 / Ascend 1e-4~6e-4, **not comparable across platforms** |

---

## 9. About This Material Itself

### 9.1 AI Collaboration

The code, experiment execution, analysis and writing in this repository were produced with the assistance of AI systems; humans were responsible for direction-setting and final review.
**The full account (including a record of the AI being proven wrong more than 10 times during this work) is in [AI-COLLABORATION.md](AI-COLLABORATION.md).**

> We chose to publish the AI's mistakes because **a narrative that never errs is more suspect than a process that errs and then checks itself.**

### 9.2 Reference Verification

The top of `REFERENCES.md` states the **scope of verification**: the titles and identifiers of all eight references were checked;
for the five cited in the body (**[1]–[5]**, with [3] cited in §6.0 as the source of the method), the wording of their conclusions
was compared against their abstracts;
the three background references had only their identifiers and titles checked, and their conclusions are not used in the body.
**Note**: the technical details of [4] V4 and [5] V4.1-Flash (sequence-wise balance loss, Sigmoid→Sqrt(Softplus)) are cited from their abstracts and have not been independently verified.

### 9.3 Corrections Welcome

This repository is **a work in progress, and we do not rule out that undiscovered errors remain**.
If you find a problem, please open an issue — that is better than letting the error spread.

## 10. If You Remember Only One Sentence

> **cv measures "how concentrated", L1 measures "who gets it".
> And every training run starts from a different initialisation — switch the seed and the median load difference is 0.3620
> (smallest pair 0.2752), which is **2.6~3.4×** the intervention effect (0.1077) you want to measure.
> Worse: after about 60~80 steps this distribution is already stable, and training 10× longer barely moves it.
> (The one exception is "the very same ckpt file": there the noise is only on the order of 1e-4.)**
