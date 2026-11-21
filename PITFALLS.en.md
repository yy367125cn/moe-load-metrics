# The Pitfalls We Ran Into

> **Added 2026-11-21: these leaks were only cleared in v0.2.1** — `0.10` in the body of PITFALLS entry 3 and
> `0.10 / 0.06` in the first hard condition of `REPRODUCE` were still present in the v0.2.0 release.
> **This is the third recurrence of the same trap in this repository.** It shows that "grep once afterwards" is not a
> reliable habit; it has to be a script that scans every number before a release.

> Why this section exists: **every item below once made one of our conclusions wrong**,
> and each has the same signature — **every number is individually correct, but together they are false.**

---

## Analysis Layer

### 1. Splicing numbers from two different intervals into one sentence

**The wrong way to write it**: "cv is constant (0.3432→0.3465) but the ordering is changing (L1 0.30)"

**What is actually the case**:
- `cv 0.3432 → 0.3465` is **s1→s2** (that stretch has an L1 of only **0.0082**)
- `L1 0.30` is **step0→s1** (in that stretch cv climbs from 0.3093 to 0.3432, **+11%**)

**Each number is correct on its own; put together they are false.**

> Self-check: **any sentence of the form "X is unchanged but Y changed a lot" requires checking that X and Y come from the same interval.**

### 2. Comparing across runs with ce_loss_end

`ce_loss_end` is the loss of the **last batch**, with no smoothing whatsoever.

Measured: with the same configuration, running 214 steps vs 215 steps gives a `ce_end` difference of **3.1%**, while `ce_ema` differs by only 0.17%.

**⇒ Comparing CE across different step counts requires `ce_ema`, or an identical window length.**

### 3. Treating a single result as a distribution

We once saw cv differ by 18% under the same configuration and briefly concluded "this metric is unusable";
6 measurements later showed that the fluctuation is indeed large, but that **switching to another ckpt costs 0.25 (L1)**, which is the larger noise source.

**⇒ Measure the noise ruler before fixing a conclusion.**

---

## Implementation Layer

### 4. Set ≠ in effect

We hit this twice:
- the forward pass with `router_mode=none` **does not read the bias at all**, and `set_router_mode` resets it
- when restoring from a ckpt, `load_state_dict` **overwrites the bias back to 0**

**In both cases "the log showed the setting succeeded, but in the actual model it was 0".**

**Fix: read the value back and print the measured value immediately after setting it; do not print only the set value.**

> **And "I checked it" must state which file / which run you checked.**
> We have been bitten: we checked the **previous version of the package**, but believed we had checked **the version we had just rebuilt** —
> so we announced "there are no junk files in the package", when the package we had just generated contained a `.pyc`.
> **We verified A and believed we had verified B.**

### 5. Arguments starting with a minus sign are parsed as new options by argparse

```bash
--bias-init -0.1,-0.1,...     # ✗ raises "expected one argument"
--bias-init=-0.1,-0.1,...     # ✓
```

> Code blocks below are translated for readability; the actual scripts in this
> repository have Chinese comments.

**The error message points at "missing value", not at "the minus sign was taken as an option"**, so a quick look suggests you forgot to write an argument.
The consequence is that **the entire batch of experiments fails silently**.

### 6. The failure path of a self-check was never executed

We wrote a "startup self-check", but `sys` was not imported —
so as soon as any item failed it raised `NameError` instead of exiting cleanly.

**We wrote something to "prevent silent failure", and it silently failed.**

**Fix: any code that is supposed to exist/alert on failure must be triggered once and have its behaviour verified.**

### 7. Silent downgrade on the platform

On the Baidu Iluvatar platform, `torch.float64` **silently returns 0** (no error).
Our metrics code accumulated in fp64 at the time ⇒ every metric was identically 0, while training looked perfectly normal.

**Fix: self-check dtype capability at startup; use int64 for integer counts rather than float.**

### 8. One column name carrying two semantics

The summary table used to have a column `bias_dose` (intervention dose). It was filled with `bias_max` — regardless of where that value came from.

Result: **runs carried out with session-boundary SBER (`--gamma-sber 0.2`) were also filled with 0.200195**,
but that is not the "injected dose"; it is "the statistic of the bias after the run".

**Readers will assume those runs also had a bias injection.**

**Correction**: only runs with `gamma_sber == 0 and bias_max > 0` (the ones that actually used `--bias-init`) get this value; the rest are left blank.

> **Self-check: any column whose value can arise from two different paths must be split into two columns.**
> Or at least add a `source` field saying where it came from.

### 9. Not rescanning the whole document after a follow-up measurement — and the follow-up conclusion itself may have to be retracted

On 2026-11-18 we added a follow-up measurement on the Baidu side at γ=0.8 and concluded that
"the Baidu side is U-shaped (cv 0.3229 at 0.8, severe overshoot)", and we changed only the
"most relevant" section — another section of the README still said "monotonically decreasing,
0.4 is still optimal". **Two sections of the same material contradicted each other.**

Cause: after the follow-up measurement we changed only the "most relevant" section,
**without rescanning the other places that mention the same thing**.

**Worse still:** a controlled comparison on 2026-11-20 found that the logs of that batch were
**never archived**, so the injection vector and statistical scope used at the time cannot be
verified — **even the conclusion "U-shaped" itself has to be retracted.**
Section 2.2 is now annotated "dose points unverified, no conclusion drawn", and every derived
number that rested on it has been removed (including "minimum at 0.4", "20.8% overshoot at 0.8"
and "bias 0→0.2 improves by 19%").

> **Two self-checks:**
> **1) After any follow-up measurement, search the whole document for the relevant numbers and the new conclusion, rather than changing only the place you remember.**
> **2) A conclusion obtained from a follow-up measurement must itself be independently re-checked in the next round of experiments —**
> **otherwise you will grow a whole tree of derived conclusions on an unverified premise.**

> A sturdier approach: gather all numbers into a single `numbers.md`, and have the body reference
> them instead of hard-coding them — then changing one place suffices.

**Why this item is kept: it is the only entry in this repository that records us retracting a
conclusion we had already published.** Deleting it would hide the evidence that this project
corrects itself.

---

## The Value of Tools: Expose Problems, Not Make Numbers Look Good

On our 65 lines of historical data, `check_pair.py` reported **25/43 pairs failing** (an early version reported 32 pairs, of which 7 were false positives by the tool itself, see item 11).

We did not make those failures go away; we wrote them into REPRODUCE.md:

> "It is not that the script has a bug; it is that those data never satisfied the pairing conditions.
> This is exactly why we open-sourced this tool: without a check, you cannot tell which data are comparable and which are not —
> and these FAILs are the list of 'not comparable'."

**A tool that reports 0 failures is more suspect than one that reports 25.**

### 10. Batch text replacement fails silently

Right after fixing the pitfalls above, while editing documents we **stepped in it once more ourselves**:

we used `text.replace(old, new)` to change numbers in bulk, and **two of the `old` strings were not in the file at all**
(a punctuation difference), so `replace` returned the text unchanged — **no error, no warning, and we believed we had changed it**.
Result: during re-review the reviewer found "claims a change that was not actually made".

**This is the same error as item 4 (set but not in effect), only at the file level.**

**Fix**: every batch replacement must carry assertions —

```python
def rep(txt, old, new, label):
    assert old in txt, "[" + label + "] old not found"
    out = txt.replace(old, new)
    assert new in out, "[" + label + "] new not present after the replacement"
    return out
```

**"I finished the edit" is not evidence; "what is in the file now" is.**

### 11. Checking tools have false positives, and false positives hurt more than misses

Before release someone took our own `check_pair.py` and ran it, and found that it reported a **false error** on one set of data:

```
card_phase_pt50_a vs card_phase_pt130_a
  tool: "L1 = 0.00e+00 => the intervention had no effect"
  actual: L1 = 0.1296
```

**The root cause has two layers:**

1. The tool read the **precomputed** `l1_vs_group_control` column from the summary table rather than computing it on the fly.
   That column is defined as "relative to a designated reference within the group", and **the reference's own cell is 0**;
   the tool, however, picked the reference as "the first row in the group with bias=0" — **the two "references" were not the same thing**.
2. A deeper layer: **the check "L1=0 ⇒ the intervention failed" is meaningless for a group that had no intervention to begin with.**
   In groups like `phase` / `rep` every bias is 0, and there L1=0 only says "these two rows are the same run".

**Fix**: restrict the check to groups that genuinely contain bias>0, and downgrade "two rows have the same bias" from FAIL to a notice.

> **Lesson**: **a checking tool that produces false positives is more dangerous than no check at all** —
> users lose trust in it and then ignore the real problems along with it.

### 12. Manually inserted rows must be compared field by field against the source data

The Baidu group in the summary table was **appended by hand**. One of its rows ended up as:

```
card_baidu_base: start_step=130, end_step=130, window_steps=0
card actually says: start_step=0,   end_step=130, window_steps=130
```

A window of 0 steps is physically impossible, yet it sat in the table for half a day without anyone noticing — because **nothing checked whether the values in the table agreed with the cards**.

**How it was found**: before release we hashed the whole package, found that the packed version and the archived version disagreed, and only then diffed it row by row.

**Fix**: every hand-entered row must be diffed against the source data right after it is written:

```python
for col, key in [("start_step", "start_step"), ("end_step", "end_step"),
                 ("cv", "expert_load_cv"), ("bias_max", "bias_abs_max")]:
    assert abs(float(row[col]) - float(card[key])) < 1e-9, (col, row[col], card[key])
```

> **Lesson**: a program-generated row with a bug will crash; **a hand-inserted row with a bug only goes quietly wrong**.

### 13. Do not label figures with numbers that can be looked up in a table but are not precise enough to reproduce

Our three figures originally labelled cv in the legend: `A1 (cv=0.3899)`.

The problem is that **cv has three sources**:

```
card expert_load_cv     <- authoritative (computed during training from full-precision load)
summary.csv            <- copied from the card
figure (recomputed)     <- the third source
```

And when `expert_load` was stored into the card it was rounded to 4 digits:
the cv field of `card_pair2B` is **0.3955**, while recomputing from its own array gives **0.395635**.

**So the number in the figure necessarily differs from the table by 1e-4, and a reader comparing them will assume one of them is wrong.**

What is more dangerous: **if someone "fixes" the table on the grounds that "summary.csv is stale", they would turn a correct value into a wrong one.**

**The fix is not to align the three sources, but to stop the figure from acting as a source of numbers** —
drop cv from the legend and keep only the shape of the curves. Readers who want numbers go to the table;
to see "how close these runs actually are", they look at the figure.

> **A figure should be evidence of shape, not a second source of numbers.**

### 14. The intervention happens "at this position", but the effect lands "at the next position"

Same baseline, 96 steps each, `--gamma-sber` raised from 0.2 to 0.8:

| Group | Configuration | cv this session | bias_max | L1 (restored vs no intervention) |
|---|---|---|---|---|
| E1 | none | 0.4737 | 0 | — |
| E2 | sber γ=0.2 | 0.4770 | 0.200195 | 0.0028 |
| E4 | sber γ=0.4 | 0.4769 | 0.400391 | 0.0026 |
| E5 | sber γ=0.8 | **0.4737** | 0.800781 | **0.0000** |
| E6 | γ=0 starting from E5 | **0.2633** | 0.800781 | **0.3980** |

**At γ=0.8 this session is bit-identical to no intervention, and the next session immediately drops by −44%.**

The mechanism is in the code (`train.py:879-901`):

```python
# session boundary: training for this session has ended
dlt  = sber_update(m.router, route_stats, gamma=args.gamma_sber)
perm = reorder_experts(model, route_stats)
```

`sber_update` **prepares for the next session**. **A single-session run with `--gamma-sber` cannot measure this session's effect; that is not a bug.**

> **Lesson: before running an experiment about "the effect within some session", first confirm whether the intervention happens inside that session or at its end.**
> We once concluded "injection at step 50 works" from a run with `gamma=0, bias_abs_max=0` —
> that run had no injection at all.

### 15. Arrays after reordering cannot be compared directly with other groups

The `sber` mode calls `reorder_experts` at the session boundary, **reordering the experts by load**,
so `expert_load` in the card is **post-sort**.

Without restoring `reorder_perm`, the L1 of E2/E4/E5 relative to E1 is **0.51** for all of them (pure permutation distance);
after restoring, it is **0.0028 / 0.0026 / 0.0000**.

**cv is unaffected (it is permutation-invariant), but L1 is completely distorted.**

> **Lesson: before comparing L1 across modes, first restore to a common physical order using `reorder_perm`.**
> This is one more facet of the main claim of this document: **L1 is permutation-sensitive, so the indices must be aligned before it is used.**

### 16. Using window-mixed repeats as the noise turns a feasible-by-8× thing into an infeasible one

There are 5 repeats of the same ckpt (`rep5_rep1..5`), and ce_ema is:

```
rep1  8.9342   130->260
rep2  8.9362   130->260
rep3  8.9351   130->260
rep4  8.9352   130->260
rep5  8.9494   130->259   <- 1 step short; ce_end is an outlier too
```

**Range: 0.0020 for n=4, 0.0152 for n=5 — a factor of 7.6.**

Taking 0.0152 as the noise yields "we would need to amplify 304×, about 3×10⁴ steps, infeasible";
using 0.0020 gives "we would need to amplify 40×, about 3,840 steps, 7 hours".
**A factor of 8, and the conclusion flips from "infeasible" to "feasible but needs scheduling".**

**And the two phrasings do not say the same thing:**

```
with 0.0152: the instrument is too noisy to measure  -> the problem is the instrument
with 0.0020: precise enough, but the effect is only 0.05x -> the problem is the effect itself
```

**The latter is stronger, and more honest.**

> **Lesson: the direction of "conservative" is not "take the larger number", but "take the cleaner number".**
> Noise must be window-aligned first; only then does conservative/aggressive come into it.
> Treating an unclean number as the noise is voluntarily giving up sensitivity.

### 17. Comparing L1 across cards requires restoring `reorder_perm` first

**The `sber` mode calls `reorder_experts` at the session boundary, reordering the experts by load.**
So `expert_load` in the card is the **post-sort** array, not the original expert-identity order.

**Comparing `L1` without restoring compares "the sorting result", not "the allocation itself".**

Measured (the same pair of cards, `g005` vs `g02`):

```
g005 perm [1,5,2,4,3,6,7,0]
g02  perm [1,3,4,5,7,0,6,2]

L1 not restored = 0.0326     <- shape similarity of the two distributions
L1 restored     = 0.0734     <- the same experts were given different shares
```

**More than a factor of two, and the direction makes the conclusion look weaker.**

**How to restore**:
```python
def restore(load, perm):
    """perm[i] = which expert originally sat at post-sort position i"""
    orig = [0] * len(load)
    for i, p in enumerate(perm):
        orig[p] = load[i]
    return orig
```

**Two questions decide which one to use**:
| Asking what | Use which |
|---|---|
| did the same experts get different shares | **after restoring** |
| do the share distributions of two configurations have a similar shape | not restored |

> **What this document argues is "same `cv` but different allocation", and "allocation" means "which expert gets how much",**
> **so we use the restored values.**
>
> **And shape similarity is naturally smaller** — because it throws away the information about "which experts were swapped".


### 18. Before comparing L1 across cards, confirm the two cards share a coordinate system

The `sber` mode calls `reorder_experts` at the session boundary, so `expert_load` in the card is the **post-sort** array;
in the `none` mode `reorder_perm` is `null`, and the array is in the original expert order.

**Adding a `none` card to a `sber` card directly amounts to treating "the i-th largest share" as "expert number i".**

Measured (the `S2cum` group):

```
base   (none, perm=null) : 0.1948 0.1371 0.1347 0.1016 0.2041 0.0631 0.0424 0.1223
S2fix  (sber, perm!=null) not restored: 0.2095 0.1849 ...
S2fix  restored                    : 0.1849 0.1429 0.1427 0.1022 0.2095 0.0536 0.0514 0.1129
```

**Without restoring, L1(measured, predicted) = 0.1891; after restoring, 0.0100 — a factor of 19, and back-solving produces negative values.**

> **Lesson: `L1` is "the distance between two distributions", and both ends must share the same coordinate system.**
> **Restoring only one end is worse than restoring neither — it produces a "plausible-looking" wrong number (0.1805) with no warning at all.**

### 19. Changing one number requires searching for all of its derived values

When we changed `0.0599 -> 0.1077` we changed that one place and missed:

```
the abstract's "L1 differs by 0.0599"     <- changed
the 0.0599 in the §2.1 table             <- changed
"L1 amplified 120×"                      <- missed (should be 215×)
"a ckpt switch costs 0.10"               <- missed (should be 0.25)
"intervention effect 0.06"               <- missed (should be 0.11)
```

**And derived values are by nature not in the paragraph where the original number sits, so the field of view of "change that one place" cannot cover them.**

> **Self-check: after changing any number, search for the number itself plus every derived value computed from it**
> **(multiples, abstract restatements, parenthetical notes, figure titles); all three must be consistent at the same time.**

**This item recurred twice within a single release** (changed a body number but not the abstract; changed a body
heading but not the front page).

**Both times the shape was the same: what changed was the "body"; what was missed was a "non-body location".**
So here is an executable version — **after editing, scan these three places**:

| Place to scan | Why it lags most easily |
|---|---|
| **Front page / release-change block** | written last, revisited least |
| **Abstract and "one-line conclusion"** | they restate the body but belong to no single section |
| **Every table and figure caption** | embedded in the body, easily skipped while editing |

**What to search**: use the old number and the old wording (e.g. "locked", "both platforms", "`N=1`") as keywords,
search each language version — **there should be zero hits**.

**A sturdier approach**: gather the outward-facing numbers in one place (e.g. `numbers.md`) and have the body reference them rather than hard-coding them.

### 20. The silent-downgrade family: you think you are measuring A, but you are measuring B

Once on each platform:

| Platform | Symptom | Harm |
|---|---|---|
| Baidu | `float64` **silently returns 0** | the data are obviously wrong, easy to notice |
| **Ascend** | `float64` **silently downgrades to `float32`** | **the result still looks reasonable — you just measured a whole lot of nothing** |

**The Ascend one is more dangerous**: it raises no error, gives no wrong number, and offers no signal that anything is off.

> **Criterion: any `dtype` / precision assumption must be verified empirically on three points —**
> **① is it accepted ② is it silently rewritten ③ is the rewritten result still plausible.**

**A plausible wrong number is harder to spot than an absurd one.**

### 21. Conclusions that lead to action need stricter verification than conclusions that lead to a stop

We made the same family of mistake twice, in opposite directions:

```
window-mixed noise 0.0152 -> judged "infeasible" -> almost killed something feasible by 8×
  that is "making a stop decision on an unverified number"

3 pairs without repeats   -> judged "linear"   -> then invested 3,840 steps
  that is "making a start decision on an unverified number"
```

> **Verification should be applied to the branch that "leads to the next action", not to the branch that "looks suspicious".**

**The reason: a negative will be challenged; a positive will not.**

### 22. When "same parameters, very different results", check the parameters first, not the theory

We once re-ran with the same parameters and cv went from 0.1535 to 0.2420, worse than the control.

Our reaction at the time was: **to derive the conclusion "repeat noise exceeds the dose effect"**, and on that basis to distrust the whole curve.

**The truth: the second run used a hand-typed injection vector (4 negative, 4 positive), while the first used a script-generated one (3 negative, 5 positive).**

**They were not the same experiment at all.**

> **Criterion: the same parameters run twice giving very different results is itself contrary to common sense.**
> **And things contrary to common sense are usually wrong — check the parameters first, and specifically check where the parameters came from (script-generated vs hand-typed).**


### 23. Multiple distances measured against one shared reference are not independent observations

When we measured "the load difference from switching seeds" we obtained 10 pairwise distances
and reported them as `N=10`. But 4 of those 10 use the same `seed101` as their reference:

> **Terminology**: "pipeline" here means the two versions of the statistics code —
> the **old** one (before 2026-11-19) added historical counts when resuming;
> the **new** one (with `FIX 2026-11-19`) zeroes them on resume and reports only the current session.
> The `seed101` card came from the old version; the other four seeds came from the new one.

```
across pipelines (all referenced to 101): 0.3266 0.3451 0.3607 0.3634   sd = 0.0169
within (102~105 pairwise)               : 0.2752 0.2855 0.3851 0.4303 0.4705 0.4736   sd = 0.0885
```

**It looks like "the across-pipeline group is 5× tighter" — but you cannot conclude
"the pipeline change made the results more consistent" from that.**

Two reasons:

**① Those 4 pairs share one reference and are therefore not independent of each other.**
They are not 4 independent observations; the independent sample is 5 `seeds`, not 10 distances.
Reporting "5 seeds, 10 pairwise distances" is more honest than reporting "`N=10`".

**② The intuition "a shared reference makes the spread smaller" is mathematically false.**

No variance calculation is needed — look at the construction:

```
across distances: d = |X_i - S|
within distances: d = |X_a - X_b|
```

**Both are "the absolute difference of two independent, identically distributed quantities" — the same construction.**
**Their distributions are therefore identical (so their spread is identical), and neither should be tighter in expectation.**

> **The real problem with a shared reference is correlation** (the 4 distances share one `S` and are not
> independent), not a reduced spread.

The 5× difference we actually measured comes from an accidental property of this particular dataset.
**With each `seed` as reference, the span (max−min) of the four distances is:**

| Reference | Four distances | Span |
|---|---|---|
| `101` | 0.3607 / 0.3451 / 0.3266 / 0.3634 | **0.0368** |
| `102` | 0.3607 / 0.4736 / 0.3851 / 0.4303 | 0.1129 |
| `103` | 0.3451 / 0.4736 / 0.2855 / 0.4705 | 0.1881 |
| `104` | 0.3266 / 0.3851 / 0.2855 / 0.2752 | 0.1099 |
| `105` | 0.3634 / 0.4303 / 0.4705 / 0.2752 | 0.1953 |

**The span for `101` (`0.0368`) is between one third and one fifth of the other four (`0.110~0.195`) — that is where the "5×" comes from.**

> **Criterion: whenever you list distances measured from one reference to several points, ask first —**
> **are these distances independent? If they share a reference they are not independent observations,
> and they cannot be reported as `N`, nor compared by spread.**

**This belongs to the family "the construction of the comparison determines the conclusion" —**
along with "comparing `L1` without undoing `perm`" (item 17) and "undoing only one side" (item 18).


### 24. The same `step` index with a different `cv_every` refers to a different window

Each entry in `cv_trace` is the statistic of the window **from `step k-cv_every+1` to `k`**.
**So the same `step` index with a different `cv_every` means two entirely different windows.**

We had three cards with the same `seed` and configuration, whose `cv_every` were 20 / 50 / 100 —
**the comparison below uses only the last two (same `step` index, different windows):**

```
long400 (cv_every=50)  step100 = window  51~100  -> cv 0.2754
long800 (cv_every=100) step100 = window   1~100  -> cv 0.2599
```

**These two differ by 0.0155, which looks like "training is still changing" — in fact only the window start differs**
(the `1~100` window includes the `1~50` stretch, which was still moving).

**After undoing each trace into per-segment windows, the real picture is:**

```
window   1~50  -> 0.2542    <- the only stretch still changing
window  51~100 -> 0.2754
window 101~150 -> 0.2753    <- 0.2753 from here on (within 1e-4, rounding level)
```

> **Criterion: before comparing two `cv_trace`s, check whether `cv_every` is the same;**
> **if not, undo both into per-segment windows (`[k-cv_every+1, k]`) first — otherwise you are comparing
> averages over two windows of different lengths.**

**Fix (adopted)**: store the window bounds `[start, end]` directly in each `cv_trace` entry instead of
only `step`; and always check `cv_every` before any cross-card comparison.

**This also explains why "cumulative vs window" style problems keep appearing —**
`cv_trace` windows, cumulative `expert_load`, and windows with different `cv_every` — **the three scopes
coexist at once**, and the card carries only the single `cv_every` field as a hint, which is easy to miss.

---

## In One Sentence

> **"All metrics green but the thing is broken" is not a bug; it is a class of bug.
> It shows up at the analysis layer, the implementation layer, and the platform layer — the only general defence is "read it back and verify".**

---

### 25. Verifying the artifacts is not the same as verifying the tools

Before releasing v0.2.0 we checked: `zip` size, all 118 `MANIFEST` hashes, the audit script
reporting `errors: 0`, byte-identical regeneration of the three figures, and that the Chinese
and English numbers agree. **All of it passed.**

**But nobody did the one thing that mattered: run the tool shipped in the package against the
data shipped in the package.**

Result: `check_pair.py` computed `L1` **directly on the stored order** for `sber` cards
(it did not undo `reorder_perm`), so it reported 0.2493 as **0.1025** — **the exact number we
had retracted a week earlier.**

```
$ python check_pair.py --a card_rep5_rep1.json --b card_phase_pt130_a.json
L1 : 0.1025      <- wrong (perm not restored)
[PASS]           <- and it passes the check too
```

**Why this class is the hardest to catch**: it does not raise, crash, or produce anything
suspicious. It quietly returns a plausible-looking wrong number — and that number even matched
what our older documents said, so it looked like a successful reproduction.

> **Rule: before releasing, run the packaged tool against the packaged data and compare the
> output line by line with the numbers stated in the text.** Verifying artifacts only shows the
> files are intact; it does not show the tools work.
> More generally: **every claim that is cited needs a path that recomputes it from the published
> artifacts themselves.**

**Fix (adopted)**: `check_pair.py` now carries `restore(load, perm)` and prints a `note:` line
when it restores; `REPRODUCE` gained a section on undoing `reorder_perm` before comparing `L1`.
