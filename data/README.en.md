# Data Notes

This directory contains the session cards and the summary table of the experiment.

| File | Notes |
|---|---|
| `session_cards/` | the session card (JSON) produced by each training run, containing the 8-dimensional `expert_load` array |
| `summary.csv` | the **65-row** summary table (v0.1.5 and earlier batches), one row per run (of which 58 rows have status=ok) |

> **⚠️ The summary table does not cover every card — this is a known inconsistency in this package (v0.2.0).**
>
> This directory currently holds **93 session cards** plus **one file under `logit_dump/` (not a card)**, but `summary.csv` has only **65 rows**.
> The 29 cards that are missing belong to the new batches made after 2026-11-20:
> `card_seed102~105`, `card_bd_M0/MC/M2/M4`, `card_bd_long400/long800/Vfix/Vcum`,
> `card_1118_base_a/b`, `card_aa_*`, `card_c4_E*` and so on.
>
> **Those new cards were generated on demand and never went through the old process of "one row per run into the summary table".**
> So: **to check any number that is not in `summary.csv`, read the corresponding card directly.**
> `check_pair.py` likewise covers only the runs that are inside the summary table (**the `43 pairs / 25 failed` output does not include the new batches**).

## `logit_dump/` (**material for part 2; not cited in this version**)

It contains a single file, `card_bd_dump_logits.json`: one per-layer logit sampling (`batches=3`, `n_layers=28`),
each layer carrying `logit_mean[8]` / `advantage[8]` / `max_adv` / `argmax`.

> **⚠️ It is not a session card. Its format differs from a card and it cannot be parsed as one.**
>
> **This version (part 1: cv vs L1) does not cite it** — the question it serves ("why does a single expert move the
> opposite way under a bias") belongs to part 2 (how to change the load) and is outside this package's scope.
> **It is kept only so the reader can see that something is there; its conclusions and usage are for later versions.**

---

## Summary Table Fields

| Column | Meaning |
|---|---|
| `theme` | experiment theme (dose / repro / ab / long / platform / other) |
| `status` | `ok`, or the reason the run was voided (`void_window_mismatch` etc.) |
| `group` | fine-grained group; L1 is only ever computed within one group |
| `window_steps` | `end_step - start_step` — runs with **different windows cannot be compared directly** |
| `cv` | coefficient of variation (dimensionless) |
| `l1_vs_group_control` | L1 distance to the control in the same group (share units, 0~2) |
| `ce_end` | **the loss of the last batch; cannot be compared across different step counts** |
| `ce_ema` | the smoothed value; **use this one** when comparing across runs |
| `bias_abs_mean` / `bias_abs_max` | the measured intervention strength |

## ⚠️ Recomputing `cv` from `expert_load` gives a slightly different value (≤2.6e-4)

The `expert_load_cv` stored in a card is **computed from full-precision `counts`**;
the `expert_load` array, by contrast, is **rounded to 4 decimal places**.

So: **if you take the 8 numbers in `expert_load` and recompute `cv` yourself, your result will differ from the `expert_load_cv` stored in the card by at most `2.6e-4`.**
(Measured across the 93 cards that carry a `cv`: about `1e-4` on average, `2.58e-4` at most, in `card_c4_E0`.)

> **This comes from rounding; it is not an error.** For example `card_ag_long_s2` has `expert_load_cv = 0.3465`,
> whereas recomputing from its 8 rounded values gives `0.3467` — both are correct, just at different precision.
>
> **The body text always quotes the `expert_load_cv` stored in the card.** If you recompute from the array instead, expect a difference of this magnitude.

---

## Important

- Voided rows in the table are **kept, not deleted**, to document the pitfalls we hit; filter on `status` to exclude them.
- **`status = ok` does not mean "the quality is usable"**: two early smoke runs (`card_none_bf16`, `card_tk2_none`)
  have status ok, but their routing had already collapsed — **they are only good as counter-examples**.
- A row with `l1_vs_group_control` left blank means "no same-group control was designated". **It does not mean 0.**
- `l1_vs_group_control = 0` means that row **is the group's own reference row** (compared with itself, so L1 is always 0). It does not mean "the two groups are the same".
