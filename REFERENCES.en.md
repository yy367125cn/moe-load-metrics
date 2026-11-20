# References

> **Scope of verification (use this to judge how far to trust them)**:
> - the **titles and arXiv identifiers** of all eight references were checked one by one, and all agree;
> - the **wording of the conclusions** of **[1] [2] [3]**, as cited in the body, has been compared against their respective abstracts, with no deviation found;
> - **for the three background references (Switch / GShard / Mixtral) only the identifiers and titles were checked**, and their conclusions are not used in the body.
>
> If you find an inaccurate paraphrase, please open an issue — that is better than letting the error spread.

## Directly Related to This Work

**[1]** *MoE Pretraining: Parallelism Layout Changes the Model, Not Quality*
arXiv:2611.15271, 2026-11-11
https://arxiv.org/abs/2611.15271

> Changing the parallelism layout leaves quality unchanged but changes the model (>90% of routing decisions flip, concentrated in near-tie tokens);
> the timing of the switch matters a great deal (almost no effect at 84%).
>
> **This document is complementary to it**: they change the parallelism layout, we run training one more time (same seed / same configuration / same data);
> both leave routing state substantially changed while task quality is unchanged.

**[2]** *Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers*
arXiv:2510.11370
https://arxiv.org/abs/2510.11370

> Even under identical conditions, repeated forward passes may choose different experts;
> they suggest recording the routing distribution and replaying it. **(The context is RL training, not general pretraining.)**
>
> **This document supplies the magnitude**: within one ckpt, Ascend 1e-4~6e-4, Baidu 0 (bit-identical).

---

## Sources of the Technique (**Required Reading**)

**[3]** *DeepSeek-V3 Technical Report*, arXiv:2412.19437
https://arxiv.org/abs/2412.19437

> **The router-bias injection method used here comes from this paper**:
> "aux-loss-free" load balancing — maintain a bias for each expert,
> update it along the sign of the load deviation, affecting only top-k selection and never entering the gradient.
> See step 3 of `REPRODUCE.md` and Section 6 of `README.md`.
>
> **This document does not propose the method; it measures "which metric should be used to evaluate it".**


**[4]** *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*, arXiv:2606.19348
https://arxiv.org/abs/2606.19348

> **Relation to this work**: the skeleton of this method (expert-wise bias + updating along the sign of the load) is retained in V4,
> but load balancing has become the combination "aux-loss-free + a small amount of sequence-wise balance loss",
> and the affinity activation changed from Sigmoid to Sqrt(Softplus).
> **This document measures the pure aux-loss-free mode and does not cover that variant.**

**[5]** *DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression*, arXiv:2609.19969
https://arxiv.org/abs/2609.19969

> **Relation to this work**: under multimodality, a correction bias is maintained separately per modality (a direct extension of this method).
> **This document does not cover that variant.**

## Other Background (not cited directly in the body; for orientation only)

| Topic | Source |
|---|---|
| Switch Transformer (where the capacity factor comes from) | arXiv:2101.03961 |
| GShard (foundational work on distributed MoE training) | arXiv:2006.16668 |
| Mixtral 8x7B (a representative open-source MoE) | arXiv:2401.04088 |

---

## Citation Format

    moe-load-metrics: MoE 专家负载指标的精度与适用范围
    (English: MoE Expert-Load Metrics: A Precision Comparison of cv and L1)
    (2026-11-18)
