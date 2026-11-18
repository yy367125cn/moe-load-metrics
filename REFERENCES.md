# 参考文献

> **核验范围（请按此判断可信度）**：
> - 六条文献的**题名与 arXiv 编号**逐条核对，全部一致；
> - 正文引用的 **[1] [2] [3]** 的**结论表述**已与各自摘要比对，未见偏差；
> - **背景三条（Switch / GShard / Mixtral）仅核编号与题名**，正文未使用其结论。
>
> 若发现转述不准确，请提 issue —— 这比错误传播出去好。

## 与本工作直接相关

**[1]** *MoE Pretraining: Parallelism Layout Changes the Model, Not Quality*
arXiv:2611.15271，2026-11-11
https://arxiv.org/abs/2611.15271

> 换并行布局不改质量、但改模型（>90% 路由决策翻转，集中在 near-tie token）；
> 切换时机影响很大（84% 处几乎无影响）。
>
> **本文与之互补**：他们换并行布局，我们换一次训练（同 seed / 同配置 / 同数据）；
> 两者都让路由状态大改，而任务质量不变。

**[2]** *Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers*
arXiv:2510.11370
https://arxiv.org/abs/2510.11370

> 即使条件完全相同，重复 forward 也可能给出不同的专家选择；
> 建议记录路由分布并重放。**（语境是 RL 训练，不是通用预训练。）**
>
> **本文给出量级**：同 ckpt 内 Ascend 1e-4~6e-4，百度 0（逐位相同）。

---

## 技术来源（**必读**）

**[3]** *DeepSeek-V3 Technical Report*，arXiv:2412.19437
https://arxiv.org/abs/2412.19437

> **本文用到的 router bias 注入方法即出自此文**：
> "aux-loss-free" 负载均衡——给每个专家维护一个 bias，
> 按负载偏离符号更新，只影响 top-k 选择、不进入梯度。
> 见 `REPRODUCE.md` 第 3 步与 `README.md` 第六节。
>
> **本文不是提出该方法，而是测量"用什么指标评价它"。**

## 其它背景（正文未直接引用，仅供定位）

| 主题 | 出处 |
|---|---|
| Switch Transformer（capacity factor 的由来） | arXiv:2101.03961 |
| GShard（MoE 分布式训练奠基） | arXiv:2006.16668 |
| Mixtral 8x7B（开源 MoE 代表） | arXiv:2401.04088 |

---

## 引用格式

    moe-load-metrics: MoE 专家负载指标的精度与适用范围
    (2026-11-18)
