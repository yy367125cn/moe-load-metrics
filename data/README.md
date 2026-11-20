# 数据说明

本目录包含实验的 session card 与汇总表。

| 文件 | 说明 |
|---|---|
| `session_cards/` | 每次训练产出的 session card（JSON），含 `expert_load` 8 维数组 |
| `summary.csv` | **65 行**汇总表（v0.1.5 及更早批次），一行一个 run（其中 status=ok 的 58 行） |

> **⚠️ 汇总表没有覆盖全部卡片 —— 这是本包已知的不一致（v0.2.0）。**
>
> 本目录现有 **93 张 session card** + **1 个 `logit_dump/` 下的文件（不是 card）**，但 `summary.csv` 只有 **65 行**。
> 差的 29 张是 2026-11-20 之后的新批次：
> `card_seed102~105`、`card_bd_M0/MC/M2/M4`、`card_bd_long400/long800/Vfix/Vcum`、
> `card_1118_base_a/b`、`card_aa_*`、`card_c4_E*` 等。
>
> **这些新卡是按需生成的，没有走"每 run 一行进汇总表"的旧流程。**
> 因此：**核 `summary.csv` 之外的数字，请直接读对应的 card。**
> `check_pair.py` 同样只覆盖汇总表内的 run（**43 对 / 25 失败**这个输出不包含新批次）。

## `logit_dump/`（**第 2 版的素材，本版不引用**）

里面只有一个文件 `card_bd_dump_logits.json`：一次逐层 logit 采样（`batches=3`, `n_layers=28`），
每层含 `logit_mean[8]` / `advantage[8]` / `max_adv` / `argmax`。

> **⚠️ 它不是 session card，格式与 card 不同，不能按 card 解析。**
>
> **本版（第 1 部分：cv vs L1）不引用它** —— 它服务的那个问题（"为什么某个专家在 bias 下会反向"）
> 属于第 2 部分（怎么改变负载），不在本包范围内。
> **保留它只是为了让读者看到"那里有东西"；结论与用法见后续版本。**

---

## 汇总表字段

| 列 | 含义 |
|---|---|
| `theme` | 实验主题（dose / repro / ab / long / platform / other） |
| `status` | `ok` 或作废原因（`void_window_mismatch` 等） |
| `group` | 细分组，L1 只在同一 group 内计算 |
| `window_steps` | `end_step - start_step` —— 不同窗口的 run **不可直接比** |
| `cv` | 变异系数（无量纲） |
| `l1_vs_group_control` | 与同组对照的 L1 距离（份额单位，0~2） |
| `ce_end` | **最后一批的 loss，不可跨步数比较** |
| `ce_ema` | 平滑值，跨 run 比较请用这个 |
| `bias_abs_mean` / `bias_abs_max` | 干预强度实测值 |

## ⚠️ 从 `expert_load` 重算 `cv` 会有小偏差（≤2.6e-4）

卡片的 `expert_load_cv` 是**用全精度 `counts` 算出来的**；
而 `expert_load` 数组**圆整到了 4 位小数**。

因此：**用 `expert_load` 的 8 个数重算 `cv`，会与卡片里存的 `expert_load_cv` 差 ≤ `2.6e-4`**
（93 张带 `cv` 的卡实测：平均约 `1e-4`，最大 `2.58e-4`，出现在 `card_c4_E0`）。

> **这是舍入造成的，不是错误。** 例如 `card_ag_long_s2` 的 `expert_load_cv = 0.3465`，
> 而用它的 8 个圆整值重算得 `0.3467` —— 两者都对，只是精度不同。
>
> **本文正文一律引用卡片里存的 `expert_load_cv`。** 若你从数组重算，请预期这个量级的差。

---

## 重要

- 表中的作废行**保留未删**，用于说明踩过的坑；用 `status` 过滤。
- **`status = ok` 不等于"质量可用"**：有两条早期冒烟 run（`card_none_bf16`、`card_tk2_none`）
  虽然 ok，但路由已塌缩，**只适合当反例用**。
- `l1_vs_group_control` 留空的行表示"未指定同组对照"，**不是 0**。
- `l1_vs_group_control = 0` 表示该行**就是同组参照行本身**（自己与自己比，L1 恒为 0），不代表"两组一样"。
