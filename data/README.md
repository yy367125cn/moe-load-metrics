# 数据说明

本目录包含实验的 session card 与汇总表。

| 文件 | 说明 |
|---|---|
| `session_cards/` | 每次训练产出的 session card（JSON），含 `expert_load` 8 维数组 |
| `summary.csv` | **65 行**汇总表，一行一个 run（其中 status=ok 的 58 行） |

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

## 重要

- 表中的作废行**保留未删**，用于说明踩过的坑；用 `status` 过滤。
- **`status = ok` 不等于"质量可用"**：有两条早期冒烟 run（`card_none_bf16`、`card_tk2_none`）
  虽然 ok，但路由已塌缩，**只适合当反例用**。
- `l1_vs_group_control` 留空的行表示"未指定同组对照"，**不是 0**。
- `l1_vs_group_control = 0` 表示该行**就是同组参照行本身**（自己与自己比，L1 恒为 0），不代表"两组一样"。
