# 复现指南

## 三条硬条件（缺一不可）

1. **同一个 ckpt 文件** —— 不是"同 seed"，不是"同配置"。
   我们实测：同 seed / 同数据 / 同步数重新训练一次，路由状态与上一次的 L1 距离是 **0.10**，
   和要测的干预效应（0.06）同量级甚至更大。
2. **同一个窗口长度** —— 不是"差不多"。
   96 步与 85 步的 run **不可放在同一坐标系里比较**。
3. **干预量由当前基线自己的负载现算** —— 不是复用别的基线的值。

> 这三条是我们踩过坑之后总结的。不写，复现必然失败。

---

## 最小复现步骤

### 第 0 步：环境

```bash
pip install torch numpy matplotlib
```

### 第 1 步：做一个基线 ckpt

用任意 MoE 训练代码训练 N 步（我们用的是 130 步），保存 ckpt。
**记录它的 `expert_load`（8 个专家的负载份额）。**

### 第 2 步：复制成两份

```bash
cp -r base_ckpt A_ckpt
cp -r base_ckpt B_ckpt
```

### 第 3 步：算干预量（只对 A）

```python
import torch

E = 8
gamma = 0.2                                   # 干预强度
frac = torch.tensor([...])                     # 第 1 步记录的 8 维负载，和为 1
delta = gamma * torch.sign(1.0 / E - frac)     # 每个专家一个数
# 写入 A 的 router bias
```

**关键：`delta` 必须用这一个基线自己的 `frac` 算。** 用别的基线的 `delta` 等于把别人的解套在你的问题上。

> **方法来源**：这个 `sign(1/E - frac)` 的更新式即 DeepSeek-V3 的 aux-loss-free 负载均衡
> （arXiv:2412.19437）。本文只测量指标，不提出方法。

### 第 4 步：两份各训同样步数

```
A：从 A_ckpt 出发，训练 M 步（我们用的是 96~128 步）
B：从 B_ckpt 出发，训练 M 步
```

⚠️ **两份必须用同一个 M。** 窗口不同，结果不可比。

### 第 5 步：算指标

```bash
python metrics.py --cards ./cards
```

或直接：

```python
from metrics import cv, l1

cv_A  = cv(load_A)          # 变异系数
dist  = l1(load_A, load_B)  # 份额距离，0~2
```

### 第 6 步：读懂结果

| 现象 | 含义 |
|---|---|
| `l1` 约 0.06，`cv` 只差 1.5% | **符合我们的观察**：cv 几乎看不见，L1 看得见 |
| `l1` 约 1e-4 | 干预没生效（检查 bias 是否真的写进了模型） |
| `l1` 约 0.10 但 `cv` 也差很多 | 检查两组的 ckpt 是不是同一个文件 |

---

## 怎么确认"干预真的生效了"

**不要相信"我设置了"。要读回来。**

```python
print("bias_abs_mean =", model.router.bias.abs().mean().item())
# 期望：A 约等于 gamma，B 等于 0
```

我们踩过两次：
1. `router_mode=none` 的 forward **根本不读 bias**，而且 `set_router_mode` 会清零
2. 从 ckpt 恢复时，`load_state_dict` 会把 bias **覆盖回 0**

**两次都是"日志显示设置成功，实际模型里是 0"。**

---

## 画图

```bash
python make_figures.py --data data_for_figures.json --out figures
```

`data_for_figures.json` 的结构见文件内注释。


---

## 自动检查：check_pair.py

**在跑任何一个配对实验之前，先跑这个。**

```bash
python check_pair.py --csv data/summary.csv      # 按 group 分组检查
python check_pair.py --a card_A.json --b card_B.json   # 单对检查
```

它检查六项：

| # | 检查 | 为什么 |
|---|---|---|
| 1 | 两组 `start_step` 相同 | 否则不是同一个基线 ckpt |
| 2 | 两组 `window_steps` 相同 | 不同窗口的指标不可比 |
| 3 | 两组 `seed` 相同 | 排除数据顺序差异 |
| 4 | 干预组有 `bias_mean > 0`，且存在 `bias_mean = 0` 的对照 | 否则"干预"或"对照"缺一个 |
| 5 | `L1` 不为 0 | 若为 0，两组其实一模一样 ⇒ 干预没生效 |
| 6 | 干预组的 `bias_mean` 不为极小值（>1e-6） | 非零但极小 ⇒ 可能被清零 / 未真正写入 |

### ⚠️ 在历史数据上跑，会报很多 FAIL —— 这是**预期**的

```
检查 43 对，失败 25 对。
```

**大部分不是脚本有 bug，是那些数据本来就不满足配对条件。**
（早期版本报 32 对，其中 7 对是**工具自身误报**——例如对"本来就没有干预"的组仍检查 L1=0。
已修正，见 `PITFALLS.md` 第 11 条。）

我们的汇总表跨越了一周的探索过程，里面混着：
- 跨天的 A/B（窗口不同）
- 不同起点的阶段对比（start_step 本就不同）
- 注入失败的对照组（bias 被清零）

**这正是我们开源这个工具的原因**：
不做检查的时候，你分不清"哪些数据能比、哪些不能"——
而这些 FAIL 就是"不能比"的清单。

> **文档靠人读，脚本靠机器执行。**
> 把一个坑写成规则，比写成一段话有用得多。

---

## 跨卡比较 `L1` 之前：必须先还原 `reorder_perm`

**这是 v0.1.5 曾发布两个错数的直接原因，也是本仓库最容易犯的一类错。**

`sber` / `baseline` 等模式的卡片里，`expert_load` 存的是**按负载降序排好的数组**，
`reorder_perm` 记录这个排序。直接按存储顺序算 `L1`，量到的不是「两套分配的距离」，
而是一个被排序抹掉身份后的距离。

```python
def restore(load, perm):
    if not perm:
        return list(load)
    out = [0.0] * len(load)
    for i, p in enumerate(perm):
        out[int(p)] = load[i]
    return out
```

| 量 | 未还原 | 还原后 |
|---|---|---|
| 干预效应 | 0.0599 | **0.1077** |
| 跨 `ckpt` 差异 | 0.1025 | **0.2493** |

`check_pair.py` 自 v0.2.0 起已内置还原，并在还原时打印一行 `note:`。
你若要自己复算，请用同一个 `restore`。

---

## 复现本版新增的三组证据

### A. 平台期（§3a）

```bash
# 同一个 seed，训三个长度（其余参数完全相同）
python train.py --data data --out exp/L130 --seed 101 --max-steps 130 --cv-every 20
python train.py --data data --out exp/L400 --seed 101 --max-steps 400 --cv-every 50
python train.py --data data --out exp/L800 --seed 101 --max-steps 800 --cv-every 100
```

**读 `cv_trace` 时先看 `cv_every`**：`step k` 的含义是窗口 `[k-cv_every+1, k]`。
**不同 `cv_every` 的两条 trace，`step` 编号相同不代表同一段**（见 `PITFALLS` 第 24 条）。

预期：`1~20` 为 0.2427 → `21~40` 过冲 0.3286 → `41~60` 0.2800 → `61~80` 0.2754 → `81` 步起恒 0.2753。

### B. 跨 seed 稳健性

```bash
for s in 101 102 103 104 105; do
  python train.py --data data --out exp/seed$s --seed $s --max-steps 130
done
```

把五张卡的 `expert_load` 两两算 `L1`（10 对）：预期落在 **0.2752 ~ 0.4736**，中位数 0.3620。

### C. 累计口径公式

```bash
# 同一个 base ckpt，分两臂续训 96 步：
#   一臂续训时清零统计（默认行为）
#   一臂续训时累加上一程（去掉清零那行）
```

预测：`cum_expected = (n1*L_base + n2*L_session) / (n1+n2)`；
实测与预测的 `L1` 应 **< 0.005**（本仓库实测 0.0003）。

---

## 判定一张卡是否自洽（零成本）

任何「从零训练 + 有 `cv_trace` + 存累计口径」的卡，都可以自查：

1. 由 `expert_load`（累计）反解区间窗口分布；
2. 算该分布的 `cv`；
3. 与 `cv_trace` 里对应窗口的 `cv` 比，差 < 1e-3 即自洽。

> **注意：自洽只能说明「公式/统计口径」没问题，不能说明「训练本身没错」——后者要靠重跑对照。**
