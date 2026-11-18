#!/usr/bin/env python3
"""cv 与 L1 的对比 —— 最小可运行脚本。

用法：
    python metrics.py --cards ./cards            # 从 session card 目录算汇总
    python metrics.py --demo                     # 用内置数据演示

两个指标：
    cv(x)    = std(x) / mean(x)                       —— 集散程度，无量纲，对置换不敏感
    L1(a, b) = sum(|a[i] - b[i]|)                     —— 份额分布距离，范围 0~2（a、b 和为 1）

为什么需要两个：
    「专家0拿40%、专家7拿5%」和「专家0拿5%、专家7拿40%」的 cv 完全相同，
    但负载落在谁身上完全不同。L1 看得见后者，cv 看不见。
"""
import argparse
import glob
import json
import os
import sys


def cv(load):
    """变异系数。load 为各专家负载（份额或计数均可，只要求非负且和 > 0）。"""
    n = len(load)
    if n == 0:
        return 0.0
    mean = sum(load) / n
    if mean <= 0:
        return 0.0
    var = sum((x - mean) ** 2 for x in load) / max(1, n - 1)
    return (var ** 0.5) / mean


def l1(a, b):
    """两个份额分布的距离。前提：len(a) == len(b)，且各自和约为 1。

    返回 0~2。若输入不是份额（比如原始计数），结果无单位意义 —— 需要先归一化。
    """
    if len(a) != len(b):
        raise ValueError("长度不一致: %d vs %d" % (len(a), len(b)))
    return sum(abs(x - y) for x, y in zip(a, b))


def normalize(v):
    s = sum(v)
    return [x / s for x in v] if s > 0 else list(v)


def load_cards(card_dir):
    """读目录下所有 *.json 的 session card，返回 [(文件名, card dict)]。"""
    out = []
    for p in sorted(glob.glob(os.path.join(card_dir, "*.json"))):
        try:
            with open(p, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            if isinstance(d, dict) and "expert_load" in d:
                out.append((os.path.basename(p), d))
        except Exception as e:
            print("[warn] 跳过 %s: %s" % (p, e), file=sys.stderr)
    return out


def summarize(cards):
    print("%-40s %8s %8s %10s %10s" % ("card", "cv", "cv_soft", "ce_ema", "steps"))
    print("-" * 82)
    for name, d in cards:
        load = d.get("expert_load") or []
        steps = ""
        if "start_step" in d and "end_step" in d:
            steps = "%s->%s" % (d.get("start_step"), d.get("end_step"))
        print("%-40s %8.4f %8.4f %10.4f %10s"
              % (name[:40],
                 cv(load) if load else float("nan"),
                 d.get("expert_load_cv_soft", float("nan")),
                 d.get("ce_loss_ema", float("nan")),
                 steps))


def demo():
    """内置演示：证明 cv 对置换不敏感而 L1 敏感。"""
    a = [0.40, 0.25, 0.15, 0.10, 0.05, 0.03, 0.01, 0.01]
    b = list(reversed(a))
    print("分布 a        :", ["%.2f" % x for x in a])
    print("分布 b（反转）:", ["%.2f" % x for x in b])
    print()
    print("cv(a) = %.6f" % cv(a))
    print("cv(b) = %.6f      <- 完全相同" % cv(b))
    print("L1(a,b) = %.4f          <- 明显不同" % l1(a, b))
    print()
    print("=> cv 看不见「谁拿哪份」，L1 看得见。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", default="", help="session card 目录")
    ap.add_argument("--demo", action="store_true", help="用内置数据演示 cv 与 L1 的差别")
    args = ap.parse_args()

    if args.demo or not args.cards:
        demo()
        return
    cards = load_cards(args.cards)
    if not cards:
        print("没有找到带 expert_load 的 card", file=sys.stderr)
        sys.exit(1)
    summarize(cards)


if __name__ == "__main__":
    main()
