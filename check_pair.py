#!/usr/bin/env python3
"""check_pair.py —— 配对实验的检查工具。

把"我们踩过的三个坑"变成"程序会拦住"：

  坑 1：两组不是同一个基线（换 ckpt 差 0.10，比效应还大）
  坑 2：两组窗口长度不同（96 步 vs 85 步，不可比）
  坑 3：干预没生效（bias 被清零或覆盖，两组其实一模一样）

用法：
    # 推荐：读汇总表，按 group 分组检查
    python check_pair.py --csv data/summary.csv

    # 单对检查（两个 card 文件）
    python check_pair.py --a card_A.json --b card_B.json

退出码：0 全部通过；1 有检查失败。
"""
import argparse
import csv
import json
import os
import sys


def cv(load):
    n = len(load)
    if n == 0:
        return 0.0
    m = sum(load) / n
    if m <= 0:
        return 0.0
    var = sum((x - m) ** 2 for x in load) / max(1, n - 1)
    return (var ** 0.5) / m


def l1(a, b):
    if len(a) != len(b):
        raise ValueError("长度不一致: %d vs %d" % (len(a), len(b)))
    return sum(abs(x - y) for x, y in zip(a, b))


def _restore(load, perm):
    """按 reorder_perm 把「降序存储」还原成原始专家顺序。

    sber 模式的卡片里 expert_load 按负载降序存储，直接用存储顺序算 L1 会得到
    错误的结果（本仓库 v0.1.5 曾因此发布两个错数：0.1025 应为 0.2493、
    0.0599 应为 0.1077）。perm 为 null 时原样返回。
    """
    if not perm:
        return list(load)
    if len(perm) != len(load):
        raise ValueError("perm 长度 %d 与 load 长度 %d 不一致" % (len(perm), len(load)))
    out = [0.0] * len(load)
    for i, p in enumerate(perm):
        out[int(p)] = load[i]
    return out

def _f(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def check_one(row, ref, name, ref_name, has_treatment=False, is_phase=False):
    """用汇总表的一行与参照行比较。返回问题列表。

    has_treatment=True 表示组内存在 bias>0 的成员（即这是干预实验），
    此时才要求"必须有一个 bias=0 的对照"。
    is_phase=True 表示这组本来就是比较不同起点的（如 pt50 vs pt130），
    此时 start_step 不同是预期行为，不报错。
    """
    p = []
    notes = []

    # 1) 同一个基线（phase 组除外）
    a, b = row.get("start_step"), ref.get("start_step")
    if not is_phase and a != b:
        p.append("start_step 不同（%s vs %s）=> 不是同一个基线 ckpt" % (a, b))

    # 2) 同一窗口
    wa, wb = row.get("window_steps"), ref.get("window_steps")
    if wa != wb:
        p.append("窗口长度不同（%s vs %s 步）=> 指标不可直接比较" % (wa, wb))

    # 3) 同一 seed
    if row.get("seed") != ref.get("seed"):
        p.append("seed 不同（%s vs %s）" % (row.get("seed"), ref.get("seed")))

    # 4) 若这是干预实验（组内有 bias>0），必须存在 bias=0 的对照
    ba, bb = _f(row.get("bias_mean")), _f(ref.get("bias_mean"))
    if has_treatment:
        if ba != 0 and bb != 0 and abs(ba - bb) < 1e-6:
            # 两行都被注入了同样的 bias：这不是对照，但也不是错误（可能是重复实验）
            notes.append("两组 bias_mean 相同（%.6g）=> 本次比较不是「干预 vs 对照」" % ba)
    # 若整组都是 bias=0，那是"重复实验"，bias 全零是预期的，不报错

    # 5) 干预的数值量级
    mx = max(abs(ba), abs(bb))
    if 0 < mx < 1e-6:
        p.append("bias 非零但极小（%.3g）=> 检查是否被清零" % mx)

    # 6) L1 不能为零（否则两组其实一样）
    # ⚠️ 只在有干预的组里检查：phase / rep 这类组本来就没有干预（bias 全 0），
    # 那里 L1 = 0 只说明"这两行是同一条"，不是"干预失败"。
    la = _f(row.get("l1_vs_group_control"), None)
    if has_treatment and la is not None and la < 1e-4:
        p.append("L1 = %.2e，两组几乎完全相同 => 干预很可能没生效" % la)

    return p, notes


def run_csv(path, show_all=False):
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    groups = {}
    for r in rows:
        if (r.get("status") or "ok") != "ok":
            continue                       # 跳过已标记作废的
        groups.setdefault(r.get("group") or "ungrouped", []).append(r)

    n_fail = 0
    n_checked = 0
    for g, items in sorted(groups.items()):
        if len(items) < 2:
            continue
        # 参照：组内 bias_mean == 0 的那一条（对照）
        has_treatment = any(_f(r.get("bias_mean")) != 0 for r in items)
        is_phase = "phase" in g.lower()
        ref = next((r for r in items if _f(r.get("bias_mean")) == 0), items[0])
        ref_name = ref.get("file", "?")
        for r in items:
            if r is ref:
                continue
            n_checked += 1
            probs, _notes = check_one(r, ref, r.get("file", "?"), ref_name,
                                      has_treatment=has_treatment, is_phase=is_phase)
            if probs:
                n_fail += 1
                print("\n[FAIL] %s vs %s" % (r.get("file"), ref_name))
                for q in probs:
                    print("       - " + q)
            elif show_all:
                print("[PASS] %s vs %s" % (r.get("file"), ref_name))
            elif show_all:
                print("[PASS] %s vs %s" % (r.get("file"), ref_name))
    print("\n检查 %d 对，失败 %d 对。" % (n_checked, n_fail))
    return 0 if n_fail == 0 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="", help="汇总表 csv（按 group 分组检查）")
    ap.add_argument("--a", default="")
    ap.add_argument("--b", default="")
    ap.add_argument("--v", action="store_true", help="也打印通过的组")
    args = ap.parse_args()

    if args.csv:
        sys.exit(run_csv(args.csv, args.v))

    if args.a and args.b:
        with open(args.a, "r", encoding="utf-8") as fh:
            A = json.load(fh)
        with open(args.b, "r", encoding="utf-8") as fh:
            B = json.load(fh)
        # 从 card 里取字段（card 没有 group，但有 start_step/end_step/seed/bias）
        def to_row(c, name):
            return {
                "file": name,
                "start_step": c.get("start_step"),
                "window_steps": c.get("end_step", 0) - c.get("start_step", 0),
                "seed": c.get("seed"),
                "bias_mean": c.get("bias_abs_mean"),
            }
        ra, rb = to_row(A, os.path.basename(args.a)), to_row(B, os.path.basename(args.b))
        ref, ref_name = (rb, rb["file"]) if _f(rb["bias_mean"]) == 0 else (ra, ra["file"])
        other = ra if ref is rb else rb
        probs, _notes = check_one(other, ref, other["file"], ref_name)
        la = _restore(A.get("expert_load") or [], A.get("reorder_perm"))
        lb = _restore(B.get("expert_load") or [], B.get("reorder_perm"))
        if la and lb:
            print("cv : %.4f / %.4f" % (cv(la), cv(lb)))
            if A.get("reorder_perm") or B.get("reorder_perm"):
                print("note: 已按 reorder_perm 还原后再算 L1（sber 卡片的 expert_load 是降序存储）")
            print("L1 : %.4f" % l1(la, lb))
        for q in probs:
            print("[FAIL] " + q)
        if not probs:
            print("[PASS]")
        sys.exit(0 if not probs else 1)

    ap.error("需要 --csv 或 --a/--b")


if __name__ == "__main__":
    main()
