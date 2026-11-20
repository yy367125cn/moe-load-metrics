#!/usr/bin/env python3
"""三张图：旋转 / 噪声标尺 / 配对信噪比。

数据来自 sessions card 的 expert_load（8 维数组）。
运行： python make_figures.py --data data_for_figures.json --out figures/
"""
import argparse
import json
import os


def _setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # 中文字体：优先用系统里存在的，找不到就退回英文标签
    for f in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"):
        if any(f == x.name for x in matplotlib.font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [f]
            break
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def fig_rotation(data, outdir, plt):
    """图 3：cv 不动，但负载排序全换。"""
    d = data["fig3_rotation"]
    keys = [k for k in ("step0", "s1", "s2") if d.get(k)]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = list(range(8))
    markers = ["o", "s", "^"]
    colors = ["#888", "#1f77b4", "#d62728"]
    for i, k in enumerate(keys):
        ax.plot(x, d[k], marker=markers[i % 3], color=colors[i % 3],
                label=k, linewidth=1.6)
    ax.set_xticks(x)
    ax.set_xticklabels(["E%d" % i for i in x])
    ax.set_ylabel("负载份额 / expert load share")
    # 标题数字一律现算
    if d.get("step0") and d.get("s1"):
        dc = (_cv(d["s1"]) - _cv(d["step0"])) / _cv(d["step0"]) * 100
        dl = _l1(d["step0"], d["s1"])
        ax.set_title("图 3 / Fig. 3  first 400 steps: cv %+.0f%%, but the load distribution moved by L1=%.2f" % (dc, dl))
    else:
        ax.set_title("图 3 / Fig. 3  The load distribution changes over training")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(outdir, "fig3_rotation.png")
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_noise(data, outdir, plt):
    """图 1：同 ckpt 内几乎重合，换一个 ckpt 就完全不同。"""
    d = data["fig1_noise"]
    # 标题与图例里的数字一律从数据现算，不硬编码
    runs = d.get("same_ckpt_runs", [])
    l1_same = 0.0
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            l1_same = max(l1_same, _l1(runs[i], runs[j]))
    l1_other = ""
    if d.get("other_ckpt") and runs:
        l1_other = "; a different ckpt differs by %.4f" % _l1(runs[0], d["other_ckpt"])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = list(range(8))
    for i, arr in enumerate(d.get("same_ckpt_runs", [])):
        ax.plot(x, arr, color="#1f77b4", alpha=0.45, linewidth=1.2,
                label="same ckpt (Ascend, 5 runs)" if i == 0 else None)
    if d.get("baidu_same"):
        for i, arr in enumerate(d["baidu_same"]):
            ax.plot(x, arr, color="#2ca02c", alpha=0.45, linewidth=1.2,
                    label="same ckpt (Baidu, 2 runs, bit-identical)" if i == 0 else None)
    if d.get("other_ckpt"):
        ax.plot(x, d["other_ckpt"], color="#d62728", marker="o", linewidth=1.8,
                label="different ckpt")
    ax.set_xticks(x)
    ax.set_xticklabels(["E%d" % i for i in x])
    ax.set_ylabel("负载份额 / expert load share")
    # 标题里的数字一律从数据现算，不硬编码
    runs = d.get("same_ckpt_runs", [])
    l1_same = 0.0
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            l1_same = max(l1_same, _l1(runs[i], runs[j]))
    l1_other = ""
    if d.get("other_ckpt") and runs:
        l1_other = "; a different ckpt differs by %.4f" % _l1(runs[0], d["other_ckpt"])
    ax.set_title("图 1 / Fig. 1  within one ckpt: max L1=%.1e%s" % (l1_same, l1_other))
    # 与 fig2 / fig3 统一：坐标轴内右上角
    # 但 fig1 的 y 上限要留白，否则图例会压住 E7 那个点（绿线末端 0.175）
    ax.set_ylim(top=0.215)
    ax.legend(fontsize=8.5, loc="upper right")
    # 两平台基线不同，绿蓝两组本就不该重合 —— 必须写明，否则会被误读
    ax.text(0.5, -0.16, "The two platforms start from different baseline load distributions, so the two same-ckpt "
                        "curves are not expected to coincide.\nWhat is comparable is only the spread within each group.",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="#444")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(outdir, "fig1_noise_scale.png")
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_pair(data, outdir, plt):
    """图 2：组内两次几乎重合，A vs B 明显分开。"""
    d = data["fig2_pair"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = list(range(8))
    for name, color in (("A1", "#d62728"), ("A2", "#ff9896"),
                        ("B1", "#1f77b4"), ("B2", "#aec7e8")):
        if d.get(name):
            ax.plot(x, d[name], marker="o", color=color, linewidth=1.5,
                    label=name)
    # 旧值（未还原坐标系）用虚线做对照 —— 让读者看到"修正了什么"
    a1r, b1r = d.get("A1_raw"), d.get("B1_raw")
    if a1r and b1r:
        ax.plot(x, a1r, "--", color="#d62728", alpha=0.35, linewidth=1.2,
                label="A1, before permutation undo")
        ax.plot(x, b1r, "--", color="#1f77b4", alpha=0.35, linewidth=1.2,
                label="B1, before permutation undo")
    ax.set_xticks(x)
    ax.set_xticklabels(["E%d" % i for i in x])
    ax.set_ylabel("负载份额 / expert load share")
    ax.set_title("图 2 / Fig. 2  Paired run: the two repeats coincide, the two groups separate")
    if a1r and b1r:
        ax.text(0.5, -0.15,
                "Solid: after undoing reorder_perm (L1(A,B)=0.1077).  "
                "Dashed: the raw stored order (L1=0.0599), shown only to make the correction visible.",
                transform=ax.transAxes, ha="center", fontsize=7.5, color="#555")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(outdir, "fig2_pair.png")
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return p


def _l1(a, b):
    """与 metrics.py / check_pair.py 一致：份额分布的 L1 距离。"""
    return sum(abs(x - y) for x, y in zip(a, b))


def _cv(load):
    n = len(load)
    if n == 0:
        return 0.0
    m = sum(load) / n
    if m <= 0:
        return 0.0
    # 分母用 n-1（无偏），与 metrics.py / check_pair.py / 训练代码 / summary.csv 一致
    var = sum((x - m) ** 2 for x in load) / max(1, n - 1)
    return (var ** 0.5) / m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    with open(args.data, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    os.makedirs(args.out, exist_ok=True)
    plt = _setup()
    made = []
    for fn, key in ((fig_rotation, "fig3_rotation"),
                    (fig_noise, "fig1_noise"),
                    (fig_pair, "fig2_pair")):
        if key in data:
            made.append(fn(data, args.out, plt))
    for p in made:
        print("wrote", p)


if __name__ == "__main__":
    main()
