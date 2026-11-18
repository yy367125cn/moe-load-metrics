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
                label="%s  (cv=%.4f)" % (k, _cv(d[k])), linewidth=1.6)
    ax.set_xticks(x)
    ax.set_xticklabels(["E%d" % i for i in x])
    ax.set_ylabel("负载份额")
    ax.set_title("图 3  前 400 步：cv +11%，但负载排序彻底改变（L1=0.30）")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(outdir, "fig3_rotation.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def fig_noise(data, outdir, plt):
    """图 1：同 ckpt 内几乎重合，换一个 ckpt 就完全不同。"""
    d = data["fig1_noise"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = list(range(8))
    for i, arr in enumerate(d.get("same_ckpt_runs", [])):
        ax.plot(x, arr, color="#1f77b4", alpha=0.45, linewidth=1.2,
                label="same ckpt (runs)" if i == 0 else None)
    if d.get("baidu_same"):
        for i, arr in enumerate(d["baidu_same"]):
            ax.plot(x, arr, color="#2ca02c", alpha=0.45, linewidth=1.2,
                    label="same ckpt (Baidu)" if i == 0 else None)
    if d.get("other_ckpt"):
        ax.plot(x, d["other_ckpt"], color="#d62728", marker="o", linewidth=1.8,
                label="different ckpt")
    ax.set_xticks(x)
    ax.set_xticklabels(["E%d" % i for i in x])
    ax.set_ylabel("负载份额")
    ax.set_title("图 1  同 ckpt 重复重合（L1~1e-4）；换一个 ckpt 差 0.10")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(outdir, "fig1_noise_scale.png")
    fig.savefig(p, dpi=160)
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
                    label="%s (cv=%.4f)" % (name, _cv(d[name])))
    ax.set_xticks(x)
    ax.set_xticklabels(["E%d" % i for i in x])
    ax.set_ylabel("负载份额")
    ax.set_title("图 2  配对实验：组内两次几乎重合，组间明显分开")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(outdir, "fig2_pair.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def _cv(load):
    n = len(load)
    if n == 0:
        return 0.0
    m = sum(load) / n
    if m <= 0:
        return 0.0
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
