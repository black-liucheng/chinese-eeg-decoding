# -*- coding: utf-8 -*-
"""生成答辩材料用的图：系统架构图、实验流程图、信噪比扫描曲线。

跑法（项目根目录下）：
    python scripts/make_slides.py

输出到 docs/答辩材料/ 与 data/figures/。
所有图都用项目自己的中文字体设置（plots.setup_chinese_font），
避免答辩现场 PPT 里冒出方块字。
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.patches as mpatches                      # noqa: E402
import matplotlib.pyplot as plt                            # noqa: E402
import numpy as np                                         # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from cneeg import plots                                    # noqa: E402

OUT = ROOT / "docs" / "答辩材料"
FIG = ROOT / "data" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
plots.setup_chinese_font()

# 配色：蓝=数据/传统方法，橙=深度学习，绿=评估，灰=配置
C_DATA, C_MODEL, C_EVAL, C_CFG = "#2b5d8a", "#d1642a", "#3f8f5b", "#6b6b6b"


def _box(ax, x, y, w, h, text, color, fontsize=10, alpha=0.13):
    """圆角框 + 居中文字。"""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.6, edgecolor=color, facecolor=color, alpha=alpha,
        mutation_aspect=0.55,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=color, linespacing=1.55)


def _arrow(ax, p1, p2, color="#555555", style="-|>"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=14,
                                 linewidth=1.5, color=color,
                                 shrinkA=2, shrinkB=2))


# ---------------------------------------------------------------------------
#  1. 系统分层架构图
# ---------------------------------------------------------------------------


def architecture() -> Path:
    fig, ax = plt.subplots(figsize=(11, 8.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # 呈现层
    _box(ax, 0.04, 0.845, 0.44, 0.115, "PyQt5 图形界面\n① 数据 ② 预处理 ③ 训练\n④ 评估 ⑤ 推理",
         C_CFG, 10)
    _box(ax, 0.52, 0.845, 0.44, 0.115, "命令行 CLI\n生成数据 / 训练 / 对比 / 扫描\n报告 / 推理 / 打开界面",
         C_CFG, 10)

    # 应用层
    _box(ax, 0.04, 0.665, 0.44, 0.105,
         "train.py\n交叉验证 · 置换检验 · 评估指标", C_EVAL, 10)
    _box(ax, 0.52, 0.665, 0.20, 0.105, "report.py\nHTML 报告", C_EVAL, 10)
    _box(ax, 0.76, 0.665, 0.20, 0.105, "plots.py\n全部图表", C_EVAL, 10)

    # 模型层
    _box(ax, 0.04, 0.485, 0.44, 0.105,
         "models.py\nSVM 基线（sklearn Pipeline）\nShallowConvNet（PyTorch）", C_MODEL, 10)
    _box(ax, 0.52, 0.485, 0.44, 0.105,
         "features.py\nSTFT 时频特征（2848 维）\n分频带功率（160 维）", C_MODEL, 10)

    # 数据层
    _box(ax, 0.04, 0.295, 0.28, 0.115,
         "synthetic.py\n仿真脑电生成器\n32 导 × 256 Hz", C_DATA, 10)
    _box(ax, 0.35, 0.295, 0.28, 0.115,
         "loaders.py\n真实数据适配器\n.mff/.vhdr/.edf/…", C_DATA, 10)
    _box(ax, 0.66, 0.295, 0.30, 0.115,
         "epoching.py\n滤波 · 重参考 · 基线\n伪迹剔除 · 标准化", C_DATA, 10)

    # 配置层
    _box(ax, 0.04, 0.11, 0.92, 0.095,
         "config.py　配置加载 + 跨字段一致性校验　　configs/default.yaml\n"
         "未知键报错 · 点号路径覆盖（--set model.cnn.n_epochs=200）· 带通上下限写反当场拦截",
         C_CFG, 9.5)

    # 层间箭头（自上而下：呈现层调用应用层，应用层用模型层，模型层吃数据层）
    for x in (0.26, 0.74):
        _arrow(ax, (x, 0.845), (x, 0.775))
        _arrow(ax, (x, 0.665), (x, 0.595))
        _arrow(ax, (x, 0.485), (x, 0.415))
    # 配置层是**横切**的：每一层都从它读参数，不存在"数据层流向配置层"。
    # 画成向下的箭头会误导，用虚线双向标出这层关系。
    ax.plot([0.5, 0.5], [0.295, 0.215], ls=":", lw=1.4, color=C_CFG)
    ax.text(0.515, 0.256, "贯穿各层：各层参数均来自配置", fontsize=9,
            color=C_CFG, va="center")

    # 层标签
    for y, name in [(0.912, "呈现层"), (0.723, "应用层"),
                    (0.543, "模型层"), (0.358, "数据层"), (0.162, "配置层")]:
        ax.text(0.012, y, name, rotation=90, ha="center", va="center",
                fontsize=10, color="#333333", weight="bold")

    ax.set_title("汉语脑电信号分类软件系统 —— 分层架构", fontsize=14, pad=12)
    p = OUT / "01_系统架构图.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
#  2. 实验流程图
# ---------------------------------------------------------------------------


def experiment_flow() -> Path:
    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    steps = [
        ("① 数据生成\n\n仿真脑电\n32 导 256 Hz\n多受试者", C_DATA),
        ("② 预处理\n\n滤波 0.5–40\n陷波 50 Hz\nCAR · 基线\n伪迹剔除", C_DATA),
        ("③ SVM 基线\n\nSTFT 时频特征\n2848 维\n\n传统特征工程", C_MODEL),
        ("④ ShallowConvNet\n\n原始试次数据\n端到端学习\n\n深度学习", C_MODEL),
        ("⑤ 交叉验证\n\n分层 K 折\n留一受试者\n置换检验", C_EVAL),
        ("⑥ 评估\n\n准确率 / 宏F1 / κ\n混淆矩阵\n逐类指标", C_EVAL),
        ("⑦ 产出\n\n图表\nHTML 报告\n可用模型", C_CFG),
    ]
    # 7 个框 + 6 条箭头必须落在 x∈[0,1] 内。上一版算错了总宽，
    # 最后一个框越界压在前一个框上，文字被裁掉一半。
    n = len(steps)
    w = 0.1255
    gap = (1.0 - n * w) / (n - 1)
    y, h = 0.42, 0.42
    for i, (text, color) in enumerate(steps):
        _box(ax, i * (w + gap), y, w, h, text, color, 8.6)
    for i in range(n - 1):
        x1 = i * (w + gap) + w
        x2 = (i + 1) * (w + gap)
        _arrow(ax, (x1 + 0.002, y + h / 2), (x2 - 0.002, y + h / 2))

    ax.text(0.5, 0.30,
            "两条建模路线共用同一套预处理与同一套评估口径 —— 对比才公平",
            ha="center", fontsize=10.5, color="#333333")
    ax.text(0.5, 0.19,
            "置换检验（打乱标签重跑）作为阴性对照，证明流水线没有数据泄漏",
            ha="center", fontsize=10.5, color=C_EVAL)
    ax.text(0.5, 0.08,
            "全部结果均标注“仿真数据”：只证明方法学有效，不代表真实汉语脑电的结论",
            ha="center", fontsize=10.5, color="#b03030", weight="bold")

    ax.set_title("实验流程", fontsize=14, pad=12)
    p = OUT / "02_实验流程图.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
#  3. 信噪比扫描曲线（用真实标定结果画，不编数）
# ---------------------------------------------------------------------------


def snr_sweep() -> Path | None:
    src = ROOT / "scripts" / "snr_sweep.csv"
    if not src.exists():
        print(f"[跳过] 找不到标定数据 {src}，先跑 scripts/calibrate_snr.py")
        return None

    records = []
    with open(src, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            records.append({
                "snr_db": float(row["snr_db"]),
                "svm": float(row["svm_acc"]),
                "cnn": float(row["cnn_acc"]),
                "chance": 0.25,
            })
    fig = plots.plot_snr_sweep(records, title="信噪比扫描 —— 多难的数据下模型才有效")
    p = FIG / "信噪比扫描.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
#  4. 数据量对照图 —— 两张主表放在一起，说明"数据量决定胜负"
# ---------------------------------------------------------------------------


def data_volume() -> Path:
    """同一个信噪比（+4 dB）下，数据量翻倍带来的差别。

    这张图是为答辩幻灯片 10 准备的：单独看每一张结果表都不足以说明问题，
    并排放才能看出"信噪比一样、只差数据量，结论却反过来"。
    """
    groups = [
        ("640 试次\n（留出法 30%）", 50.4, 53.9),
        ("1302 试次\n（5 折交叉验证）", 58.9, 66.8),
    ]
    x = np.arange(len(groups))
    w = 0.34

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    svm = [g[1] for g in groups]
    cnn = [g[2] for g in groups]
    b1 = ax.bar(x - w / 2, svm, w, label="SVM（STFT + 分频带功率）",
                color="#2b5d8a", alpha=0.85)
    b2 = ax.bar(x + w / 2, cnn, w, label="CNN（ShallowConvNet）",
                color="#d1642a", alpha=0.85)

    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2,
                    f"{b.get_height():.1f}%", ha="center", fontsize=10.5,
                    color="#222222")

    ax.axhline(25, ls="--", lw=1.4, color="#888888")
    ax.text(len(groups) - 0.52, 26.5, "随机水平 25%", fontsize=9.5,
            color="#666666")

    # 把"打平"与"CNN 胜出"直接标在图上，避免观众自己算差值。
    # 高度要留在**柱顶数值标签之上**：柱子最高 66.8，标签再高 1.2，
    # 两行注释各占约 7 个数据单位，所以 ylim 给到 88、注释中心放在 74 以上，
    # 否则注释会和"66.8%"叠在一起（第一版就是这么糊的）。
    ax.annotate("两模型打平\n（差距在标准误内）", xy=(0, 60),
                ha="center", fontsize=10, color="#444444")
    ax.annotate("CNN 五折全胜\n平均 +7.9 个百分点", xy=(1, 76),
                ha="center", fontsize=10.5, color="#b03030", weight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups], fontsize=10.5)
    ax.set_ylabel("四声调分类准确率")
    ax.set_ylim(0, 88)
    ax.set_title("信噪比同为 +4 dB，只改数据量", fontsize=13.5, pad=12)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    fig.text(0.5, -0.02,
             "两组的评估协议不同（留出法 vs 交叉验证），"
             "所以这是两个观测点的对照，不是受控实验",
             ha="center", fontsize=9, color="#b03030")
    p = OUT / "03_数据量对照.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------


if __name__ == "__main__":
    for fn in (architecture, experiment_flow, snr_sweep, data_volume):
        out = fn()
        if out:
            print(f"[完成] {out}")
