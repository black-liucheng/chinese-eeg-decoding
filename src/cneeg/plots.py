# -*- coding: utf-8 -*-
"""
可视化
======

所有图都在这里生成，界面（PyQt5）和 HTML 报告共用同一批函数 ——
保证"界面上看到的"和"报告里写进去的"是同一种图，不会出现两套口径。

.. important::
   **中文字体**：matplotlib 默认字体不含汉字，不设置的话图里所有中文都会
   变成方框（豆腐块）。:func:`setup_chinese_font` 在导入时自动执行，
   并按 Windows 上实际存在的字体依次回退。

每个函数都返回 ``matplotlib.figure.Figure``，不调用 ``plt.show()``，
这样既能内嵌到 Qt 画布，也能直接存成图片。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")            # 无界面后端：服务器/批处理也能出图，Qt 另设后端
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .data.base import CH_POS_2D

# ---------------------------------------------------------------------------
#  中文字体
# ---------------------------------------------------------------------------

#: 按优先级排列的候选中文字体（Windows 常见）
_CJK_FONTS = [
    "Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC",
    "Source Han Sans SC", "WenQuanYi Micro Hei", "Arial Unicode MS",
]


def setup_chinese_font() -> str:
    """配置 matplotlib 中文字体，返回实际选中的字体名。

    找不到任何中文字体时会退化为默认字体并给出警告 —— 此时图里的中文
    会变成方框，但不影响程序继续运行。
    """
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((f for f in _CJK_FONTS if f in available), None)

    plt.rcParams["font.sans-serif"] = ([chosen] if chosen else []) + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False   # 负号用 ASCII，避免中文字体里缺字形
    plt.rcParams["figure.autolayout"] = False
    plt.rcParams["savefig.bbox"] = "tight"

    if chosen is None:
        import warnings
        warnings.warn(
            "未找到可用的中文字体，图表中的中文可能显示为方框。"
            f"请安装以下任一字体：{_CJK_FONTS}",
            RuntimeWarning, stacklevel=2,
        )
    return chosen or "DejaVu Sans"


CJK_FONT = setup_chinese_font()

#: 仿真数据的统一水印 —— 任何一张图都不允许让人误以为是真实实验结果
SYNTHETIC_NOTE = "仿真数据（仅用于验证流水线，非真实实验结果）"


def _watermark(fig: Figure, text: str = SYNTHETIC_NOTE) -> None:
    """在图的右下角打水印，防止仿真结果被误当成真实结论。"""
    fig.text(0.995, 0.005, text, ha="right", va="bottom", fontsize=7,
             color="#b03030", alpha=0.75, style="italic")


def _new_fig(figsize=(9, 5), nrows=1, ncols=1, **kw) -> Tuple[Figure, np.ndarray]:
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kw)
    return fig, np.atleast_1d(axes)


def save(fig: Figure, path: str | Path, dpi: int = 150) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


# ---------------------------------------------------------------------------
#  信号总览
# ---------------------------------------------------------------------------


def plot_waveform(ds, n_channels: int = 8, seconds: Optional[float] = None,
                  title: str = "脑电波形总览") -> Figure:
    """若干导联的连续波形，按幅度错开排列（蝴蝶图）。"""
    n_show = min(n_channels, ds.n_channels)
    # 先定下要画几导联，再建子图 —— 不能把 n_channels 透传给 plt.subplots，
    # 它只认 nrows/ncols，多余的关键字会被 Figure.set() 拒掉。
    fig, axes = _new_fig(figsize=(11, 6), nrows=n_show, sharex=True)
    n_t = ds.n_times if seconds is None else min(ds.n_times, int(seconds * ds.sfreq))
    t = ds.times[:n_t]

    # 取各类别均衡的一段，避免只看到某一类
    order = []
    per_class = max(1, 20 // ds.n_classes)
    for c in range(ds.n_classes):
        order += list(np.where(ds.y == c)[0][:per_class])
    seg = ds.X[order[:20], :n_show, :n_t]

    offset = np.abs(seg).max() * 1.1 + 1e-6
    for ch in range(n_show):
        y = seg[:, ch, :].mean(0) + ch * offset
        axes[ch].plot(t, y, lw=0.8, color="#2b5d8a")
        axes[ch].set_ylabel(ds.ch_names[ch], rotation=0, ha="right", va="center",
                            fontsize=8)
        axes[ch].set_yticks([])
        axes[ch].spines[["top", "right", "left", "bottom"]].set_visible(False)

    axes[-1].set_xlabel("时间（秒，相对事件起点）")
    fig.suptitle(f"{title}　（{n_show} 导联，各类别平均）", fontsize=12)
    if ds.is_synthetic:
        _watermark(fig)
    return fig


def plot_erp(ds, channels: Optional[Sequence[str]] = None, title: str = "事件相关电位（ERP）") -> Figure:
    """各类别在若干导联上的平均 ERP。

    ERP 是相位锁定成分的平均，能直观看出各类别的波形差异 —— 如果这里
    看不出差异，模型大概率也学不到东西，应该先回来查数据而不是调模型。
    """
    channels = list(channels or ["Fz", "Cz", "Pz", "C3", "C4"])
    channels = [c for c in channels if c in ds.ch_names][:4] or ds.ch_names[:4]

    fig, axes = _new_fig(figsize=(11, 3.2 * len(channels)), nrows=len(channels))
    colors = plt.cm.tab10(np.linspace(0, 1, max(ds.n_classes, 3)))
    t = ds.times

    for ax, ch in zip(axes, channels):
        ci = ds.ch_names.index(ch)
        for k, cname in enumerate(ds.class_names):
            sel = ds.X[ds.y == k, ci, :]
            if sel.size == 0:
                continue
            m = sel.mean(0)
            sem = sel.std(0) / max(np.sqrt(len(sel)), 1)
            ax.plot(t, m, lw=1.6, color=colors[k], label=cname)
            ax.fill_between(t, m - sem, m + sem, color=colors[k], alpha=0.18,
                            linewidth=0)
        ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.6)
        ax.axhline(0, color="k", lw=0.5, alpha=0.3)
        ax.set_title(f"{ch} 导联", fontsize=10)
        ax.set_ylabel("幅度（µV）")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].legend(ncol=min(ds.n_classes, 4), fontsize=9, frameon=False)
    axes[-1].set_xlabel("时间（秒，0 = 刺激出现）")
    fig.suptitle(title, fontsize=12)
    if ds.is_synthetic:
        _watermark(fig)
    return fig


def plot_tfr(ds, channel: str = "Cz", title: str = "时频图（各类别平均）") -> Figure:
    """各类别的平均时频功率 —— 看节律成分（α/β）随时间的变化。"""
    from scipy.signal import stft

    ci = ds.ch_names.index(channel) if channel in ds.ch_names else 0
    fig, axes = _new_fig(figsize=(4.2 * ds.n_classes, 3.6), ncols=ds.n_classes)
    vmin, vmax = None, None

    panels = []
    for k in range(ds.n_classes):
        X = ds.X[ds.y == k, ci:ci + 1, :]
        f, t, Z = stft(X, fs=ds.sfreq, nperseg=min(64, ds.n_times),
                       noverlap=min(48, ds.n_times - 1), axis=-1)
        P = 10 * np.log10(np.abs(Z[:, 0]) ** 2 + 1e-12).mean(0)   # (freq, time)
        keep = (f >= 1) & (f <= 45)
        panels.append((f[keep], t + ds.tmin, P[keep]))

    allv = np.concatenate([p[2].ravel() for p in panels])
    vmin, vmax = np.percentile(allv, 5), np.percentile(allv, 95)

    for ax, (f, t, P), cname in zip(axes, panels, ds.class_names):
        im = ax.pcolormesh(t, f, P, shading="auto", cmap="turbo",
                           vmin=vmin, vmax=vmax)
        ax.axvline(0, color="w", lw=1.0, ls="--", alpha=0.8)
        ax.set_title(str(cname), fontsize=10)
        ax.set_xlabel("时间（秒）")
        if ax is axes[0]:
            ax.set_ylabel("频率（Hz）")
        fig.colorbar(im, ax=ax, label="功率（dB）")

    fig.suptitle(f"{title}　{channel} 导联", fontsize=12)
    if ds.is_synthetic:
        _watermark(fig)
    return fig


def plot_topomap(values: np.ndarray, ch_names: Sequence[str],
                 title: str = "头皮拓扑图", cmap: str = "RdBu_r",
                 vlim: Optional[Tuple[float, float]] = None) -> Figure:
    """头皮拓扑图（插值 + 头部轮廓）。

    自己用 ``griddata`` 插值而不用 mne 的 ``plot_topomap``：这样坐标与
    ``CH_POS_2D`` 完全一致，不依赖外部 montage 文件，离线环境也能出图。
    """
    from scipy.interpolate import griddata

    pos = np.array([CH_POS_2D.get(c, (0.0, 0.0)) for c in ch_names], dtype=float)
    values = np.asarray(values, dtype=float).ravel()

    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    grid_x, grid_y = np.mgrid[-1.15:1.15:200j, -1.15:1.15:200j]

    interp = griddata(pos, values, (grid_x, grid_y), method="cubic")
    if np.isnan(interp).any():                       # 边缘外插失败时用最近邻补
        nn = griddata(pos, values, (grid_x, grid_y), method="nearest")
        interp = np.where(np.isnan(interp), nn, interp)

    r = np.sqrt(grid_x ** 2 + grid_y ** 2)
    interp = np.ma.masked_where(r > 1.0, interp)

    if vlim is None:
        v = np.abs(values).max() + 1e-12
        vlim = (-v, v)

    im = ax.imshow(interp, extent=[-1.15, 1.15, -1.15, 1.15], origin="lower",
                   cmap=cmap, vmin=vlim[0], vmax=vlim[1])
    ax.contour(grid_x, grid_y, r, levels=[1.0], colors="k", linewidths=1.2)

    # 鼻子与耳朵：让方向一眼可辨（上方为额区）
    ax.plot([-0.10, 0.0, 0.10], [1.0, 1.12, 1.0], color="k", lw=1.2)
    ax.plot([-1.0, -1.06, -1.0], [-0.15, 0.0, 0.15], color="k", lw=1.0)
    ax.plot([1.0, 1.06, 1.0], [-0.15, 0.0, 0.15], color="k", lw=1.0)

    for (x, y), name in zip(pos, ch_names):
        ax.plot(x, y, "k.", ms=2.5)
        ax.text(x, y + 0.06, name, fontsize=6, ha="center", va="bottom", alpha=0.8)

    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.25)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.75)
    return fig


def channel_discriminability(ds, tmin: float = 0.25,
                            tmax: float = 0.60) -> np.ndarray:
    """逐电极计算"类别可分性"（单因素 F 值）。

    对每个电极：把各试次在 ``[tmin, tmax]`` 窗口内的平均幅度按类别分组，
    算**组间方差 / 组内方差**。F 值大 = 该电极上各类别的响应差异明显、
    且试次间稳定 —— 也就是"这个电极有信息"。

    之所以不做成"平均幅度拓扑图"：幅度大只说明信号强，不代表**类别之间**
    有差别（比如全类共有的 P1 就会在大幅度电极上亮起来，但它对分类毫无用处）。
    """
    t = ds.times
    win = (t >= tmin) & (t <= tmax)
    if not win.any():
        win = np.ones_like(t, dtype=bool)
    grand = np.asarray(ds.X)[:, :, win].mean(axis=2)      # (试次, 电极)

    vals = np.zeros(ds.n_channels, dtype=float)
    for c in range(ds.n_channels):
        groups = [grand[ds.y == k, c] for k in range(ds.n_classes)]
        groups = [g for g in groups if g.size > 1]
        if len(groups) < 2:
            continue
        between = np.var([g.mean() for g in groups]) * len(groups)
        within = float(np.mean([g.var() for g in groups]))
        vals[c] = between / within if within > 1e-12 else 0.0
    return vals


def plot_discriminability(ds, tmin: float = 0.25, tmax: float = 0.60) -> Figure:
    """类别可分性的头皮分布 —— 报告里用来说明"信息在哪些电极上"。"""
    vals = channel_discriminability(ds, tmin, tmax)
    return plot_topomap(
        vals, ds.ch_names, cmap="viridis", vlim=(0.0, float(vals.max()) or 1.0),
        title=f"各电极的类别可分性（F 值，{tmin:.2f}–{tmax:.2f} s）")


# ---------------------------------------------------------------------------
#  预处理对比
# ---------------------------------------------------------------------------


def plot_preprocess_compare(raw, clean, channel: str = "Fz",
                            title: str = "预处理前后对比") -> Figure:
    """同一导联在预处理前后的波形对比 —— 界面上调参时最直观的反馈。"""
    ci = raw.ch_names.index(channel) if channel in raw.ch_names else 0
    fig, axes = _new_fig(figsize=(11, 5), nrows=2, sharex=True)

    for ax, data, name in ((axes[0], raw, "预处理前（原始）"),
                           (axes[1], clean, "预处理后")):
        n_show = min(12, data.n_trials)
        idx = np.linspace(0, data.n_trials - 1, n_show).astype(int)
        off = np.abs(data.X[idx, ci, :]).max() * 1.15 + 1e-6
        for j, i in enumerate(idx):
            ax.plot(data.times, data.X[i, ci, :] + j * off, lw=0.7, color="#2b5d8a")
        ax.set_ylabel(f"{name}\n幅度（µV）", fontsize=9)
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)

    pk_raw = np.ptp(raw.X[:, ci, :], axis=-1).mean()
    pk_cln = np.ptp(clean.X[:, ci, :], axis=-1).mean()
    axes[1].set_xlabel("时间（秒）")
    fig.suptitle(f"{title}　{channel} 导联\n"
                 f"平均峰峰值 {pk_raw:.1f} µV → {pk_cln:.1f} µV", fontsize=11)
    if raw.is_synthetic:
        _watermark(fig)
    return fig


# ---------------------------------------------------------------------------
#  模型结果
# ---------------------------------------------------------------------------


def plot_confusion(result, normalize: bool = True, title: Optional[str] = None) -> Figure:
    """混淆矩阵。归一化后看的是"每一类被认成什么的概率"。"""
    cm = result.confusion.astype(float)
    if normalize:
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1e-12)

    names = [str(c) for c in result.class_names]
    fig, ax = plt.subplots(figsize=(1.1 * len(names) + 3.2, 1.0 * len(names) + 2.8))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else cm.max())

    for i in range(len(names)):
        for j in range(len(names)):
            v = cm[i, j]
            ax.text(j, i, f"{v:.2f}" if normalize else f"{int(v)}",
                    ha="center", va="center", fontsize=10,
                    color="white" if v > 0.55 * (1 if normalize else cm.max()) else "black")

    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(title or f"{result.model_name.upper()} 混淆矩阵"
                 f"（准确率 {result.accuracy:.1%}）", fontsize=11)
    fig.colorbar(im, ax=ax, label="比例" if normalize else "试次数")
    if result.is_synthetic:
        _watermark(fig)
    return fig


def plot_training_curve(history: Optional[Dict[str, List[float]]],
                        chance_level: Optional[float] = None,
                        title: str = "训练曲线") -> Figure:
    """损失、验证宏 F1、验证准确率随轮次的变化 —— CNN 训练时界面上实时刷新。

    中间那格画的是**宏 F1**，不是准确率：早停和"最佳轮次"都是按它选的，
    只画准确率会让人看不懂模型为什么停在那儿。
    """
    fig, axes = _new_fig(figsize=(13, 3.6), ncols=3)

    if not history or not history.get("epoch"):
        for ax in axes:
            ax.text(0.5, 0.5, "（该模型无训练曲线）", ha="center", va="center",
                    fontsize=12, color="gray", transform=ax.transAxes)
            ax.axis("off")
        return fig

    ep = history["epoch"]
    axes[0].plot(ep, history["train_loss"], label="训练损失", color="#2b5d8a", lw=1.6)
    vl = history.get("val_loss") or []
    if vl and not np.all(np.isnan(vl)):
        axes[0].plot(ep[:len(vl)], vl, label="验证损失", color="#d1642a", lw=1.6)
    axes[0].set_xlabel("轮次"); axes[0].set_ylabel("交叉熵损失")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False)

    vf = history.get("val_f1") or []
    if vf and not np.all(np.isnan(vf)):
        axes[1].plot(ep[:len(vf)], vf, color="#7a4fa3", lw=1.6)
    axes[1].set_xlabel("轮次"); axes[1].set_ylabel("验证宏 F1")
    axes[1].set_ylim(0, 1.02)
    axes[1].spines[["top", "right"]].set_visible(False)

    va = history.get("val_acc") or []
    if va and not np.all(np.isnan(va)):
        axes[2].plot(ep[:len(va)], np.array(va) * 100, color="#3f8f5b", lw=1.6)
        if chance_level:
            axes[2].axhline(chance_level * 100, color="crimson", ls="--", lw=1.2)
            axes[2].text(ep[0], chance_level * 100 + 1.0,
                         f"随机 {chance_level:.0%}", color="crimson", fontsize=8)
    axes[2].set_xlabel("轮次"); axes[2].set_ylabel("验证准确率（%）")
    axes[2].spines[["top", "right"]].set_visible(False)

    fig.suptitle(title, fontsize=12)
    return fig


def plot_model_comparison(results: Dict[str, object],
                          title: str = "模型分类性能对比") -> Figure:
    """SVM vs CNN 的准确率与宏 F1 对比柱状图 —— 立项报告要的核心结论图。"""
    names = list(results.keys())
    accs = [r.accuracy * 100 for r in results.values()]
    errs = [r.accuracy_std * 100 for r in results.values()]
    f1s = [r.macro_f1 * 100 for r in results.values()]
    chance = next(iter(results.values())).chance_level * 100

    fig, axes = _new_fig(figsize=(9, 4.2), ncols=2)
    x = np.arange(len(names))
    colors = ["#2b5d8a", "#d1642a", "#3f8f5b", "#8a5da8"][:len(names)]

    axes[0].bar(x, accs, yerr=errs, capsize=5, color=colors, alpha=0.88)
    axes[0].axhline(chance, color="crimson", ls="--", lw=1.4)
    axes[0].text(len(names) - 0.5, chance + 1.5, f"随机水平 {chance:.1f}%",
                 color="crimson", fontsize=9, ha="right")
    axes[0].set_xticks(x, [n.upper() for n in names])
    axes[0].set_ylabel("准确率（%）")
    axes[0].set_ylim(0, max(max(accs) + max(errs) + 12, chance + 15))
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].bar(x, f1s, color=colors, alpha=0.88)
    axes[1].set_xticks(x, [n.upper() for n in names])
    axes[1].set_ylabel("宏平均 F1（%）")
    axes[1].set_ylim(0, max(max(f1s) + 12, 40))
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.suptitle(title, fontsize=12)
    if next(iter(results.values())).is_synthetic:
        _watermark(fig)
    return fig


def plot_snr_sweep(records: List[Dict[str, float]],
                   title: str = "信噪比扫描") -> Figure:
    """准确率随信噪比的变化 —— 说明"模型在什么难度下才有效"。

    ``records`` 每项形如 ``{"snr_db": -12, "svm": 0.4, "cnn": 0.6,
    "cnn_std": 0.02, "svm_std": 0.03}``。
    """
    if not records:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "（无扫描结果）", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        ax.axis("off")
        return fig

    snr = np.array([r["snr_db"] for r in records])
    order = np.argsort(snr)
    snr = snr[order]

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    palette = {"svm": "#2b5d8a", "cnn": "#d1642a"}
    for name in ("svm", "cnn"):
        if name not in records[0]:
            continue
        vals = np.array([r[name] for r in records])[order] * 100
        stds = np.array([r.get(f"{name}_std", 0.0) for r in records])[order] * 100
        ax.errorbar(snr, vals, yerr=stds, marker="o", capsize=4, lw=1.8,
                    color=palette.get(name, None), label=name.upper())

    ax.set_xlabel("信噪比（dB）　←　越靠左越难")
    ax.set_ylabel("准确率（%）")
    if records and "chance" in records[0]:
        ax.axhline(records[0]["chance"] * 100, color="crimson", ls="--", lw=1.3)
        ax.text(snr.min(), records[0]["chance"] * 100 + 1.5, "随机水平",
                color="crimson", fontsize=9)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, fontsize=12)
    _watermark(fig)
    return fig


def plot_permutation(perm, chance_level: float, title: str = "置换检验") -> Figure:
    """零分布直方图 + 实际观测值竖线 —— 一眼看出结果是否显著。"""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    scores = np.array(perm.scores) * 100 if perm.scores else np.array([0.0])

    ax.hist(scores, bins=min(len(scores), 15), color="#8a9bb0", alpha=0.85,
            edgecolor="white", label="打乱标签后的准确率（零分布）")
    ax.axvline(perm.observed * 100, color="#d1642a", lw=2.4,
               label=f"实际观测 {perm.observed:.1%}")
    ax.axvline(chance_level * 100, color="crimson", ls="--", lw=1.4,
               label=f"随机水平 {chance_level:.1%}")

    ax.set_xlabel("准确率（%）"); ax.set_ylabel("置换次数")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"{title}　p = {perm.p_value:.3f}"
                 f"{'（显著）' if perm.p_value < 0.05 else '（不显著）'}", fontsize=12)
    _watermark(fig)
    return fig
