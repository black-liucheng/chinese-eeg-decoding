# -*- coding: utf-8 -*-
"""
数据分段与预处理
================

从连续记录切出试次，以及对试次做滤波、重参考、去伪迹、基线校正、标准化。

处理顺序（与真实脑电分析的习惯一致）::

    连续记录 → 带通滤波 → 陷波(50Hz) → 切分试次 → 重参考
             → 基线校正 → 伪迹剔除 → 逐通道标准化

**先滤波再切分**是有意为之：如果在切分之后滤波，滤波器在每个试次边界都会
产生边缘效应，而边界正好落在我们关心的 onset 附近。

所有函数都是纯函数（不改输入数组），方便在界面上反复调参预览。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt

from ..config import Config
from .base import EEGDataset

# ---------------------------------------------------------------------------
#  滤波
# ---------------------------------------------------------------------------


def bandpass_filter(X: np.ndarray, sfreq: float, l_freq: float, h_freq: float,
                    order: int = 4) -> np.ndarray:
    """零相位带通滤波（Butterworth，SOS + 前后向滤波）。

    ``X`` 形状 ``(..., n_times)``，在最后一个轴上滤波。
    """
    nyq = sfreq / 2.0
    lo, hi = max(l_freq, 1e-3), min(h_freq, nyq * 0.99)
    if lo >= hi:
        return np.asarray(X, dtype=np.float32).copy()
    sos = butter(order, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, X, axis=-1).astype(np.float32)


def notch_filter(X: np.ndarray, sfreq: float, freq: float, width: float = 2.0
                 ) -> np.ndarray:
    """工频陷波。``freq<=0`` 或超过奈奎斯特频率时原样返回。"""
    nyq = sfreq / 2.0
    if freq <= 0 or freq >= nyq:
        return np.asarray(X, dtype=np.float32).copy()
    b, a = iirnotch(w0=freq / nyq, Q=max(freq / max(width, 1e-6), 0.1))
    from scipy.signal import filtfilt
    return filtfilt(b, a, X, axis=-1).astype(np.float32)


def rereference(X: np.ndarray, mode: str = "car") -> np.ndarray:
    """重参考。``car`` = 共同平均参考（减去全导联均值）。

    参考电极的选择会显著影响 ERP 的幅度与拓扑分布，是分析里必须写明的参数。
    """
    X = np.asarray(X, dtype=np.float32)
    if mode == "none":
        return X.copy()
    if mode == "car":
        return (X - X.mean(axis=-2, keepdims=True)).astype(np.float32)
    raise ValueError(f"未知的重参考方式 {mode!r}，可选 'car' / 'none'")


# ---------------------------------------------------------------------------
#  分段
# ---------------------------------------------------------------------------


def make_epochs(
    data: np.ndarray,
    sfreq: float,
    events: Sequence[int],
    tmin: float,
    tmax: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """从连续记录 ``(n_ch, n_t)`` 按事件采样点切出试次。

    返回 ``(epochs, valid_event_idx)``。落在记录边界之外、无法完整切出的
    事件会被跳过（真实数据里首尾事件经常缺一半，不能硬切出短一截的试次
    然后在 reshape 时炸掉）。

    ``epochs`` 形状 ``(n_valid, n_ch, n_times)``，``tmin`` 对应第一列。
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"data 应为 (n_ch, n_t)，实际 shape={data.shape}")

    n_pre = int(round(-tmin * sfreq))
    n_post = int(round(tmax * sfreq))
    n_times = n_pre + n_post
    n_total = data.shape[1]

    epochs, valid = [], []
    for i, onset in enumerate(events):
        onset = int(onset)
        start, stop = onset - n_pre, onset + n_post
        if start < 0 or stop > n_total:
            continue  # 边界不完整，丢弃
        epochs.append(data[:, start:stop])
        valid.append(i)

    if not epochs:
        return (np.empty((0, data.shape[0], n_times), dtype=np.float32),
                np.array([], dtype=np.int64))
    return np.stack(epochs).astype(np.float32), np.array(valid, dtype=np.int64)


# ---------------------------------------------------------------------------
#  试次级处理
# ---------------------------------------------------------------------------


def baseline_correct(X: np.ndarray, times: np.ndarray,
                     baseline: Optional[Sequence[float]]) -> np.ndarray:
    """基线校正：减去基线窗内的时间均值。

    基线窗（通常取 onset 前的注视点阶段）反映的是该试次的"静息水平"，
    减掉它可以消除缓慢漂移与个体间幅度差异。
    """
    X = np.asarray(X, dtype=np.float32)
    if baseline is None:
        return X.copy()
    b0, b1 = baseline
    mask = (times >= b0) & (times <= b1)
    if not mask.any():
        return X.copy()
    return (X - X[..., mask].mean(axis=-1, keepdims=True)).astype(np.float32)


def detect_bad_trials(X: np.ndarray, reject_uv: float) -> np.ndarray:
    """峰峰值超阈值的试次判为伪迹。返回布尔掩码，``True`` 表示保留。

    ``reject_uv<=0`` 表示不剔除，全部保留。
    """
    if reject_uv <= 0:
        return np.ones(X.shape[0], dtype=bool)
    ptp = np.ptp(X, axis=-1)          # (n_trials, n_ch)
    return ~(ptp.max(axis=-1) > reject_uv)


def standardize(X: np.ndarray, mode: str = "zscore") -> np.ndarray:
    """逐通道标准化。

    ‼ 注意：这里的均值/标准差是**跨试次**统计的 —— 包括后面会被划进测试折的
    试次。严格的交叉验证应该只用训练折的统计量去变换测试折，也就是
    ``fit_standardizer`` / ``apply_standardizer`` 两步走。

    **本函数目前就是流水线的默认做法**（``preprocess.normalize: zscore``），
    并不是只用于预览 —— 这一点以前这里写反了，声称流水线走的是两步走那条路，
    而实际上两个调用点调的都是本函数，``fit_standardizer`` 只有测试在调。
    文档必须描述代码真正在做的事，否则下一个读代码的人会据此做出错误判断。

    影响有多大：实测过（2026-10-07）。在默认数据、5 折交叉验证下，SVM 用
    全量统计量得 58.9%、完全不做跨试次标准化得 58.8%（差 0.1 个百分点）；
    CNN 在 4 受试者 × 30 试次/类下分别是 65.3% 与 67.8% —— 方向是**偏低**
    而不是虚高。也就是说这处不严谨没有抬高任何已报出的结论，因此没有为此
    重跑全部实验。要彻底做对，就把统计量下沉到模型层、在 ``fit`` 里用训练折
    估计（见技术文档"已知局限"一节）。
    """
    X = np.asarray(X, dtype=np.float32)
    if mode == "none":
        return X.copy()
    if mode == "zscore":
        mu = X.mean(axis=(0, 2), keepdims=True)
        sd = X.std(axis=(0, 2), keepdims=True) + 1e-6
        return ((X - mu) / sd).astype(np.float32)
    raise ValueError(f"未知的标准化方式 {mode!r}，可选 'zscore' / 'none'")


def fit_standardizer(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """在训练集上估计逐通道均值/标准差。"""
    mu = X.mean(axis=(0, 2), keepdims=True)
    sd = X.std(axis=(0, 2), keepdims=True) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


def apply_standardizer(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return ((np.asarray(X, dtype=np.float32) - mu) / sd).astype(np.float32)


# ---------------------------------------------------------------------------
#  整条流水线
# ---------------------------------------------------------------------------


@dataclass
class PreprocessReport:
    """预处理留下了什么痕迹 —— 报告和界面都要显示，别让剔除静悄悄地发生。"""

    n_trials_in: int = 0
    n_trials_out: int = 0
    n_rejected: int = 0
    reject_ratio: float = 0.0
    per_class_kept: Dict[str, int] = field(default_factory=dict)
    steps: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"输入试次：{self.n_trials_in}",
            f"伪迹剔除：{self.n_rejected}（{self.reject_ratio:.1%}）",
            f"保留试次：{self.n_trials_out}",
            "各类别保留：" + "，".join(f"{k}={v}" for k, v in self.per_class_kept.items()),
            "处理步骤：" + " → ".join(self.steps),
        ]
        return "\n".join(lines)


def preprocess_dataset(ds: EEGDataset, cfg: Config) -> Tuple[EEGDataset, PreprocessReport]:
    """对已分段的 ``EEGDataset`` 跑完整预处理。

    输入应已经切好试次（仿真数据自带；真实数据先经 ``make_epochs``）。
    返回新的数据集（不修改原对象）与一份处理报告。
    """
    p = cfg.preprocess
    times = ds.times
    steps: List[str] = []

    X = ds.X.astype(np.float32)

    if p.l_freq > 0 and p.h_freq > 0:
        X = bandpass_filter(X, ds.sfreq, p.l_freq, p.h_freq)
        steps.append(f"带通 {p.l_freq:g}-{p.h_freq:g} Hz")

    if p.notch_freq and p.notch_freq > 0:
        X = notch_filter(X, ds.sfreq, p.notch_freq, p.notch_width)
        steps.append(f"陷波 {p.notch_freq:g} Hz")

    # 按配置裁剪时间窗。**必须放在滤波之后**：先滤完整段再裁，边界处的
    # 滤波上下文才够长；反过来先裁再滤，新的首尾会因为没有上下文而失真。
    # 界面上改"窗口起点/终点"能不能生效，全靠这一步。
    if p.tmin is not None and p.tmax is not None:
        win = (times >= p.tmin - 1e-9) & (times <= p.tmax + 1e-9)
        if win.any() and int(win.sum()) != X.shape[-1]:
            X = X[:, :, win]
            times = times[win]
            steps.append(f"裁剪窗口 {p.tmin:g}~{p.tmax:g} s")

    if p.reref != "none":
        X = rereference(X, p.reref)
        steps.append("共同平均参考" if p.reref == "car" else p.reref)

    if p.baseline:
        X = baseline_correct(X, times, p.baseline)
        steps.append(f"基线校正 {p.baseline[0]:g}~{p.baseline[1]:g} s")

    keep = detect_bad_trials(X, p.reject_uv)
    n_rejected = int((~keep).sum())
    steps.append(f"伪迹剔除 >{p.reject_uv:g} µV" if p.reject_uv > 0 else "未剔除伪迹")

    X_kept, y_kept = X[keep], ds.y[keep]
    if X_kept.shape[0] == 0:
        raise ValueError(
            f"所有试次都被伪迹剔除规则排除了（阈值 {p.reject_uv:g} µV）。"
            "请放宽 preprocess.reject_uv，或检查数据本身。"
        )

    if p.normalize != "none":
        X_kept = standardize(X_kept, p.normalize)
        steps.append("逐通道 z-score" if p.normalize == "zscore" else p.normalize)

    report = PreprocessReport(
        n_trials_in=ds.n_trials,
        n_trials_out=int(X_kept.shape[0]),
        n_rejected=n_rejected,
        reject_ratio=n_rejected / max(ds.n_trials, 1),
        per_class_kept={
            name: int((y_kept == i).sum()) for i, name in enumerate(ds.class_names)
        },
        steps=steps,
    )

    out = ds.subset(np.where(keep)[0])
    out.X = X_kept
    # 裁过窗口的话，tmin 要跟着改 —— 否则 times 属性会算错，
    # 图上时间轴和波形就对不上了。
    out.tmin = float(times[0])
    out.meta = dict(ds.meta)
    out.meta["preprocess"] = {
        "steps": steps,
        "n_rejected": n_rejected,
        "reject_uv": p.reject_uv,
    }
    return out, report


def preprocess_raw(
    data: np.ndarray,
    sfreq: float,
    events: Sequence[int],
    event_codes: Sequence[int],
    ch_names: Sequence[str],
    class_names: Sequence[str],
    cfg: Config,
    is_synthetic: bool = True,
    meta: Optional[dict] = None,
) -> Tuple[EEGDataset, PreprocessReport]:
    """从**连续记录**一路走到预处理好的 ``EEGDataset``。

    这是真实数据进来时走的路径：连续记录 → 滤波 → 切分 → 重参考 → 基线 →
    剔除 → 标准化。仿真数据也可以走这条路，用来验证适配器逻辑。
    """
    p = cfg.preprocess
    steps: List[str] = []

    data = np.asarray(data, dtype=np.float32)
    if p.l_freq > 0 and p.h_freq > 0:
        data = bandpass_filter(data, sfreq, p.l_freq, p.h_freq)
        steps.append(f"带通 {p.l_freq:g}-{p.h_freq:g} Hz")
    if p.notch_freq and p.notch_freq > 0:
        data = notch_filter(data, sfreq, p.notch_freq, p.notch_width)
        steps.append(f"陷波 {p.notch_freq:g} Hz")
    if p.reref != "none":
        data = rereference(data, p.reref)
        steps.append("共同平均参考")

    epochs, valid_idx = make_epochs(data, sfreq, events, p.tmin, p.tmax)
    if epochs.shape[0] == 0:
        raise ValueError(
            "没有任何事件能完整切出试次。请检查事件的采样点位置与 "
            "preprocess.tmin/tmax 是否超出记录范围。"
        )
    steps.append(f"切分 {p.tmin:g}~{p.tmax:g} s")

    codes = np.asarray(event_codes, dtype=np.int64)[valid_idx]
    times = p.tmin + np.arange(epochs.shape[-1]) / sfreq

    if p.baseline:
        epochs = baseline_correct(epochs, times, p.baseline)
        steps.append(f"基线校正 {p.baseline[0]:g}~{p.baseline[1]:g} s")

    keep = detect_bad_trials(epochs, p.reject_uv)
    n_rejected = int((~keep).sum())
    steps.append(f"伪迹剔除 >{p.reject_uv:g} µV" if p.reject_uv > 0 else "未剔除伪迹")
    if keep.sum() == 0:
        raise ValueError(
            f"所有试次都被伪迹剔除规则排除了（阈值 {p.reject_uv:g} µV）。请放宽阈值。"
        )
    epochs, codes = epochs[keep], codes[keep]

    if p.normalize != "none":
        epochs = standardize(epochs, p.normalize)
        steps.append("逐通道 z-score")

    ds = EEGDataset(
        X=epochs,
        y=codes,
        class_names=list(class_names),
        sfreq=float(sfreq),
        tmin=float(p.tmin),
        ch_names=list(ch_names),
        is_synthetic=is_synthetic,
        meta=dict(meta or {}),
    )
    report = PreprocessReport(
        n_trials_in=int(len(events)),
        n_trials_out=ds.n_trials,
        n_rejected=n_rejected,
        reject_ratio=n_rejected / max(len(events), 1),
        per_class_kept=ds.class_counts(),
        steps=steps,
    )
    return ds, report
