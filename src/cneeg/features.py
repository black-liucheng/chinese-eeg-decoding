# -*- coding: utf-8 -*-
"""
特征提取
========

给传统机器学习基线（SVM）准备特征。两种模式，对应立项报告里的
"STFT-SVM 基线模型"：

``bandpower``
    整个试次窗口内的分频带平均功率，每导联每频带一个数。
    维度 = 电极数 × 频带数。能捕获事件相关去同步/同步（ERD/ERS），
    但丢掉了时间信息。

``stft``
    短时傅里叶变换，得到"电极 × 频率 × 时间"的时频功率图，
    再按频带在频率轴上聚合、保留时间轴 —— 这样既能反映节律功率，
    也能反映诱发成分出现的时间点。维度大得多，通常需要配合 PCA。

.. note::
   对**相位锁定**的诱发成分（ERP），时域平均波形其实是最有效的特征。
   ShallowCNN 直接吃原始试次数据，能同时利用时间与空间结构，因此在
   低信噪比下通常明显优于这些手工特征 —— 这正是本项目要对比的结论。
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import stft, welch

from .config import Config


#: 已经警告过的"被跳过的频带"组合。特征提取每折都会跑一次，
#: 靠 warnings 模块自带的去重挡不住 —— 它按调用处的 __warningregistry__ 去重，
#: 而这里的调用方是 sklearn Pipeline，栈帧一路在变，结果同一条警告被打了十几遍。
#: 自己记住，保证每种组合整个进程只提醒一次。
_SKIPPED_WARNED: set = set()


def _warn_skipped_bands(skipped, f: np.ndarray, sfreq: float,
                        nperseg: int) -> None:
    key = tuple((b, float(lo), float(hi), int(nperseg), float(sfreq))
                for b, lo, hi in skipped)
    if key in _SKIPPED_WARNED:
        return
    _SKIPPED_WARNED.add(key)

    df = float(f[1] - f[0]) if f.size > 1 else float(sfreq / nperseg)
    detail = "、".join(f"{b}[{lo:g},{hi:g})" for b, lo, hi in skipped)
    warnings.warn(
        f"STFT 频率分辨率为 {df:g} Hz（nperseg={nperseg}，fs={sfreq:g} Hz），"
        f"频带 {detail} 内没有任何频点，已跳过 —— 这些频带不会出现在时频特征里"
        f"（分频带功率部分仍包含它们）。要覆盖它们请增大 features.stft_nperseg。",
        RuntimeWarning, stacklevel=3,
    )


def _as_3d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 3:
        raise ValueError(f"需要 (试次, 电极, 时间) 三维数组，实际 shape={X.shape}")
    return X


def bandpower_features(
    X: np.ndarray,
    sfreq: float,
    bands: Dict[str, Sequence[float]],
    relative: bool = False,
    log: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """分频带平均功率特征。

    返回 ``(特征矩阵 (n_trials, n_ch*n_bands), 特征名列表)``。

    ``relative=True`` 时每个频带的功率除以该导联全频带总功率 —— 消除
    个体间总体幅度差异，是 EEG 分析里的常见做法。
    """
    X = _as_3d(X)
    n_trials, n_ch, n_t = X.shape

    nperseg = min(128, n_t)
    freqs, pxx = welch(X, fs=sfreq, nperseg=nperseg, axis=-1)  # (trials, ch, freq)

    feats, names = [], []
    band_powers = []
    for bname, (lo, hi) in bands.items():
        m = (freqs >= lo) & (freqs < hi)
        if not m.any():
            raise ValueError(
                f"频带 {bname} [{lo}, {hi}] Hz 内没有频率点。"
                f"可用频率范围 0~{sfreq/2:g} Hz，请检查 features.bands 与采样率。"
            )
        bp = pxx[..., m].mean(axis=-1)          # (trials, ch)
        band_powers.append(bp)

    stack = np.stack(band_powers, axis=-1)      # (trials, ch, band)
    if relative:
        total = stack.sum(axis=-1, keepdims=True) + 1e-12
        stack = stack / total

    if log:
        stack = np.log10(stack + 1e-12)

    feats = stack.reshape(n_trials, -1)
    names = [f"{ch}_{b}" for ch in range(n_ch) for b in bands]
    return feats, names


def stft_features(
    X: np.ndarray,
    sfreq: float,
    bands: Dict[str, Sequence[float]],
    nperseg: int = 64,
    noverlap: int = 48,
    fmin: float = 1.0,
    fmax: float = 40.0,
    include_bandpower: bool = True,
    log: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """STFT 时频特征：按频带在频率轴聚合、保留时间轴。

    输出维度 = 电极 × 频带 × 时间帧（``include_bandpower`` 时再拼上整窗频带功率）。
    维度通常上千，建议在模型里配 PCA 降维。
    """
    X = _as_3d(X)
    n_trials, n_ch, n_t = X.shape

    nperseg = min(nperseg, n_t)
    noverlap = min(noverlap, nperseg - 1)

    f, t, Z = stft(X, fs=sfreq, nperseg=nperseg, noverlap=noverlap, axis=-1)
    keep = (f >= fmin) & (f <= fmax)
    f, Z = f[keep], Z[..., keep, :]
    power = np.abs(Z) ** 2                       # (trials, ch, freq, time)

    blocks, names, skipped = [], [], []
    for bname, (lo, hi) in bands.items():
        m = (f >= lo) & (f < hi)
        if not m.any():
            # 静默丢一个频带是很坑的：配置里写着 5 个频带，特征里只有 4 个，
            # 没有任何提示。最典型的就是 delta [1,4) —— nperseg=64 在 256 Hz 下
            # 频率分辨率是 4 Hz，最低可用频点就是 4 Hz，正好落在 [1,4) 之外。
            # 要么把 nperseg 调大，要么把 delta 的下限抬到分辨率上。
            skipped.append((bname, lo, hi))
            continue
        # 频带内在频率轴取平均 -> (trials, ch, time)
        blocks.append(power[..., m, :].mean(axis=-2))
        names += [f"{ch}_{bname}_t{k}" for ch in range(n_ch) for k in range(power.shape[-1])]

    if not blocks:
        raise ValueError("所有频带都没有落在 STFT 的频率范围内，请检查 features.bands")

    if skipped:
        _warn_skipped_bands(skipped, f, sfreq, nperseg)

    feat = np.concatenate(blocks, axis=1)        # (trials, ch*bands, time)
    if log:
        feat = np.log10(feat + 1e-12)
    feats = feat.reshape(n_trials, -1)

    if include_bandpower:
        bp, bp_names = bandpower_features(X, sfreq, bands, relative=False, log=log)
        feats = np.concatenate([feats, bp], axis=1)
        names += bp_names

    return feats, names


def extract_features(X: np.ndarray, sfreq: float, cfg: Config
                     ) -> Tuple[np.ndarray, List[str]]:
    """按配置提取特征 —— 模型统一走这个入口。"""
    bands = {k: tuple(v) for k, v in cfg.features.bands.items()}
    if cfg.features.method == "bandpower":
        return bandpower_features(X, sfreq, bands)
    if cfg.features.method == "stft":
        return stft_features(
            X, sfreq, bands,
            nperseg=cfg.features.stft_nperseg,
            noverlap=cfg.features.stft_noverlap,
            include_bandpower=cfg.features.stft_use_bandpower,
        )
    raise ValueError(
        f"未知的特征方法 {cfg.features.method!r}，可选 'bandpower' / 'stft'"
    )
