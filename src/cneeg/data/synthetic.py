# -*- coding: utf-8 -*-
"""
仿真脑电生成器
==============

为什么需要它
------------
真实汉语脑电数据尚未采集（实验室 EGI 采集未开始），但软件流水线必须先跑通、
必须先能出结果图和模型对比。本模块生成**生理上说得通**的多通道脑电，用来
验证整条流水线。**它不是真实数据，所有产出都必须标注"仿真数据"。**

信号模型（单试次）
------------------
::

    X[ch, t] = 诱发成分(spatial[ch] × erp[t])        ← 各类别不同，模型要学的东西
             + 1/f 粉红噪声（空间相关）               ← 背景脑活动
             + α(10Hz) / β(20Hz) 节律振荡            ← 类别间有功率差异（ERD/ERS）
             + 50Hz 工频                              ← 市电干扰
             + 伪迹（眨眼 / 肌电 / 通道脱落）

**空间拓扑**用"热点平滑场"构造：每个类别指定 2–3 个皮层热点及其权重，再对
全脑电极做高斯平滑 —— 这样得到的拓扑是连续的，和真实 ERP 的空间分布形态一致。
各类别热点不同，所以空间卷积层（CNN）有东西可学；各类别 ERP 时间波形也不同，
所以时频特征（SVM）也有东西可学。

**信噪比**由 ``snr_db`` 控制：ERP 峰值幅度 = 背景 RMS × 10^(snr_db/20)。
默认 −12 dB（ERP 峰值约为背景 RMS 的 25%，接近真实单试次 ERP 的水平）。
把它调高会更容易分类 —— 用于扫描"多难的数据下 CNN 才跑赢 SVM"。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import Config, PRESETS
from .base import CH_NAMES_32, EEGDataset, channel_positions

# ---------------------------------------------------------------------------
#  预设定义
# ---------------------------------------------------------------------------


@dataclass
class ClassSpec:
    """单个类别的信号规格。

    hotspots : 空间热点 ``[(电极名, 权重), ...]``，权重可正可负（正=正波，负=负波）
    erp      : 时间成分 ``[(潜伏期 s, 宽度 s, 幅度), ...]``
    alpha_erd: 事件相关去同步系数。>1 表示该类别诱发后 α 功率增强，<1 表示减弱
    """

    hotspots: List[Tuple[str, float]]
    erp: List[Tuple[float, float, float]]
    alpha_erd: float = 1.0


# 五个标准 ERP 成分的潜伏期/宽度（秒），各类别主要通过"幅度"区分
_COMPONENT_GRID = [
    (0.10, 0.030),   # P1   — 早期视觉/听觉成分
    (0.17, 0.035),   # N170 — 视觉词形区（左侧颞枕）
    (0.30, 0.060),   # P300 — 注意资源分配
    (0.42, 0.080),   # N400 — 语义整合（中央顶区）
    (0.62, 0.120),   # 晚期正成分 LPC — 加工负荷
]


def _erp_from_amps(amps: Sequence[float], shift: float = 0.0) -> List[Tuple[float, float, float]]:
    """把 5 个成分的幅度列表变成 ``ClassSpec.erp`` 格式，整体平移 ``shift`` 秒。"""
    return [(mu + shift, sd, float(a)) for (mu, sd), a in zip(_COMPONENT_GRID, amps)]


#: 三个预设。类别顺序必须与 ``config.PRESETS`` 一致
PRESET_SPECS: Dict[str, Dict[str, ClassSpec]] = {
    # ---- 汉语声调：/a/ 的四声。声调加工偏右侧颞叶（音高），辅以额区 ----
    "vowel_tone": {
        # 阴平（高平调）：音高稳定，右颞中等激活
        "a1": ClassSpec(
            hotspots=[("T8", 1.0), ("C4", 0.5), ("F4", 0.3)],
            erp=_erp_from_amps([0.5, -0.8, 0.9, -0.4, 0.3]),
            alpha_erd=0.96,
        ),
        # 阳平（上升调）：音高变化大 → 右颞最强，额区参与
        "a2": ClassSpec(
            hotspots=[("T8", 1.3), ("F8", 0.6), ("C4", 0.4)],
            erp=_erp_from_amps([0.7, -1.0, 1.2, -0.6, 0.5]),
            alpha_erd=0.95,
        ),
        # 上声（降升调）：加工最费力 → 双侧额区广泛激活，潜伏期更晚
        "a3": ClassSpec(
            hotspots=[("Fz", 1.2), ("F3", 0.8), ("F4", 0.8)],
            erp=_erp_from_amps([0.4, -0.6, 1.4, -1.0, 1.1], shift=0.03),
            alpha_erd=0.88,
        ),
        # 去声（下降调）：左颞语言区参与
        "a4": ClassSpec(
            hotspots=[("T7", 1.2), ("F7", 0.5), ("C3", 0.4)],
            erp=_erp_from_amps([0.6, -0.9, 0.8, -0.5, 0.4]),
            alpha_erd=1.07,
        ),
    },
    # ---- 汉语词汇：三个单音节词，差异集中在左颞语言区与顶区 ----
    "mandarin_word": {
        "ma": ClassSpec(
            hotspots=[("T7", 1.1), ("C3", 0.6), ("Pz", 0.4)],
            erp=_erp_from_amps([0.6, -1.0, 1.0, -0.9, 0.6]),
            alpha_erd=0.93,
        ),
        "ba": ClassSpec(
            hotspots=[("T7", 0.8), ("F3", 0.7), ("P3", 0.5)],
            erp=_erp_from_amps([0.5, -0.7, 0.8, -0.6, 0.5]),
            alpha_erd=1.06,
        ),
        "da": ClassSpec(
            hotspots=[("T7", 0.6), ("Pz", 1.1), ("P4", 0.6)],
            erp=_erp_from_amps([0.8, -0.6, 1.3, -0.4, 0.8], shift=0.02),
            alpha_erd=0.90,
        ),
    },
    # ---- 英语单词 yes/or/no：对齐现有 word_reading_eeg.py 范式 ----
    "english_word": {
        "yes": ClassSpec(
            hotspots=[("T7", 1.0), ("Fz", 0.7), ("C3", 0.5)],
            erp=_erp_from_amps([0.5, -0.9, 1.1, -0.7, 0.6]),
            alpha_erd=0.94,
        ),
        "or": ClassSpec(
            hotspots=[("T7", 0.7), ("Pz", 0.8), ("Cz", 0.6)],
            erp=_erp_from_amps([0.6, -0.5, 0.7, -0.5, 0.4]),
            alpha_erd=1.10,
        ),
        "no": ClassSpec(
            hotspots=[("Fz", 1.1), ("F4", 0.8), ("T8", 0.4)],
            erp=_erp_from_amps([0.7, -1.1, 1.4, -1.1, 0.9], shift=0.02),
            alpha_erd=0.90,
        ),
    },
}


# ---------------------------------------------------------------------------
#  信号构件
# ---------------------------------------------------------------------------


def _gaussian(t: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((t - mu) / sigma) ** 2)


#: 背景活动（1/f 15µV + α 8µV + β 4µV）的标称 RMS，单位 µV。
#: snr_db 是相对这个值定义的 —— 少了它，ERP 会被缩放到亚微伏级别而完全不可解。
BACKGROUND_RMS_UV: float = float(np.sqrt(15.0 ** 2 + 8.0 ** 2 + 4.0 ** 2))  # ≈ 17.5


def snr_to_scale(snr_db: float) -> float:
    """把 ``snr_db`` 换算成 ERP 模板的幅度系数。

    定义：**ERP 峰值幅度 = 背景 RMS × 10^(snr_db/20)**。
    0 dB 表示 ERP 峰值与背景 RMS 相当（约 17.5 µV，偏乐观）；
    −12 dB 约 4.4 µV（接近真实单试次 ERP）；−20 dB 约 1.7 µV（很难）。
    """
    return BACKGROUND_RMS_UV * 10.0 ** (snr_db / 20.0)


def _pink_noise(rng: np.random.Generator, n_ch: int, n_t: int, sfreq: float,
                beta: float = 1.2) -> np.ndarray:
    """1/f^beta 粉红噪声 —— 脑电背景活动的基本形态。"""
    white = rng.standard_normal((n_ch, n_t))
    freqs = np.fft.rfftfreq(n_t, d=1.0 / sfreq)
    scale = np.ones_like(freqs)
    scale[1:] = freqs[1:] ** (-beta / 2.0)
    scale[0] = 0.0  # 去直流
    spec = np.fft.rfft(white, axis=1) * scale[None, :]
    out = np.fft.irfft(spec, n=n_t, axis=1)
    out -= out.mean(axis=1, keepdims=True)
    out /= out.std(axis=1, keepdims=True) + 1e-12
    return out


def _spatial_mixing(pos: np.ndarray, sigma: float,
                    rng: np.random.Generator) -> np.ndarray:
    """由电极间距离构造空间协方差，返回 Cholesky 因子。

    乘上它之后，相邻电极的噪声就变成相关的 —— 真实的脑电噪声在头皮上
    是连续分布的，不是每个电极各自独立。
    """
    d2 = ((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1)
    cov = np.exp(-d2 / (2.0 * sigma ** 2))
    cov += np.eye(len(pos)) * 1e-6
    return np.linalg.cholesky(cov)


def _smooth_field(pos: np.ndarray, hotspots: Sequence[Tuple[str, float]],
                  ch_names: Sequence[str], sigma: float = 0.45) -> np.ndarray:
    """把若干皮层热点平滑成覆盖全脑的连续空间场。"""
    field = np.zeros(len(ch_names))
    for name, weight in hotspots:
        if name not in ch_names:
            continue
        i = ch_names.index(name)
        d2 = ((pos - pos[i]) ** 2).sum(-1)
        field += weight * np.exp(-d2 / (2.0 * sigma ** 2))
    return field


def _oscillation(rng: np.random.Generator, n_ch: int, n_t: int, sfreq: float,
                 freq: float, amp: float, env: Optional[np.ndarray] = None) -> np.ndarray:
    """带随机相位的节律振荡，可选时变包络。"""
    t = np.arange(n_t) / sfreq
    phase = rng.uniform(0, 2 * np.pi, size=(n_ch, 1))
    jitter = rng.normal(freq, freq * 0.03, size=(n_ch, 1))  # 每导联峰值频率略有差异
    wave = np.sin(2 * np.pi * jitter * t[None, :] + phase)
    if env is not None:
        wave = wave * env[None, :]
    return amp * wave


# ---------------------------------------------------------------------------
#  单试次生成
# ---------------------------------------------------------------------------


def _subject_context(
    rng: np.random.Generator,
    preset: str,
    ch_names: Sequence[str],
    pos: np.ndarray,
    variability: float,
) -> Dict[str, object]:
    """为一位受试者生成"个体差异"：空间场、ERP 增益、潜伏期偏移、噪声混合矩阵。

    同一个类别在不同受试者身上的空间分布与幅度都不同 —— 这是跨受试者泛化难的根源，
    也是本仿真数据比"每类一个固定模板"更有意义的地方。
    """
    n_ch = len(ch_names)
    classes = PRESETS[preset]
    fields: Dict[str, np.ndarray] = {}
    for cname in classes:
        base = _smooth_field(pos, PRESET_SPECS[preset][cname].hotspots, ch_names)
        # 个体差异：在基础空间场上叠加一层平滑的随机扰动
        pert = rng.standard_normal(n_ch) * variability
        # 用空间混合矩阵把扰动也变平滑（否则是逐导联白噪声，不像脑电）
        pert = _spatial_mixing(pos, 0.45, rng) @ pert
        field = base + pert * (np.abs(base).mean() + eps_guard())
        fields[cname] = field

    return {
        "fields": fields,
        "erp_gain": float(np.exp(rng.normal(0.0, variability * 0.4))),
        "latency_shift": float(rng.normal(0.0, variability * 0.03)),
        "alpha_gain": float(np.exp(rng.normal(0.0, variability * 0.3))),
        "L": _spatial_mixing(pos, 0.40, rng),
    }


def eps_guard() -> float:
    return 1e-12


def _generate_trial(
    rng: np.random.Generator,
    preset: str,
    class_name: str,
    ctx: Dict[str, object],
    t: np.ndarray,
    ch_names: Sequence[str],
    sfreq: float,
    noise_scale: float,
    blink_prob: float,
    muscle_prob: float,
    line_noise_uv: float,
    dropout_prob: float,
    include_background: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """生成一个试次的 ``(erp, noise)``，形状均为 ``(n_ch, n_t)``。

    两者分开返回，是为了能按目标 SNR 精确缩放 ERP —— 背景幅度保持生理合理。

    ``include_background=False`` 时只返回 ERP 与伪迹，不叠加 1/f 与节律背景。
    连续记录（``generate_raw_recording``）需要这个模式：背景要整段连续生成，
    逐试次再生成会在试次边界产生阶跃。
    """
    n_ch, n_t = len(ch_names), t.size
    spec = PRESET_SPECS[preset][class_name]

    # ---- 1. 诱发成分 ----
    shift = ctx["latency_shift"] + rng.normal(0.0, 0.020)  # 试次间潜伏期抖动 20ms
    erp_t = np.zeros(n_t)
    for mu, sd, amp in spec.erp:
        erp_t += amp * _gaussian(t, mu + shift, sd)
    erp_t *= ctx["erp_gain"] * rng.lognormal(0.0, 0.15)     # 试次间幅度抖动
    # 基线归零：用 onset 前的均值校正，符合真实数据处理习惯
    pre = t < 0
    if pre.any():
        erp_t -= erp_t[pre].mean()

    spat = ctx["fields"][class_name]
    spat = spat / (np.abs(spat).max() + eps_guard())
    erp = np.outer(spat, erp_t)

    # ---- 2. 背景：1/f 噪声 + α/β 节律 + 工频 ----
    if include_background:
        pink = ctx["L"] @ _pink_noise(rng, n_ch, n_t, sfreq)
        pink *= 15.0  # µV，典型自发脑电幅度

        # α 的事件相关去同步：onset 后包络平滑变化，各类别系数不同
        ramp = 0.5 * (1.0 + np.tanh((t - 0.15) / 0.08))
        alpha_env = 1.0 + (spec.alpha_erd - 1.0) * ramp
        alpha = _oscillation(rng, n_ch, n_t, sfreq, 10.0, 8.0 * ctx["alpha_gain"], alpha_env)
        beta = _oscillation(rng, n_ch, n_t, sfreq, 20.0, 4.0)

        noise = pink + alpha + beta

        # 工频干扰（中国市电 50Hz）
        if line_noise_uv > 0:
            ph = rng.uniform(0, 2 * np.pi)
            noise += line_noise_uv * np.sin(2 * np.pi * 50.0 * t + ph)
    else:
        noise = np.zeros((n_ch, n_t), dtype=np.float64)

    # ---- 3. 伪迹 ----
    if rng.random() < blink_prob:
        # 眨眼：额区大慢波，可以很不对称
        frontal = np.array([3.0 if c.startswith("Fp") else
                            1.6 if c.startswith("F") else 0.2 for c in ch_names])
        frontal = frontal * rng.uniform(0.6, 1.4, size=n_ch)
        blink_t = _gaussian(t, rng.uniform(0.05, 0.4), rng.uniform(0.08, 0.15))
        noise += np.outer(frontal * rng.uniform(40, 90), blink_t)

    if rng.random() < muscle_prob:
        # 肌电：颞区高频爆发
        temporal = np.array([2.5 if c.startswith("T") else
                             1.5 if c.startswith(("F", "C")) else 0.3 for c in ch_names])
        burst = _gaussian(t, rng.uniform(0.1, 0.7), 0.12)
        emg = rng.standard_normal((n_ch, n_t)) * temporal[:, None] * burst[None, :]
        noise += emg * rng.uniform(15, 35)

    # ---- 4. 通道脱落 ----
    if rng.random() < dropout_prob:
        bad = rng.integers(0, n_ch)
        noise[bad] = 0.0
        erp[bad] = 0.0

    return (erp * noise_scale).astype(np.float32), noise.astype(np.float32)


# ---------------------------------------------------------------------------
#  对外接口
# ---------------------------------------------------------------------------


def generate_dataset(
    cfg: Config,
    preset: Optional[str] = None,
    snr_db: Optional[float] = None,
    n_subjects: Optional[int] = None,
    n_sessions: Optional[int] = None,
    trials_per_class: Optional[int] = None,
    seed: Optional[int] = None,
) -> EEGDataset:
    """按配置生成仿真数据集。

    参数都可覆盖配置里的值 —— 做 SNR 扫描、跨受试者评估时很方便。
    """
    s = cfg.data.synthetic
    preset = preset or s.preset
    snr_db = s.snr_db if snr_db is None else snr_db
    n_subjects = n_subjects or s.n_subjects
    n_sessions = n_sessions or s.n_sessions
    trials_per_class = trials_per_class or s.trials_per_class
    seed = cfg.seed if seed is None else seed

    if preset not in PRESET_SPECS:
        raise ValueError(f"未知预设 {preset!r}，可选：{sorted(PRESET_SPECS)}")
    if preset not in PRESETS:
        raise ValueError(f"预设 {preset!r} 未在 config.PRESETS 中登记类别名")
    if len(PRESETS[preset]) != len(PRESET_SPECS[preset]):
        raise ValueError(
            f"预设 {preset!r} 的类别名 {PRESETS[preset]} 与信号规格 "
            f"{sorted(PRESET_SPECS[preset])} 数量不一致"
        )

    class_names = list(PRESETS[preset])
    ch_names = CH_NAMES_32[: s.n_channels]
    pos = channel_positions(ch_names)
    sfreq = float(s.sfreq)
    n_t = int(round((cfg.preprocess.tmax - cfg.preprocess.tmin) * sfreq))
    t = cfg.preprocess.tmin + np.arange(n_t) / sfreq

    # ERP 峰值 = 背景 RMS × 10^(snr/20)，单位 µV
    noise_scale = snr_to_scale(snr_db)

    rng = np.random.default_rng(seed)
    trials: List[np.ndarray] = []
    labels: List[int] = []
    subjects: List[int] = []
    sessions: List[int] = []

    for subj in range(n_subjects):
        # 每位受试者一个独立子种子 —— 增删受试者不会打乱其他人的数据
        subj_rng = np.random.default_rng(rng.integers(0, 2 ** 31 - 1))
        ctx = _subject_context(subj_rng, preset, ch_names, pos, s.subject_variability)

        for sess in range(n_sessions):
            for ci, cname in enumerate(class_names):
                for _ in range(trials_per_class):
                    erp, noise = _generate_trial(
                        subj_rng, preset, cname, ctx, t, ch_names, sfreq,
                        noise_scale, s.blink_prob, s.muscle_prob,
                        s.line_noise_uv, s.dropout_prob,
                    )
                    trials.append(erp + noise)
                    labels.append(ci)
                    subjects.append(subj)
                    sessions.append(sess)

    X = np.stack(trials)  # (n_trials, n_ch, n_t)
    y = np.array(labels, dtype=np.int64)

    # 打乱顺序，避免同类试次扎堆（否则交叉验证的折会不均衡）
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]
    subjects_arr = np.array(subjects, dtype=np.int64)[perm]
    sessions_arr = np.array(sessions, dtype=np.int64)[perm]

    return EEGDataset(
        X=X,
        y=y,
        class_names=class_names,
        sfreq=sfreq,
        tmin=float(cfg.preprocess.tmin),
        ch_names=list(ch_names),
        subject=subjects_arr,
        session=sessions_arr,
        is_synthetic=True,
        meta={
            "preset": preset,
            "preset_label": None,
            "snr_db": float(snr_db),
            "n_subjects": n_subjects,
            "n_sessions": n_sessions,
            "trials_per_class": trials_per_class,
            "seed": int(seed),
            "generator": "cneeg.data.synthetic",
        },
    )


def generate_raw_recording(
    cfg: Config,
    preset: Optional[str] = None,
    duration_s: float = 120.0,
    trial_dur_s: float = 6.0,
    subject: int = 0,
    seed: Optional[int] = None,
) -> Dict[str, object]:
    """生成一段**连续**记录 + 事件表 —— 模拟真实的采集文件。

    用途有两个：

    1. 给界面"推理"页和真实数据适配器（``loaders``）提供等价的输入结构，
       这样接真数据之前就能把整条链路测通；
    2. 单测里验证"从连续记录切出试次"的逻辑。

    返回 dict：``data (n_ch, n_t)``、``sfreq``、``ch_names``、``events``
    （onset 采样点索引）、``event_codes``（类别索引）、``class_names``、
    ``is_synthetic=True``。
    """
    s = cfg.data.synthetic
    preset = preset or s.preset
    seed = cfg.seed if seed is None else seed
    class_names = list(PRESETS[preset])
    ch_names = CH_NAMES_32[: s.n_channels]
    pos = channel_positions(ch_names)
    sfreq = float(s.sfreq)

    n_trial = int(trial_dur_s * sfreq)
    n_total = int(duration_s * sfreq)
    n_trials = max(1, n_total // n_trial)
    t = np.arange(n_trial) / sfreq

    rng = np.random.default_rng(seed + 9973 + subject)
    ctx = _subject_context(rng, preset, ch_names, pos, s.subject_variability)

    # 连续背景：整段一次性生成，试次首尾相接（和真实记录一样没有断点）
    pink = ctx["L"] @ _pink_noise(rng, len(ch_names), n_total, sfreq) * 15.0
    data = pink
    data += _oscillation(rng, len(ch_names), n_total, sfreq, 10.0, 8.0 * ctx["alpha_gain"])
    data += _oscillation(rng, len(ch_names), n_total, sfreq, 20.0, 4.0)
    if s.line_noise_uv > 0:
        tt = np.arange(n_total) / sfreq
        data += s.line_noise_uv * np.sin(2 * np.pi * 50.0 * tt + rng.uniform(0, 2 * np.pi))

    events, codes = [], []
    noise_scale = snr_to_scale(s.snr_db)
    for k in range(n_trials):
        cname = class_names[k % len(class_names)]
        erp, artifacts = _generate_trial(
            rng, preset, cname, ctx, t, ch_names, sfreq, noise_scale,
            s.blink_prob, s.muscle_prob, 0.0, s.dropout_prob,
            include_background=False,
        )
        onset = k * n_trial
        data[:, onset:onset + n_trial] += erp + artifacts
        events.append(onset)
        codes.append(class_names.index(cname))

    return {
        "data": data.astype(np.float32),
        "sfreq": sfreq,
        "ch_names": list(ch_names),
        "events": np.array(events, dtype=np.int64),
        "event_codes": np.array(codes, dtype=np.int64),
        "class_names": class_names,
        "is_synthetic": True,
        "meta": {"preset": preset, "subject": subject, "seed": int(seed)},
    }
