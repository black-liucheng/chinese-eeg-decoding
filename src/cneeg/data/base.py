# -*- coding: utf-8 -*-
"""
数据容器与电极坐标
==================

``EEGDataset`` 是贯穿整个流水线的统一数据结构 —— 不管数据来自仿真生成器
还是实验室的真实 ``.mff`` 文件，下游的预处理、特征、模型、界面看到的都是它。

约定：``X`` 的形状恒为 ``(n_trials, n_channels, n_times)``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
#  电极坐标（10-20 系统 32 导，归一化 2D 布局）
#  x 向右为正，y 向前（额区）为正。用于空间平滑与拓扑图。
# ---------------------------------------------------------------------------

CH_NAMES_32: List[str] = [
    "Fp1", "Fp2",
    "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC1", "FC2", "FC6",
    "T7", "C3", "Cz", "C4", "T8",
    "TP9", "CP5", "CP1", "CP2", "CP6", "TP10",
    "P7", "P3", "Pz", "P4", "P8",
    "PO9", "O1", "Oz", "O2", "PO10",
]

CH_POS_2D: Dict[str, tuple] = {
    "Fp1": (-0.30, 0.95), "Fp2": (0.30, 0.95),
    "F7": (-0.75, 0.60), "F3": (-0.38, 0.62), "Fz": (0.0, 0.65),
    "F4": (0.38, 0.62), "F8": (0.75, 0.60),
    "FC5": (-0.62, 0.35), "FC1": (-0.20, 0.38), "FC2": (0.20, 0.38), "FC6": (0.62, 0.35),
    "T7": (-0.90, 0.0), "C3": (-0.42, 0.0), "Cz": (0.0, 0.0),
    "C4": (0.42, 0.0), "T8": (0.90, 0.0),
    "TP9": (-0.85, -0.35), "CP5": (-0.60, -0.28), "CP1": (-0.20, -0.30),
    "CP2": (0.20, -0.30), "CP6": (0.60, -0.28), "TP10": (0.85, -0.35),
    "P7": (-0.78, -0.55), "P3": (-0.38, -0.55), "Pz": (0.0, -0.58),
    "P4": (0.38, -0.55), "P8": (0.78, -0.55),
    "PO9": (-0.62, -0.80), "O1": (-0.30, -0.92), "Oz": (0.0, -0.95),
    "O2": (0.30, -0.92), "PO10": (0.62, -0.80),
}


def channel_positions(ch_names: List[str]) -> np.ndarray:
    """取电极 2D 坐标，形状 ``(n_channels, 2)``。未知电极落在原点。"""
    return np.array([CH_POS_2D.get(c, (0.0, 0.0)) for c in ch_names], dtype=np.float64)


# ---------------------------------------------------------------------------
#  数据集
# ---------------------------------------------------------------------------


@dataclass
class EEGDataset:
    """一段分好段（epoched）的脑电数据。

    Attributes
    ----------
    X : (n_trials, n_channels, n_times) float32
        单位微伏（µV）。
    y : (n_trials,) int
        类别索引，对应 ``class_names`` 的下标。
    class_names : list of str
        类别名，顺序即 ``y`` 的取值含义。
    sfreq : float
        采样率 Hz。
    tmin : float
        每个试次第一个采样点相对事件 onset 的时间（秒），通常为负（基线段）。
    ch_names : list of str
        电极名。
    subject : (n_trials,) int, optional
        受试者编号，用于跨受试者评估。
    session : (n_trials,) int, optional
        场次编号。
    is_synthetic : bool
        是否来自仿真生成器。**所有对外产出（图、报告、文档）都要据此标注**，
        避免仿真结论被误当成真实实验结果。
    meta : dict
        其他元信息（预设名、SNR、生成参数等）。
    """

    X: np.ndarray
    y: np.ndarray
    class_names: List[str]
    sfreq: float
    tmin: float
    ch_names: List[str]
    subject: Optional[np.ndarray] = None
    session: Optional[np.ndarray] = None
    is_synthetic: bool = True
    meta: Dict[str, Any] = field(default_factory=dict)

    # -- 形状 --
    @property
    def n_trials(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_times(self) -> int:
        return int(self.X.shape[2])

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @property
    def times(self) -> np.ndarray:
        """每个采样点相对 onset 的时间轴（秒）。"""
        return self.tmin + np.arange(self.n_times) / self.sfreq

    @property
    def chance_level(self) -> float:
        """准确率的"什么都没学到"基线 —— 图里画虚线，判断模型是否真的学到东西。

        取**多数类的占比**，而不是 ``1/n_classes``。两者在类别均衡时完全相等
        （所以要"始终 25%"这种说法在均衡数据上没错），但类别一旦不均衡，
        ``1/n_classes`` 会**低估**基线：一个把所有试次都判成多数类的模型，
        准确率等于多数类占比，它什么都没学到，却可能"看起来"高于 1/类别数。

        这不是假想 —— 伪迹剔除本来就是按试次独立触发的，剔除之后各类别
        剩多少并不相等。真实数据上类别不均衡只会更常见。
        """
        if self.y.size == 0:
            return 1.0 / self.n_classes
        counts = np.bincount(self.y, minlength=self.n_classes)
        return float(counts.max()) / float(self.y.size)

    # -- 构造 --
    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=np.float32)
        self.y = np.asarray(self.y, dtype=np.int64)
        if self.X.ndim != 3:
            raise ValueError(f"X 应该是 3 维 (试次, 电极, 时间)，实际 shape={self.X.shape}")
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(f"X 与 y 的试次数不一致：{self.X.shape[0]} vs {self.y.shape[0]}")
        if self.X.shape[1] != len(self.ch_names):
            raise ValueError(
                f"X 的电极数 {self.X.shape[1]} 与 ch_names 长度 {len(self.ch_names)} 不一致"
            )
        if self.y.size and (self.y.min() < 0 or self.y.max() >= self.n_classes):
            raise ValueError(f"y 的取值应在 [0, {self.n_classes}) 内，实际 [{self.y.min()}, {self.y.max()}]")

    # -- 切分 --
    def subset(self, idx: np.ndarray) -> "EEGDataset":
        """按索引取子集 —— 交叉验证切折、划分训练测试集都用它。"""
        idx = np.asarray(idx)
        return EEGDataset(
            X=self.X[idx],
            y=self.y[idx],
            class_names=list(self.class_names),
            sfreq=self.sfreq,
            tmin=self.tmin,
            ch_names=list(self.ch_names),
            subject=None if self.subject is None else self.subject[idx],
            session=None if self.session is None else self.session[idx],
            is_synthetic=self.is_synthetic,
            meta=dict(self.meta),
        )

    def class_counts(self) -> Dict[str, int]:
        return {name: int((self.y == i).sum()) for i, name in enumerate(self.class_names)}

    # -- 持久化 --
    def fingerprint(self) -> str:
        """按**数据内容**算指纹，用来判断两份数据集是不是同一份。

        为什么需要它：随机种子固定，同样的配置重跑会生成**逐字节相同**的
        47 MB 数据集。而 ``run_cli.bat`` 每次跑都先执行一遍 gen-data ——
        不加判断的话，跑三次磁盘上就躺三份一模一样的数据，
        OneDrive 还得同步三遍。

        只对实际参与建模的部分（波形、标签、电极名、采样率、时间窗）取哈希，
        不掺入生成时间之类的元信息 —— 否则内容一样也会算出不同指纹。
        """
        import hashlib

        h = hashlib.blake2b(digest_size=16)
        h.update(np.ascontiguousarray(self.X, dtype=np.float32).tobytes())
        h.update(np.ascontiguousarray(self.y, dtype=np.int64).tobytes())
        h.update(("|".join(self.class_names)).encode("utf-8"))
        h.update(("|".join(self.ch_names)).encode("utf-8"))
        h.update(f"|{self.sfreq!r}|{self.tmin!r}".encode("utf-8"))
        return h.hexdigest()

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            X=self.X,
            y=self.y,
            class_names=np.array(self.class_names, dtype=object),
            sfreq=self.sfreq,
            tmin=self.tmin,
            ch_names=np.array(self.ch_names, dtype=object),
            subject=self.subject if self.subject is not None else np.array([]),
            session=self.session if self.session is not None else np.array([]),
            is_synthetic=self.is_synthetic,
            meta=np.array([self.meta], dtype=object),
            # 单独存一份指纹：读它只要解压这一个小数组，不必把 47 MB 的
            # 波形全读进来 —— 否则"查重"比"重写一份"还慢。
            fingerprint=np.array(self.fingerprint()),
        )
        return path

    @staticmethod
    def stored_fingerprint(path: str | Path) -> str:
        """只读 npz 里的指纹，不解压波形。老文件没有这个字段时返回空串。"""
        try:
            with np.load(path, allow_pickle=True) as d:
                if "fingerprint" in d.files:
                    return str(d["fingerprint"])
        except Exception:
            pass
        return ""

    @classmethod
    def load(cls, path: str | Path) -> "EEGDataset":
        path = Path(path)
        with np.load(path, allow_pickle=True) as d:
            subject = d["subject"] if d["subject"].size else None
            session = d["session"] if d["session"].size else None
            meta = d["meta"][0] if d["meta"].size else {}
            return cls(
                X=d["X"],
                y=d["y"],
                class_names=[str(s) for s in d["class_names"]],
                sfreq=float(d["sfreq"]),
                tmin=float(d["tmin"]),
                ch_names=[str(s) for s in d["ch_names"]],
                subject=subject,
                session=session,
                is_synthetic=bool(d["is_synthetic"]),
                meta=dict(meta) if isinstance(meta, dict) else {},
            )

    # -- 展示 --
    def summary(self) -> str:
        src = "仿真数据" if self.is_synthetic else "真实数据"
        lines = [
            f"数据来源：{src}",
            f"试次数：{self.n_trials}",
            f"电极数：{self.n_channels}　采样率：{self.sfreq:g} Hz",
            f"时间窗：{self.times[0]:.3f} ~ {self.times[-1]:.3f} s（{self.n_times} 采样点）",
            f"类别数：{self.n_classes}　随机基线：{self.chance_level:.1%}",
            "类别分布：" + "，".join(f"{k}={v}" for k, v in self.class_counts().items()),
        ]
        if self.subject is not None:
            lines.append(f"受试者数：{len(np.unique(self.subject))}")
        if self.is_synthetic:
            # 注意：Windows 中文控制台默认 GBK，⚠ / ☑ 这类符号不在 GBK 字符集内，
            # 打印时会抛 UnicodeEncodeError。本项目所有控制台输出只用 GBK 安全字符。
            lines.append("【注意】本数据为仿真生成，仅用于验证软件流水线，不作为真实实验结论。")
        return "\n".join(lines)
