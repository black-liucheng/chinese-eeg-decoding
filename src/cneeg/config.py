# -*- coding: utf-8 -*-
"""
配置加载与校验
==============

把 ``configs/default.yaml`` 读成带类型的数据类，缺项自动取默认值，
未知的键会给出明确报错而不是被静默忽略（少写一个字母就白跑一轮训练，很坑）。

用法::

    from cneeg.config import load_config
    cfg = load_config()                    # 读 configs/default.yaml
    cfg = load_config("my.yaml")           # 读指定文件
    cfg = load_config(overrides={"seed": 42, "model.name": "cnn"})
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ---------------------------------------------------------------------------
#  路径
# ---------------------------------------------------------------------------

#: 项目根目录 —— 即包含 ``configs/`` ``data/`` ``src/`` 的那一层
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: 默认配置文件
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


# ---------------------------------------------------------------------------
#  各配置段
# ---------------------------------------------------------------------------


@dataclass
class SyntheticConfig:
    preset: str = "vowel_tone"
    n_subjects: int = 8
    n_sessions: int = 2
    trials_per_class: int = 60
    n_channels: int = 32
    sfreq: float = 256.0
    snr_db: float = 2.0
    blink_prob: float = 0.08
    muscle_prob: float = 0.05
    line_noise_uv: float = 3.0
    dropout_prob: float = 0.02
    subject_variability: float = 0.35


@dataclass
class FileDataConfig:
    path: str = ""
    marker_log: str = ""
    marker_map: Dict[str, str] = field(default_factory=dict)
    montage: str = "standard_1020"
    onset_offset_s: float = 0.0


@dataclass
class DataConfig:
    source: str = "synthetic"  # synthetic | file
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    file: FileDataConfig = field(default_factory=FileDataConfig)


@dataclass
class PreprocessConfig:
    l_freq: float = 0.5
    h_freq: float = 40.0
    notch_freq: float = 50.0
    notch_width: float = 2.0
    reref: str = "car"  # car | none
    tmin: float = -0.2
    tmax: float = 1.0
    baseline: Optional[List[float]] = field(default_factory=lambda: [-0.2, 0.0])
    reject_uv: float = 150.0
    normalize: str = "zscore"  # zscore | none

    @property
    def n_times(self) -> int:
        return 1  # 占位，实际长度由 sfreq 决定；见 epochs 构建处


@dataclass
class FeaturesConfig:
    method: str = "stft"  # stft | bandpower
    bands: Dict[str, List[float]] = field(
        default_factory=lambda: {
            "delta": [1, 4],
            "theta": [4, 8],
            "alpha": [8, 13],
            "beta": [13, 30],
            "gamma": [30, 40],
        }
    )
    stft_nperseg: int = 64
    stft_noverlap: int = 48
    stft_use_bandpower: bool = True


@dataclass
class SVMConfig:
    C: float = 1.0
    gamma: str = "scale"
    kernel: str = "rbf"
    pca_components: int = 0


@dataclass
class CNNConfig:
    n_filters_time: int = 40
    filter_time_length: int = 25
    n_filters_spat: int = 40
    pool_time_length: int = 75
    pool_time_stride: int = 15
    dropout: float = 0.5
    n_epochs: int = 120
    batch_size: int = 32
    lr: float = 0.001
    weight_decay: float = 0.0
    early_stop_patience: int = 25
    class_weight: bool = True


@dataclass
class ModelConfig:
    name: str = "svm"  # svm | cnn
    svm: SVMConfig = field(default_factory=SVMConfig)
    cnn: CNNConfig = field(default_factory=CNNConfig)


@dataclass
class TrainConfig:
    split: str = "stratified_kfold"  # stratified_kfold | holdout
    n_splits: int = 5
    test_size: float = 0.2
    #: 留一受试者评估 —— 测试折的受试者在训练时完全没出现过。
    #: 这是更严格也更贴近实际部署的设定，结果通常明显低于普通交叉验证；
    #: 两者之差本身就说明"模型学到的是任务相关成分还是个体特征"。
    group_by_subject: bool = False
    permutation_test: bool = True
    n_permutations: int = 20


@dataclass
class OutputConfig:
    data_dir: str = "data"
    save_figures: bool = True
    figure_dpi: int = 150
    report_title: str = "汉语脑电信号分类实验报告"


@dataclass
class Config:
    seed: int = 20260913
    data: DataConfig = field(default_factory=DataConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # -- 路径快捷方式（都相对项目根目录）--
    @property
    def data_dir(self) -> Path:
        return PROJECT_ROOT / self.output.data_dir

    def sub_dir(self, *parts: str) -> Path:
        """取 ``data/`` 下的子目录并确保存在。例：``cfg.sub_dir("models")``"""
        p = self.data_dir.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p


# ---------------------------------------------------------------------------
#  构建
# ---------------------------------------------------------------------------


def _build(cls: type, raw: Any, where: str = "") -> Any:
    """把嵌套 dict 递归塞进 dataclass，未知键直接报错。"""
    if raw is None:
        return cls()
    if not isinstance(raw, dict):
        raise TypeError(f"配置项 {where or cls.__name__} 应为字典，实际是 {type(raw).__name__}")

    known = {f.name: f for f in fields(cls)}
    unknown = set(raw) - set(known)
    if unknown:
        raise KeyError(
            f"配置项 {where + '.' if where else ''}{sorted(unknown)} 不存在。"
            f"可选：{sorted(known)}"
        )

    kwargs: Dict[str, Any] = {}
    for name, f in known.items():
        if name not in raw:
            continue
        value = raw[name]
        target = f.type
        # 嵌套 dataclass（类型注解在 from __future__ annotations 下是字符串）
        sub = _resolve_dataclass(target, f)
        if sub is not None and isinstance(value, dict):
            kwargs[name] = _build(sub, value, f"{where}.{name}" if where else name)
        else:
            kwargs[name] = value
    return cls(**kwargs)


#: 注解字符串 -> 实际类型。仅在需要判断"是不是嵌套 dataclass"时用
_TYPE_REGISTRY = {
    "SyntheticConfig": SyntheticConfig,
    "FileDataConfig": FileDataConfig,
    "DataConfig": DataConfig,
    "PreprocessConfig": PreprocessConfig,
    "FeaturesConfig": FeaturesConfig,
    "SVMConfig": SVMConfig,
    "CNNConfig": CNNConfig,
    "ModelConfig": ModelConfig,
    "TrainConfig": TrainConfig,
    "OutputConfig": OutputConfig,
}


def _resolve_dataclass(annotation: Any, f: Any) -> Optional[type]:
    if is_dataclass(annotation) and isinstance(annotation, type):
        return annotation
    name = str(annotation)
    if name in _TYPE_REGISTRY:
        return _TYPE_REGISTRY[name]
    # dataclasses.field 的 default 可能是 dataclass 实例（default_factory 不算）
    if isinstance(f.default, type) and is_dataclass(f.default):
        return f.default
    return None


def _merge(base: dict, override: dict) -> dict:
    """深合并 —— override 覆盖 base，用于把 dotted overrides 塞进嵌套结构。"""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _expand_dotted(overrides: Dict[str, Any]) -> dict:
    """把 ``{"model.name": "cnn"}`` 展开成 ``{"model": {"name": "cnn"}}``。"""
    nested: dict = {}
    for key, value in overrides.items():
        if "." not in key:
            nested[key] = value
            continue
        parts = key.split(".")
        node = nested
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return nested


def _validate(cfg: Config) -> None:
    """跨字段的一致性检查 —— 这些错误早点报出来比跑一半崩掉强。"""
    if cfg.data.source not in {"synthetic", "file"}:
        raise ValueError(f"data.source 必须是 'synthetic' 或 'file'，实际是 {cfg.data.source!r}")
    if cfg.model.name not in {"svm", "cnn"}:
        raise ValueError(f"model.name 必须是 'svm' 或 'cnn'，实际是 {cfg.model.name!r}")
    if cfg.data.source == "synthetic" and cfg.data.synthetic.preset not in PRESETS:
        raise ValueError(
            f"未知的仿真预设 {cfg.data.synthetic.preset!r}。可选：{sorted(PRESETS)}"
        )
    if cfg.data.source == "file" and not cfg.data.file.path:
        raise ValueError("data.source='file' 时必须指定 data.file.path")
    if cfg.preprocess.tmax <= cfg.preprocess.tmin:
        raise ValueError("preprocess.tmax 必须大于 tmin")
    # 带通上下限写反是个静默杀手：滤波器照样返回数组，但结果是垃圾，
    # 而且要到模型准确率等于随机水平时才看得出来。这里当场拦住。
    if cfg.preprocess.l_freq >= cfg.preprocess.h_freq:
        raise ValueError(
            f"preprocess.l_freq ({cfg.preprocess.l_freq:g} Hz) 必须小于 "
            f"h_freq ({cfg.preprocess.h_freq:g} Hz)"
        )
    # 注意：这里**不**检查 h_freq 是否超过奈奎斯特频率。采样率对真实数据
    # 来自文件本身、对仿真数据来自 data.synthetic.sfreq，配置阶段无从得知
    # 最终会用哪个 —— 拿仿真采样率去卡真实数据会误伤（比如 1000 Hz 的记录
    # 配 100 Hz 低通是完全合法的）。这个检查留给处理数据的那一步。
    if cfg.preprocess.reref not in {"car", "none"}:
        raise ValueError(
            f"preprocess.reref 必须是 'car' 或 'none'，实际是 "
            f"{cfg.preprocess.reref!r}（rereference() 只实现了这两种）"
        )
    if cfg.preprocess.normalize not in {"zscore", "none"}:
        raise ValueError(
            f"preprocess.normalize 必须是 'zscore' 或 'none'，实际是 "
            f"{cfg.preprocess.normalize!r}"
        )
    if cfg.preprocess.reject_uv < 0:
        raise ValueError("preprocess.reject_uv 不能为负（0 表示不剔除）")
    if cfg.data.synthetic.n_subjects < 1:
        raise ValueError("data.synthetic.n_subjects 至少为 1")
    if cfg.data.synthetic.trials_per_class < 1:
        raise ValueError("data.synthetic.trials_per_class 至少为 1")
    if cfg.train.split == "stratified_kfold" and cfg.train.n_splits < 2:
        raise ValueError("train.n_splits 至少为 2")
    if cfg.train.n_permutations < 1:
        raise ValueError("train.n_permutations 至少为 1")


#: 仿真预设 -> 类别名。synthetic.py 会读这里的键
PRESETS: Dict[str, List[str]] = {
    "vowel_tone": ["a1", "a2", "a3", "a4"],
    "mandarin_word": ["ma", "ba", "da"],
    "english_word": ["yes", "or", "no"],
}

#: 预设的中文显示名，界面/图例用
PRESET_LABELS: Dict[str, str] = {
    "vowel_tone": "汉语声调（四声）",
    "mandarin_word": "汉语词汇（三词）",
    "english_word": "英语单词（yes/or/no）",
}

#: 每个类别的中文名，出图时用
CLASS_LABELS: Dict[str, str] = {
    "a1": "阴平 ā", "a2": "阳平 á", "a3": "上声 ǎ", "a4": "去声 à",
    "ma": "妈 mā", "ba": "八 bā", "da": "搭 dā",
    "yes": "yes", "or": "or", "no": "no",
}


def load_config(
    path: str | Path | None = None,
    overrides: Dict[str, Any] | None = None,
) -> Config:
    """读配置。``overrides`` 支持点号路径，如 ``{"model.name": "cnn"}``。"""
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"找不到配置文件：{path}")

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    if overrides:
        raw = _merge(raw, _expand_dotted(overrides))

    cfg = _build(Config, raw)
    _validate(cfg)
    return cfg


def dump_config(cfg: Config, path: str | Path) -> Path:
    """把生效配置回写一份 —— 每次实验都存一份，事后能复现。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def to_dict(obj: Any) -> Any:
        if is_dataclass(obj) and not isinstance(obj, type):
            return {f.name: to_dict(getattr(obj, f.name)) for f in fields(obj)}
        if isinstance(obj, dict):
            return {k: to_dict(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_dict(v) for v in obj]
        if isinstance(obj, Path):
            return str(obj)
        return obj

    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(to_dict(cfg), fh, allow_unicode=True, sort_keys=False)
    return path
