# -*- coding: utf-8 -*-
"""
真实数据适配器
==============

把实验室采集的脑电文件读成统一的 ``RawRecording``，交给
``epoching.preprocess_raw`` 处理。**接口与仿真数据完全一致** ——
所以从仿真切到真实数据，只需要改配置里的 ``data.source``，模型、评估、
界面、报告都不用动。

支持的格式（按扩展名自动分发）:

===========  ==========================  ====================================
扩展名        读取函数                     来源
===========  ==========================  ====================================
``.mff``      ``mne.io.read_raw_egi``     **EGI NetStation**（本实验室用）
``.vhdr``     ``read_raw_brainvision``    BrainVision
``.edf``      ``read_raw_edf``            EDF / EDF+
``.bdf``      ``read_raw_bdf``            BioSemi
``.cnt``      ``read_raw_cnt``            NeuroScan
``.set``      ``read_raw_eeglab``         EEGLAB
``.fif``      ``read_raw_fif``            MNE
===========  ==========================  ====================================

事件来源有两种，按优先级：

1. **EEG 文件内嵌事件轨**（EGI 的 ``.mff`` 会把 NetStation 收到的 marker 存进去）
   —— 这是首选，时间对齐最可靠。
2. **PsychoPy 打标日志**（``trials_*.csv``）—— 当文件里没有事件、或者需要
   交叉核对时使用。日志里的时间是"相对实验程序启动"，而脑电记录有自己的起点，
   两者的差值是 ``data.file.onset_offset_s``，**这个偏移必须自己确认**，
   填错了整份数据的事件位置就全偏了。

.. warning::
   ``onset_offset_s`` 无法自动推断 —— 事件对齐错了会得到"看起来正常但其实是
   错的"结果，是脑电分析里最隐蔽的错误来源。正式分析前请用一个已知的
   显著事件（比如实验开始的那一下）核对偏移量。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import Config

#: 扩展名 → MNE 读取函数名（在 ``mne.io`` 里查找）
SUPPORTED_FORMATS: Dict[str, str] = {
    ".mff": "read_raw_egi",
    ".vhdr": "read_raw_brainvision",
    ".edf": "read_raw_edf",
    ".bdf": "read_raw_bdf",
    ".cnt": "read_raw_cnt",
    ".set": "read_raw_eeglab",
    ".fif": "read_raw_fif",
    ".gz": "read_raw_fif",     # .fif.gz
}


@dataclass
class RawRecording:
    """一段连续脑电记录 + 事件表。真实数据与仿真的统一入口结构。"""

    data: np.ndarray                    # (n_channels, n_times)，单位 µV
    sfreq: float
    ch_names: List[str]
    events: np.ndarray                  # 事件起始采样点索引
    event_codes: np.ndarray             # 类别索引，对应 class_names
    class_names: List[str]
    is_synthetic: bool = False
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return self.data.shape[1] / self.sfreq

    def summary(self) -> str:
        return "\n".join([
            f"数据来源：{'仿真' if self.is_synthetic else '真实采集'}",
            f"通道数：{len(self.ch_names)}　采样率：{self.sfreq:g} Hz",
            f"时长：{self.duration_s:.1f} 秒（{self.data.shape[1]} 采样点）",
            f"事件数：{len(self.events)}　类别：{self.class_names}",
            f"幅度范围：{self.data.min():.1f} ~ {self.data.max():.1f} µV",
        ])


# ---------------------------------------------------------------------------
#  格式分发
# ---------------------------------------------------------------------------


def _detect_format(path: Path) -> str:
    suffixes = [s.lower() for s in path.suffixes]
    if not suffixes:
        raise ValueError(f"无法从文件名判断格式（没有扩展名）：{path}")
    if suffixes[-1] == ".gz" and len(suffixes) >= 2:
        return ".gz"
    return suffixes[-1]


def load_mne_raw(path: str | Path, verbose: str = "ERROR"):
    """按扩展名调用对应的 MNE 读取函数，返回 ``mne.io.BaseRaw``。"""
    try:
        import mne
    except ImportError as exc:      # pragma: no cover
        raise ImportError(
            "读取真实脑电文件需要 MNE。请先安装：\n"
            "    pip install mne -i https://pypi.tuna.tsinghua.edu.cn/simple"
        ) from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件：{path}")

    fmt = _detect_format(path)
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(
            f"不支持的文件格式 {fmt!r}。支持的格式：{sorted(SUPPORTED_FORMATS)}"
        )

    reader = getattr(mne.io, SUPPORTED_FORMATS[fmt])
    return reader(str(path), preload=True, verbose=verbose)


def inspect_file(path: str | Path) -> Dict[str, object]:
    """快速看一眼文件里有什么 —— 界面上"导入真实数据"时先调它。

    只读文件头，不加载全部数据，避免大文件卡住界面。
    """
    path = Path(path)
    fmt = _detect_format(path)
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"不支持的文件格式 {fmt!r}")

    raw = load_mne_raw(path)
    try:
        info: Dict[str, object] = {
            "path": str(path),
            "format": fmt,
            "sfreq": float(raw.info["sfreq"]),
            "n_channels": len(raw.ch_names),
            "ch_names": list(raw.ch_names),
            "duration_s": float(raw.times[-1]) if len(raw.times) else 0.0,
        }
        try:
            ann = raw.annotations
            info["n_annotations"] = len(ann)
            info["annotation_labels"] = sorted({str(d) for d in ann.description})[:30]
        except Exception:
            info["n_annotations"] = 0
            info["annotation_labels"] = []
        return info
    finally:
        try:
            raw.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
#  事件提取
# ---------------------------------------------------------------------------


def events_from_annotations(raw, marker_map: Dict[str, str],
                            class_names: Sequence[str],
                            onset_offset_s: float = 0.0
                            ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """从 MNE ``Raw`` 的 annotations 里抽取类别事件。

    ``marker_map`` 把文件里的 marker 字符串映射到类别名。匹配时会先做
    ``strip()`` —— NetStation 的 ``event_type`` 必须恰好 4 个字符，所以
    ``yes`` 实际存的是 ``"YES "``（末尾补空格），直接比对会全部匹配不上。
    """
    sfreq = float(raw.info["sfreq"])
    norm_map = {str(k).strip().upper(): v for k, v in marker_map.items()}
    label_to_code = {str(c): i for i, c in enumerate(class_names)}

    onsets, codes, matched = [], [], {}
    try:
        ann = raw.annotations
    except Exception:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), []

    for onset, desc in zip(ann.onset, ann.description):
        key = str(desc).strip().upper()
        if key not in norm_map:
            continue
        cls = norm_map[key]
        if cls not in label_to_code:
            continue
        sample = int(round((float(onset) + onset_offset_s) * sfreq))
        onsets.append(sample)
        codes.append(label_to_code[cls])
        matched[key] = matched.get(key, 0) + 1

    order = np.argsort(onsets)
    return (np.array(onsets, dtype=np.int64)[order],
            np.array(codes, dtype=np.int64)[order],
            [f"{k}→{norm_map[k]}×{v}" for k, v in sorted(matched.items())])


def load_psychopy_log(csv_path: str | Path,
                      marker_map: Optional[Dict[str, str]] = None,
                      time_column: str = "word_onset_s",
                      label_column: str = "word",
                      ) -> Tuple[np.ndarray, List[str]]:
    """读 PsychoPy 打标日志（本实验室 ``word_reading_eeg.py`` 的输出）。

    日志列：``trial, word, marker, fix_onset_s, word_onset_s, rest_onset_s,
    fix_dur_actual, word_dur_actual``。

    返回 ``(相对实验起点的秒数数组, 类别名列表)``。

    ``time_column`` 默认取 ``word_onset_s`` —— 单词出现的那一刻，也就是脑电
    分析窗口的锁时点。用 ``fix_onset_s`` 的话锚点会偏移 0.5 秒。

    文件带 UTF-8 BOM（Excel/PsychoPy 常见），所以按 ``utf-8-sig`` 读。
    """
    import csv

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到打标日志：{csv_path}")

    times: List[float] = []
    labels: List[str] = []
    norm_map = {str(k).strip().upper(): v
                for k, v in (marker_map or {}).items()}

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"打标日志为空或格式不对：{csv_path}")
        fields = [f.strip() for f in reader.fieldnames]
        if time_column not in fields:
            raise ValueError(
                f"打标日志里没有列 {time_column!r}。实际列：{fields}"
            )

        for row in reader:
            row = {(k or "").strip(): v for k, v in row.items()}
            raw_t = row.get(time_column, "")
            if raw_t in ("", None):
                continue
            try:
                t = float(raw_t)
            except ValueError:
                continue

            label = (row.get(label_column) or "").strip()
            if not label:
                marker = (row.get("marker") or "").strip().upper()
                label = norm_map.get(marker, "")
            if not label:
                continue

            times.append(t)
            labels.append(label)

    if not times:
        raise ValueError(
            f"从 {csv_path} 里没有解析出任何事件。请检查列名与内容是否与 "
            f"word_reading_eeg.py 的输出格式一致。"
        )
    return np.array(times, dtype=np.float64), labels


def events_from_log_times(times: Sequence[float],
                          labels: Sequence[str],
                          class_names: Sequence[str],
                          sfreq: float,
                          n_times: int,
                          onset_offset_s: float = 0.0,
                          ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """把打标日志的秒数换算成采样点索引。

    这是整条真实数据链路上**最容易错、错了又看不出来**的一步，所以单独
    拎出来：不依赖 MNE，可以脱离 EEG 文件单独测。

    日志的时间原点（实验程序启动）和脑电记录的原点（开始录制）不是同一个，
    两者之差就是 ``onset_offset_s``。不加它，所有事件会整体平移到错误位置，
    而截出来的波形**看上去完全正常** —— 只是锁时点对不上，ERP 被抹平。

    落在 ``[0, n_times)`` 之外的事件会被丢掉而不是保留：负索引在分段时
    会从数据末尾绕回来，取到一段毫不相干的信号。

    返回 ``(采样点索引, 类别编码, 诊断信息)``。
    """
    label_to_code = {str(c): i for i, c in enumerate(class_names)}
    keep: List[Tuple[int, int]] = []
    n_unmapped = n_outside = 0

    for t, l in zip(times, labels):
        if l not in label_to_code:
            n_unmapped += 1
            continue
        sample = int(round((float(t) + onset_offset_s) * sfreq))
        if sample < 0 or sample >= n_times:
            n_outside += 1
            continue
        keep.append((sample, label_to_code[l]))

    diag = [f"日志共 {len(times)} 条事件，采用 {len(keep)} 条"]
    if n_unmapped:
        diag.append(f"类别未在 marker_map 中：{n_unmapped} 条")
    if n_outside:
        diag.append(
            f"落在记录范围外：{n_outside} 条 —— 多半是 "
            f"data.file.onset_offset_s（当前 {onset_offset_s:+.3f} s）填得不对")

    return (np.array([k[0] for k in keep], dtype=np.int64),
            np.array([k[1] for k in keep], dtype=np.int64),
            diag)


def load_recording(path: str | Path, cfg: Config) -> RawRecording:
    """读取真实脑电文件，组装成 ``RawRecording``。

    事件优先取文件内嵌事件轨；没有匹配到任何事件时，若配置了
    ``marker_log``，改用 PsychoPy 日志。
    """
    f = cfg.data.file
    path = Path(path)

    raw = load_mne_raw(path)
    try:
        data = raw.get_data() * 1e6          # V → µV
        sfreq = float(raw.info["sfreq"])
        ch_names = list(raw.ch_names)
        class_names = _class_names_from_map(f.marker_map)

        events, codes, matched = events_from_annotations(
            raw, f.marker_map, class_names, f.onset_offset_s)

        source = "文件内嵌事件轨"
        # 日志那条路的诊断信息，留到下面报错时用 —— 最可能出错的地方是
        # onset_offset_s 填错，而不是 marker 字符串对不上。
        log_diag: List[str] = []
        if len(events) == 0 and f.marker_log:
            times, labels = load_psychopy_log(f.marker_log, f.marker_map)
            events, codes, log_diag = events_from_log_times(
                times, labels, class_names, sfreq,
                n_times=int(data.shape[-1]), onset_offset_s=f.onset_offset_s)
            if len(events):
                matched = list(log_diag)
                source = f"PsychoPy 打标日志（偏移 {f.onset_offset_s:+.3f} s）"

        if len(events) == 0:
            extra = ""
            if f.marker_log and log_diag:
                extra = ("  打标日志：" + "；".join(log_diag) + "\n")
            raise ValueError(
                f"从 {path.name} 里没有读到任何可用事件。\n"
                f"  文件里找到的标注：{sorted({str(d) for d in raw.annotations.description})[:20]}\n"
                f"  配置的 marker 映射：{f.marker_map}\n"
                f"{extra}"
                "请确认 marker 字符串是否完全一致（NetStation 的 event_type "
                "恰好 4 个字符，不足会补空格，例如 'YES '）。"
            )

        return RawRecording(
            data=data.astype(np.float32),
            sfreq=sfreq,
            ch_names=ch_names,
            events=events,
            event_codes=codes,
            class_names=class_names,
            is_synthetic=False,
            meta={
                "path": str(path),
                "event_source": source,
                "matched": matched,
                "n_events": int(len(events)),
            },
        )
    finally:
        try:
            raw.close()
        except Exception:
            pass


def _class_names_from_map(marker_map: Dict[str, str]) -> List[str]:
    """从 marker 映射里推出类别顺序（按首次出现的顺序，保证可复现）。"""
    seen: List[str] = []
    for v in marker_map.values():
        if v not in seen:
            seen.append(v)
    return seen


# ---------------------------------------------------------------------------
#  仿真数据的"伪真实"入口 —— 用来在接真数据之前先把整条链路测通
# ---------------------------------------------------------------------------


def load_demo_recording(cfg: Config, subject: int = 0,
                        duration_s: float = 120.0) -> RawRecording:
    """用仿真生成器造一段连续记录，冒充"从文件读进来"的数据。

    这样界面的"导入数据 → 预处理 → 训练"整条链路在真实数据到位之前
    就能完整跑通并被测试覆盖，不用等到实验室采集完才发现接口对不上。
    """
    from .synthetic import generate_raw_recording

    rec = generate_raw_recording(cfg, duration_s=duration_s, subject=subject)
    return RawRecording(
        data=rec["data"],
        sfreq=rec["sfreq"],
        ch_names=list(rec["ch_names"]),
        events=rec["events"],
        event_codes=rec["event_codes"],
        class_names=list(rec["class_names"]),
        is_synthetic=True,
        meta={**rec["meta"], "event_source": "仿真连续记录"},
    )


def load_any(path: str | Path | None, cfg: Config) -> RawRecording:
    """统一入口：``path`` 为空时返回仿真记录，否则读真实文件。"""
    if path in (None, ""):
        return load_demo_recording(cfg)
    return load_recording(path, cfg)
