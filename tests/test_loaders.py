# -*- coding: utf-8 -*-
"""真实数据适配器测试 —— 打标日志解析与事件对齐。

为什么要单独测这块：把日志里的秒数换算成采样点索引，是整条真实数据链路上
**最容易错、错了又看不出来**的一步。偏移填错时截出来的波形看上去完全正常，
只是锁时点对不上、ERP 被抹平，往往要等到写论文才发现。

``tests/data/trials_sample.csv`` 是实验室 ``word_reading_eeg.py`` 的真实输出
（空跑那次的 ``trials_20260913_160740.csv``，含 UTF-8 BOM）。用真文件而不是
手搓的假数据，是为了锁住格式本身 —— 列名、BOM、精度都得对得上。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cneeg.data.loaders import (events_from_log_times, load_psychopy_log,
                                _class_names_from_map)

DATA = Path(__file__).resolve().parent / "data"
LOG = DATA / "trials_sample.csv"

#: 默认配置 marker_map 推出来的类别顺序（见 configs/default.yaml）
CLASSES = ["yes", "or", "no"]


# ---------------------------------------------------------------------------
#  打标日志解析
# ---------------------------------------------------------------------------


def test_real_lab_log_parses():
    """真实日志：5 个试次，时间取 word_onset_s（单词出现那一刻）。"""
    times, labels = load_psychopy_log(LOG)

    assert len(times) == 5
    assert labels == ["no", "no", "no", "yes", "no"]
    np.testing.assert_allclose(
        times, [16.6797, 22.1953, 27.7143, 33.2350, 38.7556], atol=1e-6)


def test_bom_does_not_corrupt_first_column():
    """文件带 UTF-8 BOM，首列名不能被读成 '\\ufefftrial'。

    按 utf-8 而不是 utf-8-sig 读的话，首列会多一个不可见字符 ——
    取值时看着列名一模一样却取不到，很难查。
    """
    import csv

    with open(LOG, "r", encoding="utf-8-sig", newline="") as fh:
        fields = next(csv.reader(fh))

    assert fields[0] == "trial"
    assert not fields[0].startswith("﻿")
    assert fields == ["trial", "word", "marker", "fix_onset_s", "word_onset_s",
                      "rest_onset_s", "fix_dur_actual", "word_dur_actual"]


def test_time_column_switches_anchor():
    """换用 fix_onset_s 当锚点，时间整体前移约 0.5 秒。

    这两个列都能读，但物理含义完全不同，选错了就是 0.5 秒的系统性错位。
    """
    t_word, _ = load_psychopy_log(LOG, time_column="word_onset_s")
    t_fix, _ = load_psychopy_log(LOG, time_column="fix_onset_s")

    assert len(t_fix) == len(t_word)
    lag = t_word - t_fix
    assert np.all(lag > 0.3) and np.all(lag < 0.8), lag


def test_missing_time_column_reports_actual_columns():
    """列名写错时，报错要把文件里真实存在的列列出来。"""
    with pytest.raises(ValueError) as ei:
        load_psychopy_log(LOG, time_column="onset_s")

    msg = str(ei.value)
    assert "onset_s" in msg
    assert "word_onset_s" in msg, "报错里应列出实际列名，方便对照"


def test_missing_file_raises_filenotfound():
    with pytest.raises(FileNotFoundError):
        load_psychopy_log(DATA / "不存在.csv")


def test_label_falls_back_to_marker_map(tmp_path):
    """word 列空着时，退回用 marker 列查映射表。"""
    p = tmp_path / "log.csv"
    p.write_text("trial,word,marker,word_onset_s\n"
                 "1,,YES,1.0\n"
                 "2,,NO,2.0\n",
                 encoding="utf-8")

    times, labels = load_psychopy_log(p, {"YES": "yes", "NO": "no"})
    np.testing.assert_allclose(times, [1.0, 2.0])
    assert labels == ["yes", "no"]


def test_blank_time_rows_are_skipped(tmp_path):
    """中途异常退出的日志会有空行，不能让整份日志解析失败。"""
    p = tmp_path / "log.csv"
    p.write_text("trial,word,marker,word_onset_s\n"
                 "1,no,NO,1.0\n"
                 "2,no,NO,\n"
                 "3,no,NO,abc\n"
                 "4,yes,YES,4.0\n",
                 encoding="utf-8")

    times, labels = load_psychopy_log(p)
    np.testing.assert_allclose(times, [1.0, 4.0])
    assert labels == ["no", "yes"]


# ---------------------------------------------------------------------------
#  秒 → 采样点 换算（真实数据链路里最关键的一步）
# ---------------------------------------------------------------------------


def test_sample_index_conversion():
    """无偏移时，秒数 × 采样率即为采样点索引。"""
    times = [0.0, 1.0, 2.5]
    idx, codes, diag = events_from_log_times(
        times, ["yes", "no", "yes"], CLASSES, sfreq=256.0, n_times=2000)

    np.testing.assert_array_equal(idx, [0, 256, 640])
    np.testing.assert_array_equal(codes, [0, 2, 0])   # yes=0, no=2
    assert "采用 3 条" in diag[0]


def test_offset_is_actually_applied():
    """``onset_offset_s`` 必须真的加到事件位置上。

    这个偏移表示"脑电记录起点"比"实验程序起点"晚了多久。以前这条路径
    把它漏掉了，却在来源标签里写着已应用偏移 —— 报了却没做，比不报更糟。
    """
    times = [1.0, 2.0]
    labels = ["yes", "no"]

    base, _, _ = events_from_log_times(times, labels, CLASSES, 256.0, 10000)
    shifted, _, _ = events_from_log_times(times, labels, CLASSES, 256.0, 10000,
                                          onset_offset_s=0.5)

    np.testing.assert_array_equal(shifted - base, [128, 128])


def test_events_outside_recording_are_dropped():
    """超出记录范围的事件丢掉并计入诊断 —— 不能留负索引。

    负索引在分段时会从数据末尾绕回来，取到一段毫不相干的信号，
    得到的分段看着有数据、其实全是错的。
    """
    times = [-1.0, 0.5, 100.0]          # 头尾都在记录外
    idx, codes, diag = events_from_log_times(
        times, ["yes", "no", "yes"], CLASSES, sfreq=256.0, n_times=2560)

    np.testing.assert_array_equal(idx, [128])   # 只剩 0.5 s 那条
    np.testing.assert_array_equal(codes, [2])   # 它是 "no"，CLASSES 里排第 2
    assert any("落在记录范围外" in d and "2 条" in d for d in diag), diag
    assert any("onset_offset_s" in d for d in diag), \
        "范围外的报错要把人往偏移量上引，而不是让人去查 marker"


def test_unmapped_labels_are_reported():
    """日志里出现了 marker_map 没有的类别，要计数上报而不是默默丢掉。"""
    idx, codes, diag = events_from_log_times(
        ["1.0", "2.0"], ["yes", "maybe"], CLASSES, 256.0, 10000)

    assert len(idx) == 1
    assert any("类别未在 marker_map 中" in d and "1 条" in d for d in diag), diag


def test_wrong_offset_wipes_out_all_events():
    """偏移填得离谱时，一条事件都不剩 —— 诊断信息必须说清是这个原因。"""
    idx, codes, diag = events_from_log_times(
        [10.0, 20.0], ["yes", "no"], CLASSES, 256.0, n_times=512,
        onset_offset_s=-100.0)

    assert len(idx) == 0 and len(codes) == 0
    assert any("onset_offset_s" in d for d in diag), diag


def test_class_codes_follow_marker_map_order():
    """类别编码顺序 = marker_map 里首次出现的顺序，保证可复现。"""
    names = _class_names_from_map({"YES ": "yes", "OR  ": "or", "NO  ": "no"})
    assert names == ["yes", "or", "no"]

    _, codes, _ = events_from_log_times(
        [1.0], ["or"], names, 256.0, 10000)
    np.testing.assert_array_equal(codes, [1])
