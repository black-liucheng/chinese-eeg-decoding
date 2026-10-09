# -*- coding: utf-8 -*-
"""仿真数据生成器测试。

重点不是"跑得通"，而是**数据本身有没有它该有的性质** —— 形状、类别可分性、
可复现性、信噪比单调性。这些性质错了，后面所有模型结论都是假的。
"""

from __future__ import annotations

import numpy as np
import pytest

from cneeg.config import PRESETS
from cneeg.data.synthetic import (BACKGROUND_RMS_UV, generate_dataset,
                                  snr_to_scale)


# ---------------------------------------------------------------------------
#  形状与元信息
# ---------------------------------------------------------------------------


def test_shape_and_metadata(raw_ds, small_cfg):
    assert raw_ds.X.ndim == 3
    assert raw_ds.n_channels == small_cfg.data.synthetic.n_channels
    assert raw_ds.n_classes == len(PRESETS[small_cfg.data.synthetic.preset])
    assert raw_ds.n_trials == (small_cfg.data.synthetic.n_subjects
                               * small_cfg.data.synthetic.trials_per_class
                               * raw_ds.n_classes)
    assert len(raw_ds.ch_names) == raw_ds.n_channels
    assert raw_ds.is_synthetic is True
    # 时间窗：tmin 到 tmin + (n_times-1)/sfreq
    assert raw_ds.tmin < 0, "仿真数据必须含基线段（tmin 为负）"
    assert raw_ds.times[-1] > 0.5, "刺激后至少要有 500 ms"


def test_labels_balanced(raw_ds):
    """每类试次数应当大体均衡 —— 严重不均衡会让准确率这个指标失去意义。"""
    counts = np.bincount(raw_ds.y, minlength=raw_ds.n_classes)
    assert counts.min() > 0
    assert counts.max() / counts.min() < 1.5


def test_subject_ids_present(raw_ds, small_cfg):
    """跨受试者评估依赖 subject 字段，缺了就没法做留一受试者。"""
    assert raw_ds.subject is not None
    assert len(np.unique(raw_ds.subject)) == small_cfg.data.synthetic.n_subjects


def test_finite_and_plausible_amplitude(raw_ds):
    """不该有 NaN/Inf；幅值量级要像脑电（µV 级，不是 1e6 也不是 1e-9）。"""
    assert np.isfinite(raw_ds.X).all()
    rms = float(np.sqrt(np.mean(raw_ds.X ** 2)))
    assert 1.0 < rms < 500.0, f"背景 RMS={rms:.2f} µV，不像脑电"


# ---------------------------------------------------------------------------
#  可复现性
# ---------------------------------------------------------------------------


def test_same_seed_identical(small_cfg):
    a = generate_dataset(small_cfg)
    b = generate_dataset(small_cfg)
    np.testing.assert_array_equal(a.X, b.X)
    np.testing.assert_array_equal(a.y, b.y)


def test_different_seed_differs(small_cfg):
    import copy

    a = generate_dataset(small_cfg)
    c = copy.deepcopy(small_cfg)
    c.seed = small_cfg.seed + 1
    b = generate_dataset(c)
    assert not np.array_equal(a.X, b.X)


# ---------------------------------------------------------------------------
#  类别可分性 —— 这是整个项目的前提
# ---------------------------------------------------------------------------


def test_classes_are_separable(raw_ds):
    """各类别的平均 ERP 之间必须有实打实的差异。

    做法：用"类间方差 / 类内方差"衡量。若这个比值接近 1，说明各类别的响应
    没有区别 —— 那模型学不到任何东西，后面所有实验都白做，必须在这里拦住。
    """
    grand = raw_ds.X.mean(axis=2)                     # (试次, 电极)
    f_values = []
    for ch in range(raw_ds.n_channels):
        groups = [grand[raw_ds.y == k, ch] for k in range(raw_ds.n_classes)]
        between = np.var([g.mean() for g in groups]) * len(groups)
        within = float(np.mean([g.var() for g in groups]))
        if within > 1e-12:
            f_values.append(between / within)

    assert f_values, "算不出可分性 —— 数据可能全是常数"
    best = max(f_values)
    assert best > 3.0, (
        f"最好的电极 F 值只有 {best:.2f}，各类别几乎不可分。"
        "仿真生成器的类别模板可能没起作用。")


def test_snr_monotonic_separability(cfg):
    """信噪比越高，类别越可分 —— 这条单调性不成立，SNR 参数就是假的。"""
    import copy

    scores = []
    for snr in (-6.0, 12.0):
        c = copy.deepcopy(cfg)
        c.data.synthetic.n_subjects = 1
        c.data.synthetic.trials_per_class = 12
        c.data.synthetic.snr_db = snr
        d = generate_dataset(c)
        grand = d.X.mean(axis=2)
        f = []
        for ch in range(d.n_channels):
            groups = [grand[d.y == k, ch] for k in range(d.n_classes)]
            between = np.var([g.mean() for g in groups]) * len(groups)
            within = float(np.mean([g.var() for g in groups]))
            if within > 1e-12:
                f.append(between / within)
        scores.append(max(f))

    assert scores[1] > scores[0] * 2, (
        f"SNR 12 dB 的可分性（{scores[1]:.2f}）没有明显高于 -6 dB"
        f"（{scores[0]:.2f}）—— SNR 参数可能没接进模型")


def test_snr_to_scale():
    """SNR→幅度系数的换算关系。

    注意返回的是**绝对幅度**（µV），不是相对倍数：
    0 dB 表示 ERP 峰值 = 背景 RMS（约 17.5 µV），不是"1 倍系数"。
    """
    assert snr_to_scale(0.0) == pytest.approx(BACKGROUND_RMS_UV, rel=1e-9)
    # 严格按定义：幅度 = 背景 RMS × 10^(dB/20)。
    # （注意"每 6 dB 翻倍"只是近似 —— 精确翻倍在 20·lg2 ≈ 6.0206 dB，
    #  所以这里断言的是公式本身，不是那条经验口诀。）
    for db in (-20.0, -12.0, -6.0, 0.0, 4.0, 6.0, 12.0):
        assert snr_to_scale(db) == pytest.approx(
            BACKGROUND_RMS_UV * 10.0 ** (db / 20.0), rel=1e-12)
    # 差值换算：任意两点之比只取决于 SNR 之差
    assert snr_to_scale(6.0) / snr_to_scale(0.0) == pytest.approx(
        snr_to_scale(12.0) / snr_to_scale(6.0), rel=1e-12)
    # 单调递增
    xs = [-12, -6, 0, 6, 12]
    ys = [snr_to_scale(x) for x in xs]
    assert all(b > a for a, b in zip(ys, ys[1:]))


# ---------------------------------------------------------------------------
#  伪迹
# ---------------------------------------------------------------------------


def test_artifacts_are_injected(cfg):
    """伪迹概率调满时，应当产生远超正常水平的极端幅度。

    这条测试防的是"伪迹代码写了但没被调用"这类静默失效 —— 那样预处理里的
    剔除逻辑就永远测不到东西。
    """
    import copy

    clean = copy.deepcopy(cfg)
    clean.data.synthetic.n_subjects = 1
    clean.data.synthetic.trials_per_class = 10
    clean.data.synthetic.blink_prob = 0.0
    clean.data.synthetic.muscle_prob = 0.0
    clean.data.synthetic.dropout_prob = 0.0

    dirty = copy.deepcopy(clean)
    dirty.data.synthetic.blink_prob = 0.9
    dirty.data.synthetic.muscle_prob = 0.9

    a = np.abs(generate_dataset(clean).X).max()
    b = np.abs(generate_dataset(dirty).X).max()
    assert b > a, "把伪迹概率调到 0.9 却没有让幅度变得更大 —— 伪迹没被注入"


def test_presets_all_generate(cfg):
    """三个预设都要能生成 —— 答辩现场切预设不能崩。"""
    import copy

    for preset in PRESETS:
        c = copy.deepcopy(cfg)
        c.data.synthetic.preset = preset
        c.data.synthetic.n_subjects = 1
        c.data.synthetic.trials_per_class = 3
        d = generate_dataset(c)
        assert d.n_classes == len(PRESETS[preset])
        assert np.isfinite(d.X).all()
