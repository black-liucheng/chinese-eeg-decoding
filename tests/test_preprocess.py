# -*- coding: utf-8 -*-
"""预处理测试。

预处理是最容易"悄悄出错"的一环 —— 参数写反了照样跑得通，但结果全错。
所以这里不只测"能跑"，还测每条处理是否真的**做了它该做的事**。
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from cneeg.data.epoching import (apply_standardizer, bandpass_filter,
                                 baseline_correct, detect_bad_trials,
                                 fit_standardizer, notch_filter,
                                 preprocess_dataset, rereference)


# ---------------------------------------------------------------------------
#  基本性质
# ---------------------------------------------------------------------------


def test_preprocess_returns_new_dataset(raw_ds, small_cfg):
    """``preprocess_dataset`` 必须返回新对象、不改原对象。

    预处理页要靠"原对象还在"来做前后对比图 —— 一旦就地修改，
    对比图两张会一模一样，而且再改参数就是在已处理的数据上叠加处理。
    """
    before = raw_ds.X.copy()
    out, _ = preprocess_dataset(raw_ds, small_cfg)
    np.testing.assert_array_equal(raw_ds.X, before,
                                 err_msg="preprocess_dataset 修改了输入数据")
    assert out is not raw_ds


def test_output_shape_consistent(ds):
    assert ds.X.ndim == 3
    assert ds.X.shape[0] == len(ds.y)
    assert ds.X.shape[1] == len(ds.ch_names)
    assert np.isfinite(ds.X).all()


def test_baseline_is_removed(ds):
    """基线校正后，基线段均值应当接近 0。

    这里直接用**已处理**的数据反推：z-score 之后整体被缩放过，所以不能
    断言"绝对值=0"，但可以断言基线窗的均值远小于整段的标准差。
    """
    base = (ds.times >= ds.tmin) & (ds.times < 0)
    if not base.any():
        pytest.skip("该配置没有基线段")
    m = np.abs(ds.X[:, :, base].mean(axis=2)).mean()
    s = ds.X.std()
    assert m < 0.35 * s, f"基线均值 {m:.3f} 相对整体标准差 {s:.3f} 偏大，基线没扣干净"


def test_zscore_normalizes(raw_ds, small_cfg):
    """逐通道 z-score 之后，每个通道整体应近似零均值、单位方差。"""
    c = copy.deepcopy(small_cfg)
    c.preprocess.normalize = "zscore"
    c.preprocess.reject_uv = 0.0            # 别让剔除干扰统计
    out, _ = preprocess_dataset(raw_ds, c)
    means = out.X.mean(axis=(0, 2))
    stds = out.X.std(axis=(0, 2))
    assert np.abs(means).max() < 0.05, f"通道均值没归零：{np.abs(means).max():.4f}"
    assert np.abs(stds - 1.0).max() < 0.05, "通道方差没归一到 1"


def test_normalize_none_keeps_scale(raw_ds, small_cfg):
    """关掉标准化时，幅值量级应当仍是 µV 级 —— 不该被悄悄改掉。"""
    c = copy.deepcopy(small_cfg)
    c.preprocess.normalize = "none"
    c.preprocess.reject_uv = 0.0
    out, _ = preprocess_dataset(raw_ds, c)
    rms = float(np.sqrt(np.mean(out.X ** 2)))
    assert 0.5 < rms < 500.0


# ---------------------------------------------------------------------------
#  滤波
# ---------------------------------------------------------------------------


def test_bandpass_removes_dc_and_high_freq():
    """带通滤波要同时干掉直流漂移和远高于截止频率的成分。"""
    sfreq, n = 256.0, 2560
    t = np.arange(n) / sfreq
    x = (np.ones_like(t) * 50.0            # 直流
         + 10.0 * np.sin(2 * np.pi * 10 * t)   # 10 Hz，该保留
         + 8.0 * np.sin(2 * np.pi * 100 * t))  # 100 Hz，该滤掉

    y = bandpass_filter(x, sfreq, 0.5, 40.0)

    def amp(sig, f):
        spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
        i = int(round(f * len(sig) / sfreq))
        return spec[i]

    assert amp(y, 10.0) > 5 * amp(y, 100.0), "100 Hz 没被低通滤掉"
    assert abs(y.mean()) < 1.0, "直流漂移没被高通去掉"


def test_bandpass_preserves_length():
    x = np.random.default_rng(0).standard_normal((4, 500))
    assert bandpass_filter(x, 256.0, 0.5, 40.0).shape == x.shape


def test_notch_removes_line_noise():
    """50 Hz 陷波要显著压低 50 Hz 分量，同时基本不动 10 Hz。"""
    sfreq, n = 256.0, 2560
    t = np.arange(n) / sfreq
    clean = 10.0 * np.sin(2 * np.pi * 10 * t)
    noisy = clean + 20.0 * np.sin(2 * np.pi * 50 * t)

    y = notch_filter(noisy, sfreq, 50.0)

    def amp(sig, f):
        spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
        return spec[int(round(f * len(sig) / sfreq))]

    assert amp(y, 50.0) < 0.3 * amp(noisy, 50.0), "50 Hz 没被压下去"
    assert amp(y, 10.0) > 0.5 * amp(clean, 10.0), "陷波误伤了 10 Hz"


def test_notch_zero_is_noop():
    """陷波频率填 0（关闭）时必须原样返回，不能崩。"""
    x = np.random.default_rng(1).standard_normal((2, 300))
    out = notch_filter(x, 256.0, 0.0)
    np.testing.assert_allclose(out, x, atol=1e-9)


# ---------------------------------------------------------------------------
#  重参考 / 基线
# ---------------------------------------------------------------------------


def test_car_makes_channels_sum_to_zero():
    """共同平均参考（CAR）：减去全通道均值后，各通道在每一时刻之和应为 0。

    容差放到 1e-4：内部按 float32 算，量级 10 µV 时 float32 的精度
    本来就只有 ~1e-6，卡到 1e-8 是在考验浮点而不是考验代码。
    """
    x = np.random.default_rng(2).standard_normal((8, 200)) * 10
    y = rereference(x, "car")
    np.testing.assert_allclose(y.sum(axis=0), 0.0, atol=1e-4)


def test_reref_none_is_noop():
    x = np.random.default_rng(3).standard_normal((8, 200))
    np.testing.assert_allclose(rereference(x, "none"), x, atol=1e-12)


def test_reref_rejects_unknown_mode():
    """不支持的方式要明确报错，不能悄悄返回原数据。

    悄悄返回的后果：界面上选了"平均参考"、日志里也写着做了，实际什么都没做。
    """
    x = np.random.default_rng(31).standard_normal((8, 200))
    with pytest.raises(ValueError, match="重参考"):
        rereference(x, "avg")


def test_baseline_correct_zeroes_window():
    x = np.random.default_rng(4).standard_normal((5, 4, 100)) * 10 + 7.0
    times = np.linspace(-0.2, 0.8, 100)
    y = baseline_correct(x, times, (-0.2, 0.0))
    base = (times >= -0.2) & (times <= 0.0)
    np.testing.assert_allclose(y[:, :, base].mean(axis=2), 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
#  伪迹剔除
# ---------------------------------------------------------------------------


def test_detect_bad_trials_flags_spikes():
    """峰峰值超限的试次必须被抓出来，正常试次不能被误杀。

    注意返回的是**保留掩码**（``True`` = 保留），不是"坏试次掩码" ——
    这个方向搞反的话会正好把好数据全丢掉、只留伪迹。
    """
    x = np.random.default_rng(5).standard_normal((6, 4, 100)) * 2
    x[3, 1, 50] = 900.0                    # 第 3 个试次埋一个尖峰

    keep = detect_bad_trials(x, reject_uv=150.0)
    assert not keep[3], "超限试次没被检出"
    assert keep[[0, 1, 2, 4, 5]].all(), "正常试次被误判为伪迹"


def test_detect_bad_trials_disabled_keeps_everything():
    x = np.random.default_rng(6).standard_normal((5, 4, 100)) * 1e4
    assert detect_bad_trials(x, reject_uv=0.0).all()


def test_reject_disabled_keeps_all(raw_ds, small_cfg):
    c = copy.deepcopy(small_cfg)
    c.preprocess.reject_uv = 0.0
    out, rep = preprocess_dataset(raw_ds, c)
    assert out.n_trials == raw_ds.n_trials
    assert rep.n_rejected == 0


def test_reject_all_trials_raises(raw_ds, small_cfg):
    """阈值定得不合理（把所有试次都剔光）时要明确报错。

    报错好过返回一个空数据集 —— 后者会在后面某处引发难以定位的
    "shape (0,) 无法计算"错误，让人完全想不到是预处理参数的问题。
    """
    c = copy.deepcopy(small_cfg)
    c.preprocess.reject_uv = 1e-6
    with pytest.raises(ValueError, match="剔除|试次"):
        preprocess_dataset(raw_ds, c)


def test_report_counts_are_consistent(ds, raw_ds, small_cfg):
    """报告里的"输入/剔除/保留"三个数必须自洽。"""
    _, rep = preprocess_dataset(raw_ds, small_cfg)
    assert rep.n_trials_in == raw_ds.n_trials
    assert rep.n_trials_out == ds.n_trials
    assert rep.n_rejected == rep.n_trials_in - rep.n_trials_out
    assert 0.0 <= rep.reject_ratio <= 1.0


# ---------------------------------------------------------------------------
#  标准化的 fit/apply 分离
# ---------------------------------------------------------------------------


def test_standardizer_train_test_consistency():
    """标准化器必须能"在训练集上 fit、在测试集上 apply"。

    这是防数据泄漏的关键：如果每批数据各自标准化，测试集的信息就漏进了
    训练流程，交叉验证的分数会虚高。
    """
    rng = np.random.default_rng(7)
    train = rng.standard_normal((20, 4, 50)) * 5 + 3
    test = rng.standard_normal((5, 4, 50)) * 5 + 3

    mu, sd = fit_standardizer(train)
    a = apply_standardizer(train, mu, sd)
    b = apply_standardizer(test, mu, sd)

    assert abs(a.mean()) < 0.05
    # 测试集用的是训练集的均值方差 → 不该被归一到恰好零均值
    assert b.shape == test.shape
    # 同一份标准化器作用两次，结果必须一致（无隐藏随机性）
    np.testing.assert_allclose(apply_standardizer(test, mu, sd), b)
    # 关键：用的必须是**训练集**的统计量，不是测试集自己的
    assert not np.allclose(b.mean(axis=(0, 2)), 0.0, atol=1e-3) or \
        np.allclose(test.mean(axis=(0, 2)), 0.0, atol=1e-2)


def test_window_config_respected(raw_ds, small_cfg):
    """配置里的时间窗必须真的被用上，而不是被忽略。

    这是界面上"窗口起点/终点"能不能生效的依据。以前 preprocess_dataset
    不裁剪，用户改了窗口却什么都没发生，且没有任何提示。
    """
    c = copy.deepcopy(small_cfg)
    c.preprocess.tmin = -0.1
    c.preprocess.tmax = 0.5
    out, _ = preprocess_dataset(raw_ds, c)
    # 窗口只能落在采样点上，所以容差取一个采样间隔 —— 要求精确到 -0.1
    # 等于要求亚采样点精度，那是做不到的。
    step = 1.0 / out.sfreq
    assert out.tmin == pytest.approx(-0.1, abs=step)
    assert out.times[-1] == pytest.approx(0.5, abs=step)
    assert out.tmin >= -0.1 - 1e-9, "起点不该早于配置值"
    assert out.times[-1] <= 0.5 + 1e-9, "终点不该晚于配置值"
    assert out.n_times < raw_ds.n_times, "窗口变窄了，采样点数却没减少"


def test_window_crop_keeps_baseline_correct(raw_ds, small_cfg):
    """裁剪窗口后基线窗仍在数据里时，基线校正要作用于**裁剪后**的窗口。"""
    c = copy.deepcopy(small_cfg)
    c.preprocess.tmin = -0.15
    c.preprocess.tmax = 0.6
    out, _ = preprocess_dataset(raw_ds, c)
    base = (out.times >= -0.15) & (out.times <= 0.0)
    assert base.any(), "裁剪后基线窗不该为空"
    m = np.abs(out.X[:, :, base].mean(axis=2)).mean()
    assert m < 0.35 * out.X.std(), "裁剪窗口后基线没扣干净"
