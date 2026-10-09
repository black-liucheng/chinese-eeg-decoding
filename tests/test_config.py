# -*- coding: utf-8 -*-
"""配置加载与校验测试。

配置错了要在**启动时**就报错，而不是跑完 20 分钟训练之后才发现
"哦，n_splits 写成了字符串"。这个文件就是守这条线的。
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from cneeg.config import (CLASS_LABELS, DEFAULT_CONFIG_PATH, PRESET_LABELS,
                          PRESETS, dump_config, load_config)


# ---------------------------------------------------------------------------
#  加载
# ---------------------------------------------------------------------------


def test_default_config_loads():
    cfg = load_config()
    assert cfg.seed > 0
    assert cfg.data.source in ("synthetic", "file", "real", "mff")
    assert cfg.preprocess.l_freq < cfg.preprocess.h_freq
    assert cfg.train.n_splits >= 2


def test_shipped_yaml_matches_defaults():
    """随包发布的 default.yaml 必须能被解析 —— 它坏了整个程序就起不来。"""
    assert DEFAULT_CONFIG_PATH.exists(), f"找不到 {DEFAULT_CONFIG_PATH}"
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert isinstance(cfg.seed, int)


def test_dotted_overrides():
    """``--set a.b.c=v`` 用的点号路径覆盖。"""
    cfg = load_config(overrides={"model.cnn.n_epochs": 7,
                                 "data.synthetic.n_subjects": 3,
                                 "seed": 42})
    assert cfg.model.cnn.n_epochs == 7
    assert cfg.data.synthetic.n_subjects == 3
    assert cfg.seed == 42


def test_overrides_do_not_touch_other_fields():
    base = load_config()
    cfg = load_config(overrides={"model.cnn.lr": 0.05})
    assert cfg.model.cnn.lr == 0.05
    # 其余字段保持原样（防止覆盖逻辑把整棵子树替换掉）
    assert cfg.model.name == base.model.name
    assert cfg.model.svm.C == base.model.svm.C
    assert cfg.data.synthetic.preset == base.data.synthetic.preset


def test_missing_config_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("does_not_exist_anywhere.yaml")


def test_unknown_key_raises():
    """写错键名要报错，不能默默忽略。

    默默忽略的后果：用户改的是 model.cnn.n_epoch，程序照跑，
    用的是默认值 —— 他会以为"这个参数没用"。
    """
    with pytest.raises(Exception) as ei:
        load_config(overrides={"model.cnn.n_epoch": 5})
    assert "n_epoch" in str(ei.value)


# ---------------------------------------------------------------------------
#  校验
# ---------------------------------------------------------------------------


def test_invalid_bandpass_rejected():
    """高通必须低于低通 —— 写反了要当场拦住。"""
    with pytest.raises(Exception):
        load_config(overrides={"preprocess.l_freq": 45.0,
                               "preprocess.h_freq": 30.0})


def test_invalid_preset_rejected():
    with pytest.raises(Exception):
        load_config(overrides={"data.synthetic.preset": "no_such_preset"})


def test_invalid_window_rejected():
    with pytest.raises(Exception):
        load_config(overrides={"preprocess.tmin": 0.5,
                               "preprocess.tmax": 0.2})


def test_invalid_model_name_rejected():
    with pytest.raises(Exception):
        load_config(overrides={"model.name": "transformer"})


def test_invalid_reref_rejected():
    with pytest.raises(Exception):
        load_config(overrides={"preprocess.reref": "avg"})


def test_valid_configs_accepted():
    """合法取值不该被误拦 —— 校验写太严会把能用的配置也挡掉。"""
    cfg = load_config(overrides={
        "preprocess.l_freq": 1.0, "preprocess.h_freq": 30.0,
        "preprocess.reref": "none", "preprocess.normalize": "none",
        "preprocess.notch_freq": 0.0, "preprocess.reject_uv": 0.0,
        "model.name": "cnn", "train.n_splits": 3,
    })
    assert cfg.preprocess.reref == "none"
    assert cfg.model.name == "cnn"


# ---------------------------------------------------------------------------
#  预设与标签
# ---------------------------------------------------------------------------


def test_presets_and_labels_agree():
    """每个预设都要有中文标签，且类名都要在 CLASS_LABELS 里有对应 ——
    否则图上和表里会出现内部代号（如 a1），答辩时很难看。"""
    assert set(PRESETS) == set(PRESET_LABELS)
    for name, classes in PRESETS.items():
        assert len(classes) >= 2, f"预设 {name} 至少要有 2 类"
        for c in classes:
            assert c in CLASS_LABELS, f"预设 {name} 的类别 {c} 没有中文标签"


def test_english_preset_matches_psychopy_paradigm():
    """english_word 预设要能对上现有 PsychoPy 范式里打的 mark。

    ``word_reading_eeg.py`` 里打的是 'YES ' / 'OR  ' / 'NO  '，
    所以这里的类名必须落在同样的词上，否则真数据接进来时对不上号。
    """
    assert set(PRESETS["english_word"]) == {"yes", "or", "no"}


# ---------------------------------------------------------------------------
#  落盘
# ---------------------------------------------------------------------------


def test_dump_and_reload_roundtrip(tmp_path):
    """配置存下来再读回来，关键参数必须一致 —— 实验可复现的前提。"""
    cfg = load_config(overrides={"model.cnn.n_epochs": 11, "seed": 999})
    out = dump_config(cfg, tmp_path / "used.yaml")
    assert out.exists()

    again = load_config(out)
    assert again.seed == 999
    assert again.model.cnn.n_epochs == 11
    assert again.data.synthetic.preset == cfg.data.synthetic.preset


def test_sub_dir_creates_directory(tmp_path):
    cfg = load_config()
    cfg.output.data_dir = str(tmp_path / "data")
    p = cfg.sub_dir("models", "deep")
    assert p.is_dir()
