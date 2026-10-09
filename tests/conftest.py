# -*- coding: utf-8 -*-
"""pytest 公共夹具。

原则：测试全部用小规模数据（几受试者 × 几试次、少 epoch），保证
``pytest tests/`` 在一两分钟内跑完。跑得慢的测试没人愿意跑，最后就没人跑了。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# 让 tests/ 直接 import src/cneeg，不需要装包
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cneeg.config import load_config                       # noqa: E402
from cneeg.data.synthetic import generate_dataset          # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    """默认配置的一份副本 —— 测试会改它，所以每个测试自己 deepcopy。"""
    return load_config()


@pytest.fixture
def small_cfg(cfg):
    """小而快的配置：2 受试者 × 8 试次/类，2 折，不跑置换。"""
    import copy

    c = copy.deepcopy(cfg)
    c.data.synthetic.n_subjects = 2
    c.data.synthetic.trials_per_class = 8
    c.data.synthetic.snr_db = 6.0        # 高信噪比 → 小数据也能学到东西
    c.train.n_splits = 2
    c.train.permutation_test = False
    c.train.n_permutations = 3
    c.model.cnn.n_epochs = 3
    c.model.cnn.batch_size = 8
    c.model.cnn.early_stop_patience = 2
    return c


@pytest.fixture
def raw_ds(small_cfg):
    """未预处理的仿真数据集。"""
    return generate_dataset(small_cfg)


@pytest.fixture
def ds(raw_ds, small_cfg):
    """预处理后的数据集。"""
    from cneeg.data.epoching import preprocess_dataset

    out, _ = preprocess_dataset(raw_ds, small_cfg)
    return out


@pytest.fixture
def rng():
    return np.random.default_rng(12345)
