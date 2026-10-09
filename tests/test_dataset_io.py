# -*- coding: utf-8 -*-
"""数据集落盘 / 读取 / 查重。

``--data`` 参数复用缓存、Gen-data 跳过重复文件，都建立在这套读写之上。
以前这块**一个测试都没有** —— 而它一旦坏掉，症状是"读回来的数据静默少了
某些字段"，比如受试者编号丢了，留一受试者评估就会悄悄退化成普通交叉验证。
"""

from __future__ import annotations

import numpy as np
import pytest

from cneeg.data.base import EEGDataset


def _ds(n_trials: int = 6, n_ch: int = 4, n_t: int = 32, seed: int = 0):
    rng = np.random.default_rng(seed)
    return EEGDataset(
        X=rng.standard_normal((n_trials, n_ch, n_t)).astype(np.float32),
        y=np.array([i % 2 for i in range(n_trials)], dtype=np.int64),
        class_names=["a", "b"],
        sfreq=256.0,
        tmin=-0.2,
        ch_names=[f"ch{i}" for i in range(n_ch)],
        subject=np.array([i // 3 for i in range(n_trials)]) if n_trials >= 6 else None,
        is_synthetic=True,
        meta={"note": "测试"},
    )


def test_save_load_roundtrip_keeps_every_field(tmp_path):
    """往返之后所有字段都要还在，尤其是 subject。
    受试者编号丢了，留一受试者评估会静默退化成普通交叉验证。
    """
    ds = _ds()
    p = ds.save(tmp_path / "d.npz")
    got = EEGDataset.load(p)

    np.testing.assert_array_equal(got.X, ds.X)
    np.testing.assert_array_equal(got.y, ds.y)
    assert got.class_names == ds.class_names
    assert got.ch_names == ds.ch_names
    assert got.sfreq == ds.sfreq and got.tmin == ds.tmin
    assert got.is_synthetic is True
    np.testing.assert_array_equal(got.subject, ds.subject)
    assert got.n_trials == ds.n_trials


def test_load_without_subject_returns_none(tmp_path):
    """没存受试者时读回来必须是 None，不能变成空数组 —— 空数组会让
    ``group_by_subject`` 走进去又算不出组，报错位置离原因很远。
    """
    ds = EEGDataset(
        X=np.zeros((4, 3, 16), dtype=np.float32),
        y=np.array([0, 1, 0, 1], dtype=np.int64),
        class_names=["a", "b"], sfreq=256.0, tmin=-0.2,
        ch_names=["c0", "c1", "c2"], subject=None, is_synthetic=True,
    )
    got = EEGDataset.load(ds.save(tmp_path / "nosubj.npz"))
    assert got.subject is None


def test_fingerprint_is_content_addressed(tmp_path):
    """内容相同的两份数据指纹必须一样，内容变了指纹必须变。"""
    a, b = _ds(seed=1), _ds(seed=1)
    assert a.fingerprint() == b.fingerprint()

    c = _ds(seed=2)
    assert a.fingerprint() != c.fingerprint()


def test_fingerprint_ignores_meta_but_tracks_reference(tmp_path):
    """指纹只看参与建模的内容：只改 meta 不算变，改了时间窗就算变。"""
    a = _ds(seed=3)
    b = _ds(seed=3)
    b.meta = {"note": "换了个说明"}
    assert a.fingerprint() == b.fingerprint(), "meta 不影响建模，不该改变指纹"

    c = _ds(seed=3)
    c.tmin = -0.5
    assert a.fingerprint() != c.fingerprint(), "时间窗变了就是另一份数据"


def test_stored_fingerprint_reads_back(tmp_path):
    ds = _ds(seed=4)
    p = ds.save(tmp_path / "fp.npz")
    assert EEGDataset.stored_fingerprint(p) == ds.fingerprint()


def test_stored_fingerprint_tolerates_old_files(tmp_path):
    """没有 fingerprint 字段的老文件返回空串，不能抛异常 ——
    用户手里早就存好的 npz 不该因为升级而读不了。
    """
    p = tmp_path / "old.npz"
    np.savez_compressed(p, X=np.zeros((2, 2, 8), dtype=np.float32),
                        y=np.array([0, 1]))
    assert EEGDataset.stored_fingerprint(p) == ""
    assert EEGDataset.stored_fingerprint(tmp_path / "不存在.npz") == ""


# ---------------------------------------------------------------------------
#  随机基线（chance_level）
# ---------------------------------------------------------------------------


def _with_labels(y, class_names=None):
    y = np.asarray(y, dtype=np.int64)
    ds = _ds(n_trials=len(y))
    ds.y = y
    # 空标签时 y.max() 会抛异常，所以允许显式给出类别名。
    ds.class_names = class_names or [f"c{i}" for i in range(int(y.max()) + 1)]
    return ds


def test_chance_level_is_majority_share_when_imbalanced():
    """类别不均衡时，随机基线是**多数类占比**，不是 1/类别数。

    这不是吹毛求疵：伪迹剔除按试次独立触发，剔完各类别剩多少并不相等，
    所以不均衡是常态。用 1/类别数会**低估**基线 —— 一个把所有试次都判成
    多数类的模型什么都没学到，却会显得"高于随机"。
    """
    ds = _with_labels([0] * 6 + [1] * 3 + [2] * 3)      # 6/12 = 50% 是多数类
    assert ds.chance_level == pytest.approx(0.5)
    assert ds.chance_level > 1.0 / ds.n_classes, "不均衡时基线必须高于 1/类别数"
    assert 1.0 / ds.n_classes == pytest.approx(1 / 3)


def test_chance_level_equals_one_over_k_when_balanced():
    """均衡时两者完全相等 —— 所以"始终 25%"这种说法在均衡数据上并没有错，
    文档里那句 25% 仍然成立。这条测试把这个等价关系钉住。
    """
    ds = _with_labels([0, 1, 2, 3] * 5)
    assert ds.chance_level == pytest.approx(0.25)


def test_chance_level_on_empty_labels_does_not_crash():
    """空数据集（比如全被伪迹剔除）不该在这里抛异常。"""
    ds = _with_labels(np.zeros(0, dtype=np.int64),
                      class_names=["a", "b", "c", "d"])
    assert ds.chance_level == pytest.approx(0.25)


def test_dedup_hits_legacy_file_without_fingerprint(small_cfg, tmp_path):
    """老数据集文件（早于 fingerprint 字段）也必须能被查重命中。

    这类文件里没有内嵌指纹，``stored_fingerprint`` 返回空串，而空串永远不等于
    真指纹 —— 查重于是**静默失效**，每次 gen-data 都再写一份一模一样的 47 MB。
    项目早期生成的文件就是这个格式（磁盘上因此躺了两份）。
    """
    import copy

    from cneeg.cli import _save_dataset
    from cneeg.data.epoching import preprocess_dataset
    from cneeg.data.synthetic import generate_dataset

    c = copy.deepcopy(small_cfg)
    c.output.data_dir = str(tmp_path)
    ds, _ = preprocess_dataset(generate_dataset(c), c)

    # 手工写一个"老格式"文件：数据齐全，但**没有** fingerprint 字段
    legacy = tmp_path / "processed" / "dataset_20200101_000000.npz"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        legacy,
        X=ds.X, y=ds.y,
        class_names=np.array(ds.class_names, dtype=object),
        sfreq=ds.sfreq, tmin=ds.tmin,
        ch_names=np.array(ds.ch_names, dtype=object),
        subject=np.array([]), session=np.array([]),
        is_synthetic=True, meta=np.array([{}], dtype=object),
    )
    assert EEGDataset.stored_fingerprint(legacy) == "", "前提：老文件应当没有指纹字段"

    out = _save_dataset(ds, c, "dataset")
    assert out == legacy, "内容相同的老文件应当被复用，而不是再写一份新的"
    assert len(list((tmp_path / "processed").glob("dataset_*.npz"))) == 1
