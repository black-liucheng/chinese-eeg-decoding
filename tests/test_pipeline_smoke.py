# -*- coding: utf-8 -*-
"""端到端冒烟测试 —— 小规模但跑**完整**流水线。

单测各自通过、拼起来却崩掉，是这类项目最典型的事故。这个文件专门防它：
生成 → 预处理 → 特征 → 训练 → 交叉验证 → 评估 → 画图 → 报告，
每一环都真的走一遍。

规模压得很小（2 受试者 × 8 试次/类、CNN 只跑 3 轮），整体控制在一两分钟内。
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from cneeg.data.epoching import preprocess_dataset
from cneeg.data.synthetic import generate_dataset
from cneeg.features import extract_features
from cneeg.models import build_model
from cneeg.train import (comparison_table, cross_validate, holdout_evaluate,
                         permutation_test)


# ---------------------------------------------------------------------------
#  特征
# ---------------------------------------------------------------------------


def test_features_shape_and_finite(ds, small_cfg):
    X, names = extract_features(ds.X, ds.sfreq, small_cfg)
    assert X.ndim == 2
    assert X.shape[0] == ds.n_trials
    assert X.shape[1] > 0
    assert len(names) == X.shape[1], "特征名数量与特征列数对不上"
    assert np.isfinite(X).all()


def test_features_deterministic(ds, small_cfg):
    """同样的输入必须给出同样的特征 —— 否则交叉验证结果无法复现。"""
    a, _ = extract_features(ds.X, ds.sfreq, small_cfg)
    b, _ = extract_features(ds.X, ds.sfreq, small_cfg)
    np.testing.assert_allclose(a, b)


def test_unknown_feature_method_raises(ds, small_cfg):
    c = copy.deepcopy(small_cfg)
    c.features.method = "wavelet"
    with pytest.raises(ValueError, match="特征方法"):
        extract_features(ds.X, ds.sfreq, c)


# ---------------------------------------------------------------------------
#  单个模型能 fit / predict
# ---------------------------------------------------------------------------


def test_svm_fits_and_predicts(ds, small_cfg):
    c = copy.deepcopy(small_cfg)
    c.model.name = "svm"
    m = build_model(c, n_channels=ds.n_channels, n_times=ds.n_times,
                    n_classes=ds.n_classes, sfreq=ds.sfreq)
    m.fit(ds.X, ds.y)
    pred = m.predict(ds.X)
    assert pred.shape == ds.y.shape
    assert set(np.unique(pred)) <= set(range(ds.n_classes))
    proba = m.predict_proba(ds.X)
    np.testing.assert_allclose(np.asarray(proba).sum(axis=1), 1.0, atol=1e-5)


def test_cnn_fits_and_predicts(ds, small_cfg):
    c = copy.deepcopy(small_cfg)
    c.model.name = "cnn"
    c.model.cnn.n_epochs = 2
    m = build_model(c, n_channels=ds.n_channels, n_times=ds.n_times,
                    n_classes=ds.n_classes, sfreq=ds.sfreq)
    m.fit(ds.X, ds.y)
    pred = m.predict(ds.X)
    assert pred.shape == ds.y.shape
    assert m.history_ and len(m.history_["epoch"]) > 0, "训练没产生历史记录"


# ---------------------------------------------------------------------------
#  交叉验证
# ---------------------------------------------------------------------------


def test_cross_validate_svm_result_complete(ds, small_cfg):
    c = copy.deepcopy(small_cfg)
    c.model.name = "svm"
    res = cross_validate(c, ds, model_name="svm")

    assert len(res.folds) == c.train.n_splits
    assert 0.0 <= res.accuracy <= 1.0
    assert 0.0 <= res.macro_f1 <= 1.0
    assert -1.0 <= res.kappa <= 1.0
    # 随机基线是**多数类占比**，不是 1/类别数 —— 伪迹剔除按试次独立触发，
    # 剔完之后各类别剩下多少并不相等（这个 fixture 就不相等，所以这里
    # 刻意不去断言 1/4=0.25，那会把不均衡数据上的错误基线又钉回去）。
    counts = np.bincount(ds.y, minlength=ds.n_classes)
    assert res.chance_level == pytest.approx(counts.max() / len(ds.y))
    assert res.confusion.shape == (ds.n_classes, ds.n_classes)
    assert res.confusion.sum() == len(ds.y), "混淆矩阵总数应等于试次数"
    assert len(res.y_true) == len(ds.y) == len(res.y_pred)
    assert set(res.per_class) == set(ds.class_names)
    assert res.summary().strip(), "summary() 不该是空字符串"


def test_cnn_learns_above_chance(cfg):
    """CNN 在**够用规模 + 高信噪比**的数据上必须明显高于随机。

    这是整条流水线的核心断言：如果连简单可控的仿真数据都学不会，
    那要么生成器没做出可分性、要么模型/训练有 bug —— 无论哪种都得出声。

    这里单独造一份数据（而不是复用 ``ds`` 夹具）：CNN 参数量比 SVM 大得多，
    2 受试者 × 8 试次/类 才 64 个试次，它学不动是**正常**的，拿这个判成败
    只会得到假警报。规模取够、信噪比拉高，测的才是"流水线通不通"。
    """
    c = copy.deepcopy(cfg)
    c.data.synthetic.n_subjects = 4
    c.data.synthetic.trials_per_class = 30
    c.data.synthetic.snr_db = 6.0
    c.model.name = "cnn"
    c.model.cnn.n_epochs = 40
    c.model.cnn.early_stop_patience = 40     # 别让早停过早掐断
    c.train.n_splits = 3

    raw = generate_dataset(c)
    d, _ = preprocess_dataset(raw, c)
    res = cross_validate(c, d, model_name="cnn")
    assert res.accuracy > res.chance_level + 0.15, (
        f"CNN 准确率 {res.accuracy:.1%} 只比随机 {res.chance_level:.1%} 高一点点，"
        "流水线某处可能有问题")
    assert res.history and res.history.get("val_f1"), "CNN 没有产生验证曲线"


def test_cross_validate_returns_full_model(ds, small_cfg):
    """交叉验证要额外产出一个全量训练的模型，供推理页使用。

    它**不该**参与指标计算（它见过所有数据），但必须存在，
    否则推理页拿不到任何可用的模型。
    """
    c = copy.deepcopy(small_cfg)
    c.model.name = "svm"
    res = cross_validate(c, ds, model_name="svm")
    assert res.full_model is not None, "没有产出可供推理使用的全量模型"
    assert res.full_model.predict(ds.X[:3]).shape == (3,)


def test_group_by_subject_reduces_leakage(ds, small_cfg):
    """留一受试者必须能跑通，且用到的折数不超过受试者数。"""
    c = copy.deepcopy(small_cfg)
    c.model.name = "svm"
    res = cross_validate(c, ds, model_name="svm", group_by_subject=True)
    n_subj = len(np.unique(ds.subject))
    assert len(res.folds) <= n_subj
    assert res.group_by_subject is True
    assert "受试者" in res.split


def test_cross_validate_rejects_missing_subject(ds, small_cfg):
    """数据里没有受试者信息时，跨受试者评估要明确报错而不是瞎跑。"""
    d = copy.deepcopy(ds)
    d.subject = None
    with pytest.raises(ValueError, match="受试者"):
        cross_validate(small_cfg, d, model_name="svm", group_by_subject=True)


def test_more_folds_than_samples_per_class_is_capped(ds, small_cfg):
    """折数超过"最小类别试次数"时自动降低，而不是崩溃。

    同时验证上限用的是**最小类别试次数**，不是类别数 —— 用类别数当上限
    会让"4 分类要 5 折"被悄悄降成 4 折。
    """
    c = copy.deepcopy(small_cfg)
    c.model.name = "svm"
    c.train.n_splits = 5
    res = cross_validate(c, ds, model_name="svm")
    per_class = int(np.bincount(ds.y).min())
    assert len(res.folds) == min(5, per_class)
    c.train.n_splits = 999
    res2 = cross_validate(c, ds, model_name="svm")
    assert len(res2.folds) == per_class, "折数上限应当是每类试次数，不是类别数"


# ---------------------------------------------------------------------------
#  留出法 / 置换检验 / 对比
# ---------------------------------------------------------------------------


def test_holdout_evaluate(ds, small_cfg):
    c = copy.deepcopy(small_cfg)
    c.model.name = "svm"
    res = holdout_evaluate(c, ds, model_name="svm")
    assert res.n_test > 0 and res.n_train > 0
    assert 0.0 <= res.accuracy <= 1.0


def test_permutation_gives_chance_level(ds, small_cfg):
    """置换检验是**阴性对照**：打乱标签后准确率必须掉回随机水平附近。

    这是证明整条流水线没有数据泄漏的关键证据。如果打乱标签还能拿到
    高准确率，说明有信息从训练集漏到了测试集，所有结果都不可信。
    """
    c = copy.deepcopy(small_cfg)
    c.model.name = "svm"
    perm = permutation_test(c, ds, model_name="svm",
                            observed=0.5, n_permutations=5)
    assert len(perm.scores) == 5
    assert 0.0 <= perm.p_value <= 1.0
    # 打乱标签后应当接近随机（这里给足余量，因为小数据 + 折数少）
    assert perm.mean < ds.chance_level + 0.25, (
        f"打乱标签后平均准确率 {perm.mean:.1%} 远高于随机 "
        f"{ds.chance_level:.1%}，疑似存在数据泄漏")
    assert perm.summary().strip()


def test_comparison_table_and_models(ds, small_cfg):
    """两个模型都跑一遍并出对比表 —— 立项报告要的核心结论。"""
    c = copy.deepcopy(small_cfg)
    results = {}
    for name in ("svm", "cnn"):
        c.model.name = name
        results[name] = cross_validate(c, ds, model_name=name)
    table = comparison_table(results)
    assert "SVM" in table.upper()
    assert "CNN" in table.upper()
    assert results["svm"].model_name == "svm"
    assert results["cnn"].model_name == "cnn"


# ---------------------------------------------------------------------------
#  画图与报告 —— 这些最容易在改代码时被悄悄改坏
# ---------------------------------------------------------------------------


def test_all_plots_render(ds):
    """每个画图函数都要能出图。界面和报告全靠它们，崩一个就少一张图。"""
    import matplotlib.pyplot as plt

    from cneeg import plots

    figs = {
        "波形": lambda: plots.plot_waveform(ds),
        "ERP": lambda: plots.plot_erp(ds),
        "时频": lambda: plots.plot_tfr(ds),
        "可分性拓扑": lambda: plots.plot_discriminability(ds),
    }
    for name, fn in figs.items():
        fig = fn()
        assert fig is not None, f"{name} 没有返回 Figure"
        plt.close(fig)


def test_result_plots_render(ds, small_cfg):
    import matplotlib.pyplot as plt

    from cneeg import plots

    c = copy.deepcopy(small_cfg)
    c.model.name = "svm"
    res = cross_validate(c, ds, model_name="svm")

    fig = plots.plot_confusion(res)
    plt.close(fig)
    # history 可能是 None（SVM 没有训练曲线）—— 这种情况不该崩
    fig = plots.plot_training_curve(None, res.chance_level)
    plt.close(fig)
    fig = plots.plot_model_comparison({"svm": res, "cnn": res})
    plt.close(fig)


def test_report_builds(tmp_path, ds, raw_ds, small_cfg):
    """HTML 报告要能生成，且是自包含的单文件。"""
    c = copy.deepcopy(small_cfg)
    c.model.name = "svm"
    res = cross_validate(c, ds, model_name="svm")
    _, rep = preprocess_dataset(raw_ds, c)

    from cneeg.report import build_report

    out = build_report(c, ds, {"svm": res}, preprocess_report=rep,
                       out_path=tmp_path / "report.html")
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert len(html) > 5000
    # 图片必须是内嵌的 base64 —— 否则发给老师就是一堆红叉
    assert "data:image/png;base64," in html
    assert "<img src=\"http" not in html, "报告引用了外部图片，不是自包含的"
    # 仿真数据必须带警示，不能让人误以为是真实实验结论
    assert "仿真" in html, "仿真数据的报告缺少标注"


def test_report_marks_real_data(tmp_path, ds, small_cfg):
    """真实数据的报告不该出现"仿真"警示。"""
    import copy as _copy

    from cneeg.report import build_report
    from cneeg.train import cross_validate as cv

    c = _copy.deepcopy(small_cfg)
    res = cv(c, ds, model_name="svm")
    d = _copy.deepcopy(ds)
    d.is_synthetic = False
    out = build_report(c, d, {"svm": res}, out_path=tmp_path / "real.html")
    html = out.read_text(encoding="utf-8")
    assert "真实数据" in html


# ---------------------------------------------------------------------------
#  全部预设都能走通
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset", ["vowel_tone", "mandarin_word", "english_word"])
def test_every_preset_runs_pipeline(small_cfg, preset):
    """三个预设都要能一路跑到评估 —— 答辩现场切预设不能翻车。"""
    c = copy.deepcopy(small_cfg)
    c.data.synthetic.preset = preset
    c.data.synthetic.trials_per_class = 6
    c.model.name = "svm"

    raw = generate_dataset(c)
    d, _ = preprocess_dataset(raw, c)
    res = cross_validate(c, d, model_name="svm")
    assert len(res.folds) >= 2
    assert set(res.per_class) == set(d.class_names)


def test_cross_validate_progress_events_are_well_formed(ds, small_cfg):
    """进度回调的事件序列必须自洽 —— 界面和终端的进度显示全靠它。

    这里曾经有过一个**会骗人**的细节：交叉验证跑完、开始训练部署模型之前，
    代码又发了一次 ``(n_splits, n_splits, None, None, None)``，和"某折结束"
    是同一个事件。于是最后一折的"第 N/N 折完成"被打印两遍，而且第二遍恰好
    出现在"马上要训部署模型（CNN 要好几分钟）"的时刻 —— 进度条停着不动，
    文字却说某一折完成了，报告的是没发生的事。

    现在部署阶段用 ``fold=0`` 明确区分。这条测试就是要让那个重复事件再也回不来。
    """
    c = copy.deepcopy(small_cfg)
    c.train.n_splits = 3
    events = []
    cross_validate(c, ds, model_name="svm",
                   progress_callback=lambda *a: events.append(a))

    fold_done = [e[0] for e in events if e[2] is None]
    assert fold_done == [1, 2, 3, 0], (
        f"折结束事件应恰好是 1..N 各一次、最后一次是部署阶段（fold=0），"
        f"实际为 {fold_done}；出现重复的 n_splits 就说明旧的重复上报又回来了")
    assert fold_done.count(0) == 1, "部署阶段只该上报一次"
