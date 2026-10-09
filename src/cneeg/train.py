# -*- coding: utf-8 -*-
"""
训练与评估
==========

统一跑 SVM 与 ShallowCNN 的训练、交叉验证、指标计算与置换检验。
两个模型都是 sklearn 风格估计器，所以这里的代码对它们完全一致。

评估口径
--------
- **分层 K 折交叉验证**（默认）：每折类别比例与总体一致。
- **留一受试者交叉验证**（``group_by_subject``）：按受试者分组，
  测试折里的受试者训练时完全没见过。这是更严格、也更贴近实际部署的设定 ——
  报告里两个结果都给出，可以说明模型到底学到的是"通用规律"还是"个体特征"。
- **置换检验**：把标签随机打乱后重跑，得到"没有真实效应时能拿到多好的结果"
  的零分布。如果真实准确率没有显著高于这个分布，说明模型学到的是数据泄漏
  或噪声。**这是证明结果可信的关键一步，不能省。**
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix, f1_score)
from sklearn.model_selection import (StratifiedGroupKFold, StratifiedKFold,
                                     train_test_split)

from .config import Config
from .data.base import EEGDataset
from .models import build_model


# ---------------------------------------------------------------------------
#  结果容器
# ---------------------------------------------------------------------------


@dataclass
class FoldResult:
    """单折的评估结果。"""

    fold: int
    accuracy: float
    macro_f1: float
    kappa: float
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: Optional[np.ndarray] = None


@dataclass
class PermutationResult:
    """置换检验的零分布。"""

    scores: List[float] = field(default_factory=list)
    observed: float = 0.0
    n_permutations: int = 0
    elapsed_s: float = 0.0

    @property
    def mean(self) -> float:
        return float(np.mean(self.scores)) if self.scores else float("nan")

    @property
    def std(self) -> float:
        return float(np.std(self.scores)) if self.scores else float("nan")

    @property
    def p_value(self) -> float:
        """经验 p 值：(不低于观测值的置换次数 + 1) / (总次数 + 1)。

        加 1 是标准做法 —— 避免在置换次数有限时得到 p=0 这种无法解释的值。
        """
        if not self.scores:
            return float("nan")
        n_ge = sum(1 for s in self.scores if s >= self.observed)
        return (n_ge + 1) / (len(self.scores) + 1)

    def summary(self) -> str:
        return (
            f"置换检验（{self.n_permutations} 次打乱标签重跑）：\n"
            f"  零分布：{self.mean:.1%} ± {self.std:.1%}"
            f"（打乱后模型应接近随机水平）\n"
            f"  实际观测：{self.observed:.1%}\n"
            f"  经验 p 值：{self.p_value:.3f}"
            f"{'  ← 显著（p<0.05）' if self.p_value < 0.05 else '  ← 不显著，结果存疑'}"
            f"\n  耗时：{self.elapsed_s:.1f} s"
        )


@dataclass
class EvaluationResult:
    """一次完整评估的全部产物 —— 报告、界面、图表都从这里取数。"""

    model_name: str
    class_names: List[str]
    folds: List[FoldResult]
    confusion: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    chance_level: float
    split: str
    group_by_subject: bool = False
    history: Optional[Dict[str, List[float]]] = None
    permutation: Optional[PermutationResult] = None
    feature_dim: int = 0
    n_train: int = 0
    n_test: int = 0
    elapsed_s: float = 0.0
    is_synthetic: bool = True
    dataset_summary: str = ""
    per_class: Dict[str, Dict[str, float]] = field(default_factory=dict)
    best_epoch: int = 0

    #: 用**全部数据**重新训练得到的模型，供推理页/部署使用。
    #:
    #: 上面那些指标一律来自交叉验证的折模型 —— 它们的测试折是"没见过"的，
    #: 所以数字可信。而这个 ``full_model`` 见过所有数据，**不能用它汇报性能**，
    #: 否则就是自欺欺人。它唯一正当的用途是：交叉验证已经证明方法可行之后，
    #: 把全部数据用上，得到一个实际可用的分类器。
    full_model: object = None

    # -- 汇总指标 --
    @property
    def accuracy(self) -> float:
        return float(np.mean([f.accuracy for f in self.folds])) if self.folds else float("nan")

    @property
    def accuracy_std(self) -> float:
        return float(np.std([f.accuracy for f in self.folds])) if self.folds else 0.0

    @property
    def macro_f1(self) -> float:
        return float(np.mean([f.macro_f1 for f in self.folds])) if self.folds else float("nan")

    @property
    def macro_f1_std(self) -> float:
        return float(np.std([f.macro_f1 for f in self.folds])) if self.folds else 0.0

    @property
    def kappa(self) -> float:
        return float(np.mean([f.kappa for f in self.folds])) if self.folds else float("nan")

    @property
    def gain_over_chance(self) -> float:
        """相对随机水平的提升 —— 比裸准确率更能说明模型是否真的学到了东西。"""
        if self.chance_level <= 0:
            return float("nan")
        return (self.accuracy - self.chance_level) / (1.0 - self.chance_level)

    def summary(self) -> str:
        tag = "【仿真数据】" if self.is_synthetic else "【真实数据】"
        lines = [
            f"{tag} 模型：{self.model_name.upper()}　"
            f"划分：{self.split}"
            f"{'（按受试者分组）' if self.group_by_subject else ''}",
            f"准确率　　：{self.accuracy:.1%} ± {self.accuracy_std:.1%}"
            f"　（随机 {self.chance_level:.1%}）",
            f"宏平均 F1：{self.macro_f1:.3f} ± {self.macro_f1_std:.3f}",
            f"Cohen's κ：{self.kappa:.3f}",
            f"相对随机的提升：{self.gain_over_chance:+.1%}（归一化）",
        ]
        if self.feature_dim:
            lines.append(f"特征维度：{self.feature_dim}")
        if self.best_epoch:
            lines.append(f"最佳轮次：{self.best_epoch}")
        lines.append(f"耗时：{self.elapsed_s:.1f} s")
        if self.permutation is not None:
            lines.append("")
            lines.append(self.permutation.summary())
        return "\n".join(lines)


# ---------------------------------------------------------------------------
#  指标
# ---------------------------------------------------------------------------


def _fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      class_names: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """逐类精确率/召回率/F1 —— 只看总准确率会掩盖"某一类完全学不会"。"""
    rep = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))),
        target_names=list(class_names), output_dict=True, zero_division=0,
    )
    out = {}
    for name in class_names:
        d = rep.get(name, {})
        out[name] = {
            "precision": float(d.get("precision", 0.0)),
            "recall": float(d.get("recall", 0.0)),
            "f1": float(d.get("f1-score", 0.0)),
            "support": float(d.get("support", 0.0)),
        }
    return out


# ---------------------------------------------------------------------------
#  交叉验证
# ---------------------------------------------------------------------------


def _make_splitter(cfg: Config, ds: EEGDataset, group_by_subject: bool):
    # 分层 K 折的上限是**最小类别的试次数**，不是类别数 ——
    # 用类别数当上限会让"4 分类想要 5 折"被悄悄降成 4 折，
    # 用户拿到的划分方式和配置里写的对不上，却没有任何提示。
    per_class = int(np.bincount(ds.y).min()) if ds.y.size else 0
    requested = cfg.train.n_splits
    n_splits = max(2, min(requested, max(per_class, 2)))

    reasons = []
    if n_splits != requested:
        reasons.append(f"分层 K 折要求每折都能分到每个类别，上限是最小类别的"
                       f"试次数（{per_class}）")

    if group_by_subject:
        if ds.subject is None:
            raise ValueError("数据里没有受试者信息（subject），无法做跨受试者评估")
        n_groups = len(np.unique(ds.subject))
        if n_groups < 2:
            raise ValueError(f"只有 {n_groups} 位受试者，无法做留一受试者交叉验证")
        if n_groups < n_splits:
            reasons.append(f"留一受试者评估的折数不能超过受试者数（{n_groups}）")
        n_splits = min(n_splits, n_groups)

    # 降折本身是必须的（不降 sklearn 直接报错），但**不能悄悄地降**。
    # 用户按 999 折等上一个小时，最后拿到的是另一种划分方式 ——
    # 耗时是真的，结论却不是他要的那个。所以这里明确说清楚。
    if reasons:
        print(f"[提示] 交叉验证折数由 {requested} 调整为 {n_splits}："
              + "；".join(reasons) + "。")

    if group_by_subject:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                    random_state=cfg.seed), n_splits
    return StratifiedKFold(n_splits=n_splits, shuffle=True,
                           random_state=cfg.seed), n_splits


def cross_validate(
    cfg: Config,
    ds: EEGDataset,
    model_name: Optional[str] = None,
    group_by_subject: bool = False,
    progress_callback=None,
) -> EvaluationResult:
    """分层 K 折（或留一受试者）交叉验证。

    每折都重新训练一个全新模型 —— 这是交叉验证唯一正确的做法。共用模型会
    让测试折的信息通过权重泄漏进后续折，得到虚高的准确率。

    ``progress_callback(fold, n_splits, epoch, total, f1)`` 一共三类事件：

    =====================  ==================================================
    ``fold == 0``          交叉验证已经跑完，正在用**全量数据**训练部署模型
                           （纯为推理服务，不参与任何指标 —— 见下面的注释）
    ``epoch is None``      第 ``fold`` 折刚刚结束
    其他                   第 ``fold`` 折的第 ``epoch`` 轮，验证宏 F1 = ``f1``
    =====================  ==================================================
    """
    t0 = time.time()
    model_name = model_name or cfg.model.name
    splitter, n_splits = _make_splitter(cfg, ds, group_by_subject)

    groups = ds.subject if group_by_subject else None
    folds: List[FoldResult] = []
    all_true, all_pred = [], []
    last_history, feature_dim, best_epoch = None, 0, 0

    split_iter = splitter.split(ds.X, ds.y, groups)

    for k, (tr, te) in enumerate(split_iter):
        # 每折单独一份配置副本，只改模型名，其余超参保持一致
        fold_cfg = _clone_cfg_with_model(cfg, model_name)
        n_classes = len(np.unique(ds.y))
        model = build_model(
            fold_cfg,
            n_channels=ds.n_channels, n_times=ds.n_times, n_classes=n_classes,
            sfreq=ds.sfreq,
            progress_callback=(
                (lambda ep, tot, f1, k=k: progress_callback(k, n_splits, ep, tot, f1))
                if progress_callback else None
            ),
        )
        model.fit(ds.X[tr], ds.y[tr])

        y_pred = model.predict(ds.X[te])
        try:
            y_proba = model.predict_proba(ds.X[te])
        except Exception:
            y_proba = None

        m = _fold_metrics(ds.y[te], y_pred)
        folds.append(FoldResult(
            fold=k + 1, y_true=ds.y[te], y_pred=y_pred, y_proba=y_proba, **m))

        all_true.append(ds.y[te])
        all_pred.append(y_pred)

        if getattr(model, "history_", None):
            last_history = model.history_
        best_epoch = getattr(model, "best_epoch_", best_epoch)
        if feature_dim == 0 and hasattr(model, "feature_dim"):
            try:
                feature_dim = model.feature_dim(ds.n_channels, ds.n_times)
            except Exception:
                feature_dim = 0

        if progress_callback is not None:
            progress_callback(k + 1, n_splits, None, None, None)

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    labels = list(range(len(ds.class_names)))

    # 交叉验证已经跑完（上面的指标全部来自"没见过测试折"的折模型），
    # 这里再用全量数据训一个最终模型，纯为推理/部署服务，不参与任何指标。
    #
    # 用 fold=0 这个哨兵值报告"进入部署阶段"。这里原先报的是
    # ``progress_callback(n_splits, n_splits, None, None, None)`` —— 和上面每折
    # 结束时发的是同一个事件，于是最后一折的 "第 N/N 折完成" 会被打印**两遍**，
    # 而且第二遍恰好出现在"马上要训部署模型、CNN 要跑好几分钟"的时刻：
    # 界面上进度条停在原地、文字却说某一折完成了，说的是没发生的事。
    full_model = None
    if progress_callback is not None:
        progress_callback(0, n_splits, None, None, None)
    try:
        deploy_cfg = _clone_cfg_with_model(cfg, model_name)
        full_model = build_model(deploy_cfg, n_channels=ds.n_channels,
                                 n_times=ds.n_times,
                                 n_classes=len(np.unique(ds.y)), sfreq=ds.sfreq)
        full_model.fit(ds.X, ds.y)
    except Exception as exc:                      # noqa: BLE001
        # 部署模型训不出来不该让整次评估失败 —— 指标已经拿到了。
        # 但**必须说出来**：这个异常一旦被吞掉，界面推理页只会显示
        # "尚未载入模型"、--save-model 只会说"没有产出模型"，两处都离
        # 真正的原因很远，用户根本无从查起。静默失败比失败更难修。
        full_model = None
        print(f"[警告] 部署模型（全量数据重训）失败，本次不产出可推理的模型：\n"
              f"        {type(exc).__name__}: {exc}\n"
              f"        交叉验证的各项指标不受影响，仍然有效。")

    return EvaluationResult(
        model_name=model_name,
        class_names=list(ds.class_names),
        folds=folds,
        confusion=confusion_matrix(y_true, y_pred, labels=labels),
        y_true=y_true,
        y_pred=y_pred,
        chance_level=ds.chance_level,
        split=f"{n_splits} 折交叉验证" if not group_by_subject
              else f"留一受试者 {n_splits} 折",
        group_by_subject=group_by_subject,
        history=last_history,
        feature_dim=feature_dim,
        n_train=int(len(ds.y) * (1 - 1 / n_splits)),
        n_test=int(len(ds.y) / n_splits),
        elapsed_s=time.time() - t0,
        is_synthetic=ds.is_synthetic,
        dataset_summary=ds.summary(),
        per_class=per_class_metrics(y_true, y_pred, ds.class_names),
        best_epoch=best_epoch,
        full_model=full_model,      # 仅供推理使用，不参与上面任何指标
    )


def holdout_evaluate(
    cfg: Config,
    ds: EEGDataset,
    model_name: Optional[str] = None,
    test_size: Optional[float] = None,
) -> EvaluationResult:
    """单次训练/测试划分 —— 比交叉验证快得多，界面上的"快速试跑"用它。"""
    t0 = time.time()
    model_name = model_name or cfg.model.name
    test_size = test_size or cfg.train.test_size

    tr, te = train_test_split(
        np.arange(len(ds.y)), test_size=test_size, stratify=ds.y,
        random_state=cfg.seed)

    fold_cfg = _clone_cfg_with_model(cfg, model_name)
    model = build_model(fold_cfg, n_channels=ds.n_channels, n_times=ds.n_times,
                        n_classes=len(np.unique(ds.y)), sfreq=ds.sfreq)
    model.fit(ds.X[tr], ds.y[tr])
    y_pred = model.predict(ds.X[te])
    try:
        y_proba = model.predict_proba(ds.X[te])
    except Exception:
        y_proba = None

    m = _fold_metrics(ds.y[te], y_pred)
    labels = list(range(len(ds.class_names)))
    feature_dim = 0
    if hasattr(model, "feature_dim"):
        try:
            feature_dim = model.feature_dim(ds.n_channels, ds.n_times)
        except Exception:
            feature_dim = 0

    return EvaluationResult(
        model_name=model_name,
        class_names=list(ds.class_names),
        folds=[FoldResult(fold=1, y_true=ds.y[te], y_pred=y_pred,
                          y_proba=y_proba, **m)],
        confusion=confusion_matrix(ds.y[te], y_pred, labels=labels),
        y_true=ds.y[te], y_pred=y_pred,
        chance_level=ds.chance_level,
        split=f"留出法（测试集 {test_size:.0%}）",
        history=getattr(model, "history_", None),
        feature_dim=feature_dim,
        n_train=len(tr), n_test=len(te),
        elapsed_s=time.time() - t0,
        is_synthetic=ds.is_synthetic,
        dataset_summary=ds.summary(),
        per_class=per_class_metrics(ds.y[te], y_pred, ds.class_names),
        best_epoch=getattr(model, "best_epoch_", 0),
    )


# ---------------------------------------------------------------------------
#  置换检验
# ---------------------------------------------------------------------------


def permutation_test(
    cfg: Config,
    ds: EEGDataset,
    model_name: Optional[str] = None,
    observed: Optional[float] = None,
    n_permutations: Optional[int] = None,
    progress_callback=None,
) -> PermutationResult:
    """打乱标签重跑，建立零分布。

    每次置换用一次留出法（而不是完整交叉验证）—— 置换次数通常有几十次，
    再乘上 K 折会让 CNN 跑到天亮。留出法足以说明"效应是否真实存在"。

    ``observed`` 不传的话，先用同一套留出法跑一次真实标签得到观测值，
    保证观测值与零分布用的是同一评估口径（可比的）。
    """
    t0 = time.time()
    n_permutations = n_permutations or cfg.train.n_permutations
    model_name = model_name or cfg.model.name

    if observed is None:
        observed = holdout_evaluate(cfg, ds, model_name).accuracy

    rng = np.random.default_rng(cfg.seed + 12345)
    scores: List[float] = []
    for i in range(n_permutations):
        shuffled = ds.subset(np.arange(ds.n_trials))
        shuffled.y = rng.permutation(ds.y)
        try:
            res = holdout_evaluate(cfg, shuffled, model_name)
            scores.append(res.accuracy)
        except Exception as exc:      # noqa: BLE001 — 单次失败不该毁掉整个检验
            print(f"  第 {i+1} 次置换失败（{exc}），跳过")
        if progress_callback is not None:
            progress_callback(i + 1, n_permutations)

    return PermutationResult(
        scores=scores, observed=float(observed),
        n_permutations=len(scores), elapsed_s=time.time() - t0,
    )


# ---------------------------------------------------------------------------
#  辅助
# ---------------------------------------------------------------------------


def _clone_cfg_with_model(cfg: Config, model_name: str) -> Config:
    """返回一份把 ``model.name`` 换掉的配置副本（浅拷贝足够，其余字段只读）。"""
    import copy
    c = copy.deepcopy(cfg)
    c.model.name = model_name
    return c


def compare_models(
    cfg: Config,
    ds: EEGDataset,
    models: Sequence[str] = ("svm", "cnn"),
    group_by_subject: bool = False,
    progress_callback=None,
) -> Dict[str, EvaluationResult]:
    """把多个模型跑同一套评估，用于"对比验证不同模型的分类性能"。

    ``progress_callback(name, payload)`` 里 ``payload`` 有两种：

    - ``None``：开始评估 ``name`` 这个模型了；
    - ``(fold, n_splits, epoch, total, f1)``：该模型的训练进度。

    进度必须往上传。CNN 跑一次交叉验证要十几分钟，中间一个字都不打印的话，
    命令行看起来就像卡死了 —— 用户会以为程序崩了然后 Ctrl+C 掉，
    白白浪费前面所有的时间。
    """
    results: Dict[str, EvaluationResult] = {}
    for name in models:
        if progress_callback is not None:
            progress_callback(name, None)
        results[name] = cross_validate(
            cfg, ds, model_name=name, group_by_subject=group_by_subject,
            progress_callback=(
                (lambda *a, _n=name: progress_callback(_n, a))
                if progress_callback else None
            ),
        )
    return results


def results_to_dict(results: Dict[str, EvaluationResult]) -> dict:
    """把评估结果转成可 JSON 序列化的字典。

    为什么除了 HTML 报告还要落一份 JSON：报告是给人看的，而**逐折指标**
    这种要拿去做配对比较、重新绘图、写进论文表格的东西，必须有机器可读的
    形式。只看聚合均值 ± 标准差，是做不了"两个模型逐折配对比较"的 ——
    而这恰恰是"CNN 到底有没有真的超过 SVM"唯一能说清的方式。
    """
    out = {}
    for name, r in results.items():
        folds = [{"fold": f.fold, "accuracy": float(f.accuracy),
                  "macro_f1": float(f.macro_f1), "kappa": float(f.kappa)}
                 for f in r.folds]
        accs = [f["accuracy"] for f in folds]
        entry = {
            "model": name,
            "split": r.split,
            "group_by_subject": bool(r.group_by_subject),
            "is_synthetic": bool(r.is_synthetic),
            "chance_level": float(r.chance_level),
            "class_names": list(r.class_names),
            "n_folds": len(folds),
            "n_train": int(r.n_train),
            "n_test": int(r.n_test),
            "accuracy_mean": float(r.accuracy),
            "accuracy_std": float(r.accuracy_std),
            "macro_f1_mean": float(r.macro_f1),
            "macro_f1_std": float(r.macro_f1_std),
            "kappa_mean": float(r.kappa),
            "gain_over_chance": float(r.gain_over_chance),
            "feature_dim": int(r.feature_dim),
            "best_epoch": int(r.best_epoch),
            "elapsed_s": float(r.elapsed_s),
            "folds": folds,
            # 平均值的标准误（std/√n），逐折独立时用它比较两个模型。
            "accuracy_sem": (float(np.std(accs, ddof=1) / np.sqrt(len(accs)))
                             if len(accs) > 1 else float("nan")),
            "per_class": {k: {kk: float(vv) for kk, vv in v.items()}
                          for k, v in r.per_class.items()},
        }
        if r.permutation is not None:
            p = r.permutation
            entry["permutation"] = {
                "observed": float(p.observed),
                "n_permutations": int(p.n_permutations),
                "null_mean": float(np.mean(p.scores)) if p.scores else float("nan"),
                "null_std": float(np.std(p.scores)) if p.scores else float("nan"),
                "null_max": float(np.max(p.scores)) if p.scores else float("nan"),
                "scores": [float(s) for s in p.scores],
                "p_value": float(p.p_value),
                "elapsed_s": float(p.elapsed_s),
            }
        out[name] = entry
    return out


def paired_model_comparison(results: Dict[str, EvaluationResult],
                            a: str = "svm", b: str = "cnn") -> Optional[str]:
    """两个模型**逐折配对**比较 —— 同一折上用同一份数据划分。

    不能用"均值 ± 标准差"直接比：两条流水线跑的是完全相同的折划分，误差是
    相关的，独立双样本的算法在这里既不准也不必要。配对看每一折谁赢、差多少，
    才说明差距稳不稳定。
    """
    if a not in results or b not in results:
        return None
    ra, rb = results[a], results[b]
    fa = {f.fold: f.accuracy for f in ra.folds}
    fb = {f.fold: f.accuracy for f in rb.folds}
    common = sorted(set(fa) & set(fb))
    if len(common) < 2:
        return None

    d = np.array([fb[k] - fa[k] for k in common])     # b 相对 a 的逐折增量
    wins = int(np.sum(d > 0))
    mean_d = float(d.mean())
    # 配对 t 统计量（自由度为折数-1）。折数少时功效低，所以下面同时
    # 给出"几折几胜"，不把结论全押在这个 p 值上。
    sem = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    t = mean_d / sem if sem and sem > 0 else float("nan")

    # 这里的减号用 ASCII 的 "-"：Chinese Windows 控制台是 GBK 编码，
    # 数学减号 U+2212、乘号 U+00D7 这类字符不在 GBK 里，一打印就是
    # UnicodeEncodeError，整个命令在最后一步崩掉。±、² 这些 GBK 里有，可以用。
    lines = [
        f"逐折配对比较（{b.upper()} - {a.upper()}，共 {len(common)} 折）",
        "-" * 52,
    ]
    for k, delta in zip(common, d):
        lines.append(f"  第 {k} 折　{a.upper()} {fa[k]:>6.1%}　"
                     f"{b.upper()} {fb[k]:>6.1%}　差 {delta:>+6.1%}")
    lines.append("-" * 52)
    lines.append(f"  平均差 {mean_d:+.1%}　配对标准误 {sem:.1%}　"
                 f"t = {t:.2f}（df={len(common)-1}）")
    lines.append(f"  {b.upper()} 胜 {wins}/{len(common)} 折")
    return "\n".join(lines)


def comparison_table(results: Dict[str, EvaluationResult]) -> str:
    """把多个模型的结果排成一张表（等宽，控制台与文档都能看）。"""
    if not results:
        return "（无结果）"

    any_res = next(iter(results.values()))
    head = (f"{'模型':<10}{'准确率':>16}{'宏F1':>16}{'κ':>10}{'耗时':>10}")
    sep = "-" * len(head.encode("gbk", errors="ignore"))
    lines = [
        f"分类性能对比（随机水平 {any_res.chance_level:.1%}"
        f"{'，仿真数据' if any_res.is_synthetic else ''}）",
        sep,
        head,
        sep,
    ]
    for name, r in results.items():
        lines.append(
            f"{name.upper():<10}"
            f"{r.accuracy:>9.1%} ± {r.accuracy_std:<4.1%}"
            f"{r.macro_f1:>9.3f} ± {r.macro_f1_std:<4.3f}"
            f"{r.kappa:>10.3f}"
            f"{r.elapsed_s:>9.1f}s"
        )
    lines.append(sep)
    return "\n".join(lines)
