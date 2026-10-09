# -*- coding: utf-8 -*-
"""
模型
====

两个模型，对应立项报告里的技术路线：

``SVMClassifier``
    STFT / 频带功率特征 → 标准化 →（可选 PCA）→ RBF 核 SVM。
    传统机器学习基线，参数量小、可解释、训练秒级完成。

``CNNClassifier``
    ShallowConvNet（Schirrmeister et al., 2017, *Deep learning with
    convolutional neural networks for EEG decoding and visualization*）。
    直接吃原始试次数据，时间卷积 + 空间卷积自动学习滤波器，
    不需要手工设计特征。

两者都实现 sklearn 的 ``fit`` / ``predict`` / ``predict_proba`` 接口，
所以 ``cross_val_score``、``StratifiedKFold`` 可以直接套用，交叉验证与
置换检验的代码对两个模型完全一致。

.. note::
   ShallowConvNet 的关键设计是"平方 + 平均池化 + 对数"三步，它等价于在
   带通滤波后的信号上计算**频带功率**，同时让滤波器本身可学习 —— 这正是
   EEG 解码里被反复验证有效的结构。卷积核长度约 25 个采样点（256Hz 下
   约 100ms），与 ERP 成分的时间尺度相匹配。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import Config

# ---------------------------------------------------------------------------
#  特征提取（作为 sklearn 变换器，便于整体交叉验证）
# ---------------------------------------------------------------------------


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """把 ``(n_trials, n_ch, n_time)`` 的试次数据变成特征矩阵。

    做成 sklearn 变换器的意义：它必须**放在交叉验证的折内**，但特征本身
    不依赖训练数据（没有需要 fit 的参数），所以 fit 是空操作，不会泄漏。
    """

    def __init__(self, sfreq: float = 256.0, method: str = "stft",
                 bands: Optional[Dict[str, Sequence[float]]] = None,
                 nperseg: int = 64, noverlap: int = 48,
                 use_bandpower: bool = True):
        self.sfreq = sfreq
        self.method = method
        self.bands = bands
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.use_bandpower = use_bandpower

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        from .features import bandpower_features, stft_features

        bands = self.bands or {
            "delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
            "beta": (13, 30), "gamma": (30, 40),
        }
        if self.method == "bandpower":
            feats, _ = bandpower_features(X, self.sfreq, bands)
        else:
            feats, _ = stft_features(
                X, self.sfreq, bands,
                nperseg=self.nperseg, noverlap=self.noverlap,
                include_bandpower=self.use_bandpower,
            )
        return feats


# ---------------------------------------------------------------------------
#  SVM 基线
# ---------------------------------------------------------------------------


class SVMClassifier(BaseEstimator, ClassifierMixin):
    """特征 → 标准化 →（PCA）→ SVM。整条链路作为一个估计器。"""

    def __init__(self, cfg: Config, sfreq: float = 256.0):
        self.cfg = cfg
        self.sfreq = sfreq
        self.pipeline_: Optional[Pipeline] = None

    def _build(self) -> Pipeline:
        c = self.cfg
        steps = [
            ("features", FeatureExtractor(
                sfreq=self.sfreq,
                method=c.features.method,
                bands={k: tuple(v) for k, v in c.features.bands.items()},
                nperseg=c.features.stft_nperseg,
                noverlap=c.features.stft_noverlap,
                use_bandpower=c.features.stft_use_bandpower,
            )),
            ("scaler", StandardScaler()),
        ]
        if c.model.svm.pca_components and c.model.svm.pca_components > 0:
            steps.append(("pca", PCA(n_components=c.model.svm.pca_components,
                                     random_state=c.seed)))
        steps.append(("svm", SVC(
            C=c.model.svm.C, gamma=c.model.svm.gamma, kernel=c.model.svm.kernel,
            probability=True, random_state=c.seed,
        )))
        return Pipeline(steps)

    def fit(self, X, y):
        if hasattr(X, "X"):          # 允许直接传 EEGDataset
            X, y = X.X, X.y
        self.pipeline_ = self._build()
        self.pipeline_.fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        if hasattr(X, "X"):
            X = X.X
        return self.pipeline_.predict(X)

    def predict_proba(self, X):
        if hasattr(X, "X"):
            X = X.X
        return self.pipeline_.predict_proba(X)

    def feature_dim(self, n_channels: int, n_times: int) -> int:
        """这套特征在这个数据形状下是多少维 —— 报告里要写清楚。"""
        probe = np.zeros((1, n_channels, n_times), dtype=np.float32)
        return int(self._build().named_steps["features"].transform(probe).shape[1])


# ---------------------------------------------------------------------------
#  ShallowConvNet
# ---------------------------------------------------------------------------


def _torch():
    """延迟导入 torch：没装 torch 时其余功能（数据、特征、SVM）仍可用。

    torch 是可选依赖 —— 只跑 SVM 基线的话完全用不上它，所以不在模块顶层导入。
    代价是没装的时候报错发生在训练中途，而 ``ModuleNotFoundError: No module
    named 'torch'`` 对不熟悉 Python 的使用者没有任何指向性，所以这里换成
    能直接照做的提示。
    """
    try:
        import torch
    except ImportError as exc:                             # pragma: no cover
        raise ImportError(
            "深度模型需要 PyTorch，但当前环境里没有装。\n"
            "  SVM 基线不依赖它，可以先用 --model svm 跑通全流程。\n"
            "  要跑 CNN，请安装 CPU 版（体积小、无 CUDA 兼容问题）：\n"
            "      pip install torch --index-url "
            "https://download.pytorch.org/whl/cpu\n"
            "  装完确认：python -c \"import torch; print(torch.__version__)\""
        ) from exc
    return torch


class _ShallowConvNetModule:
    """真正的网络定义。放在函数里惰性构造，避免模块导入时就要求 torch。"""

    @staticmethod
    def build(n_ch: int, n_times: int, n_classes: int, cfg: Config):
        torch = _torch()
        nn = torch.nn
        c = cfg.model.cnn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_time = nn.Conv2d(1, c.n_filters_time,
                                           (1, c.filter_time_length), bias=False)
                self.bn_time = nn.BatchNorm2d(c.n_filters_time)
                self.conv_spat = nn.Conv2d(c.n_filters_time, c.n_filters_spat,
                                           (n_ch, 1), bias=False)
                self.bn_spat = nn.BatchNorm2d(c.n_filters_spat)
                self.pool = nn.AvgPool2d((1, c.pool_time_length),
                                         (1, c.pool_time_stride))
                self.drop = nn.Dropout(c.dropout)
                with torch.no_grad():
                    n_flat = self._features(torch.zeros(1, 1, n_ch, n_times)
                                            ).flatten(1).shape[1]
                self.fc = nn.Linear(n_flat, n_classes)

            def _features(self, x):
                x = self.bn_time(self.conv_time(x))
                x = self.bn_spat(self.conv_spat(x))
                x = x * x                       # 平方 ≈ 瞬时功率
                x = self.pool(x)                # 时间窗平均 ≈ 频带能量
                return torch.log(torch.clamp(x, min=1e-6))

            def forward(self, x):
                return self.fc(self.drop(self._features(x).flatten(1)))

            @property
            def n_flat(self):
                return self.fc.in_features

        return Net()


class CNNClassifier(BaseEstimator, ClassifierMixin):
    """ShallowConvNet 的 sklearn 风格封装。

    ``fit`` 时从训练集内部划出一小块做验证，用于早停并记录训练曲线
    （界面上实时画的那两条线）。预测时返回 softmax 概率。
    """

    def __init__(self, cfg: Config, n_channels: int = 32, n_times: int = 307,
                 n_classes: int = 4, verbose: bool = False,
                 progress_callback=None):
        self.cfg = cfg
        self.n_channels = n_channels
        self.n_times = n_times
        self.n_classes = n_classes
        self.verbose = verbose
        self.progress_callback = progress_callback
        self.net_ = None
        self.history_: Dict[str, List[float]] = {}

    # -- 数据格式 --
    def _to_tensor(self, X):
        torch = _torch()
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 3:
            X = X[:, None, :, :]              # (n, 1, ch, t)
        return torch.from_numpy(X)

    # -- 训练 --
    def fit(self, X, y):
        from sklearn.model_selection import train_test_split

        torch = _torch()
        nn = torch.nn
        cfg = self.cfg
        c = cfg.model.cnn

        if hasattr(X, "X"):
            X, y = X.X, X.y
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        self.n_channels = X.shape[1]
        self.n_times = X.shape[2]
        self.n_classes = int(len(np.unique(y)))
        self.classes_ = np.unique(y)

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        # 标准化：只用训练部分统计量
        mu, sd = float(X.mean()), float(X.std() + 1e-6)
        Xn = (X - mu) / sd
        self._mu, self._sd = mu, sd

        # 内部验证集（早停用）
        n_val = max(int(0.15 * len(Xn)), min(self.n_classes * 2, len(Xn) // 4))
        if n_val >= 5 and len(Xn) - n_val >= self.n_classes * 2:
            Xtr, Xva, ytr, yva = train_test_split(
                Xn, y, test_size=n_val, stratify=y, random_state=cfg.seed)
        else:                                  # 数据太少，退化为不早停
            Xtr, ytr, Xva, yva = Xn, y, None, None

        self.net_ = _ShallowConvNetModule.build(
            self.n_channels, self.n_times, self.n_classes, cfg)

        opt = torch.optim.Adam(self.net_.parameters(), lr=c.lr,
                               weight_decay=c.weight_decay)
        if c.class_weight:
            counts = np.bincount(ytr, minlength=self.n_classes).astype(np.float32)
            counts[counts == 0] = 1.0
            w = torch.tensor(counts.sum() / (self.n_classes * counts),
                             dtype=torch.float32)
            lossf = nn.CrossEntropyLoss(weight=w)
        else:
            lossf = nn.CrossEntropyLoss()

        Xtr_t, ytr_t = self._to_tensor(Xtr), torch.from_numpy(ytr)
        if Xva is not None:
            Xva_t = self._to_tensor(Xva)

        # val_f1 是**早停和选最佳轮次真正依据的指标**，必须留下 ——
        # 只记 val_acc 的话，界面上的训练曲线画的是准确率，
        # 而模型实际优化的却是宏 F1，两者对不上会让人看不懂早停为什么停在那一轮。
        hist = {"epoch": [], "train_loss": [],
                "val_loss": [], "val_acc": [], "val_f1": []}
        best_f1, best_state, best_epoch, patience = -1.0, None, 0, 0
        n = len(Xtr_t)
        bs = min(c.batch_size, n)

        for epoch in range(c.n_epochs):
            self.net_.train()
            perm = torch.randperm(n)
            total = 0.0
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                loss = lossf(self.net_(Xtr_t[idx]), ytr_t[idx])
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)

            hist["epoch"].append(epoch + 1)
            hist["train_loss"].append(total / n)

            if Xva_t is not None:
                self.net_.eval()
                with torch.no_grad():
                    logits = self.net_(Xva_t)
                    vloss = float(lossf(logits, torch.from_numpy(yva)))
                    pred = logits.argmax(1).numpy()
                from sklearn.metrics import f1_score
                vf1 = f1_score(yva, pred, average="macro", zero_division=0)
                hist["val_loss"].append(vloss)
                hist["val_acc"].append(float((pred == yva).mean()))
                hist["val_f1"].append(float(vf1))
                if self.progress_callback is not None:
                    self.progress_callback(epoch + 1, c.n_epochs, vf1)
                if vf1 > best_f1 + 1e-4:
                    best_f1, best_epoch, patience = vf1, epoch + 1, 0
                    best_state = {k: v.detach().clone()
                                  for k, v in self.net_.state_dict().items()}
                else:
                    patience += 1
                    if patience >= c.early_stop_patience:
                        if self.verbose:
                            print(f"  早停于第 {epoch+1} 轮（最佳第 {best_epoch} 轮，"
                                  f"验证宏F1={best_f1:.3f}）")
                        break
            else:
                hist["val_loss"].append(float("nan"))
                hist["val_acc"].append(float("nan"))
                hist["val_f1"].append(float("nan"))

        if best_state is not None:
            self.net_.load_state_dict(best_state)
        self.history_ = hist
        self.best_epoch_ = best_epoch
        return self

    # -- 预测 --
    def _logits(self, X):
        torch = _torch()
        X = np.asarray(X, dtype=np.float32)
        Xn = (X - self._mu) / self._sd
        self.net_.eval()
        with torch.no_grad():
            return self.net_(self._to_tensor(Xn)).numpy()

    def predict(self, X):
        if hasattr(X, "X"):
            X = X.X
        return self._logits(X).argmax(1)

    def predict_proba(self, X):
        if hasattr(X, "X"):
            X = X.X
        z = self._logits(X)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
#  工厂
# ---------------------------------------------------------------------------


def build_model(cfg: Config, n_channels: int = 32, n_times: int = 307,
                n_classes: int = 4, progress_callback=None,
                sfreq: float = 256.0):
    """按 ``cfg.model.name`` 造一个模型。"""
    if cfg.model.name == "svm":
        return SVMClassifier(cfg, sfreq=sfreq)
    if cfg.model.name == "cnn":
        return CNNClassifier(cfg, n_channels, n_times, n_classes,
                             progress_callback=progress_callback)
    raise ValueError(f"未知模型 {cfg.model.name!r}，可选 'svm' / 'cnn'")
