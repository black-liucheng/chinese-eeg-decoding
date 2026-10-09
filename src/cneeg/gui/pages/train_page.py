# -*- coding: utf-8 -*-
"""训练页：选模型、调超参、后台训练、实时训练曲线。"""

from __future__ import annotations

from PyQt5.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
                             QHBoxLayout, QLabel, QPushButton, QSpinBox,
                             QVBoxLayout)

from ...plots import plot_training_curve
from ..workers import TrainWorker
from .base import FigurePanel, PageBase


class TrainPage(PageBase):

    title = "③ 训练"

    def __init__(self, cfg, parent=None):
        super().__init__(cfg, parent)
        self.result = None

        root = QVBoxLayout(self)
        root.addWidget(self.hint(
            "训练在后台线程里跑，界面不会卡。CNN 在 CPU 上一般要几分钟，"
            "进度条下方会实时画验证集 F1 曲线。"))

        # ---- 模型选择 ----
        top = QHBoxLayout()
        top.addWidget(QLabel("模型"))
        self.model_combo = QComboBox()
        self.model_combo.addItem("SVM（STFT 特征 + RBF 核，快速基线）", "svm")
        self.model_combo.addItem("ShallowCNN（端到端深度学习）", "cnn")
        i = self.model_combo.findData(cfg.model.name)
        self.model_combo.setCurrentIndex(max(i, 0))
        self.model_combo.currentIndexChanged.connect(self._sync_model_widgets)
        top.addWidget(self.model_combo, stretch=1)
        root.addLayout(top)

        # ---- 评估设置 ----
        ev = QFormLayout()
        self.n_splits = QSpinBox()
        self.n_splits.setRange(2, 10)
        self.n_splits.setValue(cfg.train.n_splits)
        self.n_splits.setToolTip("分层 K 折。折数越多每折训练集越大，但总耗时线性增长")
        ev.addRow("交叉验证折数", self.n_splits)

        self.group = QCheckBox("留一受试者评估（测试受试者训练时完全没见过）")
        self.group.setChecked(cfg.train.group_by_subject)
        self.group.setToolTip(
            "更严格也更贴近实际部署。结果通常明显低于普通交叉验证 —— "
            "这个差距本身就说明模型学到的有多少是个体特征")
        ev.addRow("", self.group)

        self.perm = QCheckBox("额外做置换检验（阴性对照，证明没有数据泄漏）")
        self.perm.setChecked(cfg.train.permutation_test)
        self.perm.setToolTip("打乱标签重跑若干次建立零分布。CNN 上会使总耗时增加数倍")
        ev.addRow("", self.perm)

        self.n_perm = QSpinBox()
        self.n_perm.setRange(5, 200)
        self.n_perm.setValue(cfg.train.n_permutations)
        ev.addRow("置换次数", self.n_perm)
        root.addLayout(ev)

        # ---- 各自超参 ----
        self.svm_form = QFormLayout()
        self.C = QDoubleSpinBox()
        self.C.setRange(0.01, 1000)
        self.C.setDecimals(3)
        self.C.setValue(cfg.model.svm.C)
        self.C.setToolTip("惩罚系数。太小欠拟合，太大过拟合")
        self.svm_form.addRow("SVM C", self.C)

        self.gamma = QComboBox()
        self.gamma.setEditable(True)
        for g in ("scale", "auto"):
            self.gamma.addItem(g)
        self.gamma.setCurrentText(str(cfg.model.svm.gamma))
        self.svm_form.addRow("SVM gamma", self.gamma)

        self.pca = QSpinBox()
        self.pca.setRange(0, 2000)
        self.pca.setValue(cfg.model.svm.pca_components or 0)
        self.pca.setSpecialValueText("不降维")
        self.svm_form.addRow("PCA 维度", self.pca)

        self.cnn_form = QFormLayout()
        self.epochs = QSpinBox()
        self.epochs.setRange(5, 500)
        self.epochs.setValue(cfg.model.cnn.n_epochs)
        self.cnn_form.addRow("最大轮数", self.epochs)

        self.lr = QDoubleSpinBox()
        self.lr.setRange(1e-5, 0.1)
        self.lr.setDecimals(5)
        self.lr.setSingleStep(1e-4)
        self.lr.setValue(cfg.model.cnn.lr)
        self.cnn_form.addRow("学习率", self.lr)

        self.batch = QSpinBox()
        self.batch.setRange(4, 512)
        self.batch.setValue(cfg.model.cnn.batch_size)
        self.cnn_form.addRow("批大小", self.batch)

        self.patience = QSpinBox()
        self.patience.setRange(2, 100)
        self.patience.setValue(cfg.model.cnn.early_stop_patience)
        self.patience.setToolTip("验证集 F1 连续多少轮不提升就早停")
        self.cnn_form.addRow("早停耐心", self.patience)

        root.addLayout(self.svm_form)
        root.addLayout(self.cnn_form)

        # ---- 按钮 ----
        btns = QHBoxLayout()
        self.train_btn = QPushButton("开始训练")
        self.train_btn.setDefault(True)
        self.train_btn.clicked.connect(self.start_training)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.cancel_training)
        self.cancel_btn.setEnabled(False)
        btns.addWidget(self.train_btn)
        btns.addWidget(self.cancel_btn)
        btns.addStretch(1)
        root.addLayout(btns)

        self.progress = self.make_progress()
        root.addWidget(self.progress)
        self.status = QLabel("就绪")
        self.status.setStyleSheet("color:#2b5d8a;font-size:12px;")
        root.addWidget(self.status)

        self.panel = FigurePanel(figsize=(9, 3.2))
        root.addWidget(self.panel, stretch=2)
        self.log = self.make_log(120)
        root.addWidget(self.log, stretch=2)

        self._sync_model_widgets()

    # ------------------------------------------------------------------

    def _sync_model_widgets(self) -> None:
        is_cnn = self.model_combo.currentData() == "cnn"
        for i in range(self.svm_form.rowCount()):
            for role in (QFormLayout.LabelRole, QFormLayout.FieldRole):
                it = self.svm_form.itemAt(i, role)
                if it and it.widget():
                    it.widget().setEnabled(not is_cnn)
        for i in range(self.cnn_form.rowCount()):
            for role in (QFormLayout.LabelRole, QFormLayout.FieldRole):
                it = self.cnn_form.itemAt(i, role)
                if it and it.widget():
                    it.widget().setEnabled(is_cnn)

    def _apply_cfg(self) -> None:
        c = self.cfg
        c.model.name = self.model_combo.currentData()
        c.train.n_splits = int(self.n_splits.value())
        c.train.group_by_subject = bool(self.group.isChecked())
        c.train.permutation_test = bool(self.perm.isChecked())
        c.train.n_permutations = int(self.n_perm.value())
        c.model.svm.C = float(self.C.value())
        g = self.gamma.currentText().strip()
        c.model.svm.gamma = g if g in ("scale", "auto") else float(g)
        c.model.svm.pca_components = int(self.pca.value())
        c.model.cnn.n_epochs = int(self.epochs.value())
        c.model.cnn.lr = float(self.lr.value())
        c.model.cnn.batch_size = int(self.batch.value())
        c.model.cnn.early_stop_patience = int(self.patience.value())

    # ------------------------------------------------------------------

    def start_training(self) -> None:
        page = self.window().page("data")
        ds = getattr(page, "ds", None) if page else None
        if ds is None:
            self.log_line("【提示】还没有数据。请先到“① 数据”页生成或导入。")
            return

        self._apply_cfg()
        self.log_line("─" * 46)
        name = self.model_combo.currentData()
        self.log_line(f"开始训练 {name.upper()}　"
                      f"{self.n_splits.value()} 折"
                      f"{'（留一受试者）' if self.group.isChecked() else '（分层）'}"
                      f"　{'含置换检验' if self.perm.isChecked() else '不含置换检验'}")

        worker = TrainWorker(self.cfg, ds, name,
                             group_by_subject=self.group.isChecked(),
                             permutation=self.perm.isChecked())
        self.cancel_btn.setEnabled(True)
        self.start_worker(worker, self._on_trained, label="训练")

    def cancel_training(self) -> None:
        if self.busy and self.worker is not None:
            self.worker.cancel()
            self.log_line("已请求取消，等待当前折结束 …")
            self.cancel_btn.setEnabled(False)

    def _on_trained(self, result) -> None:
        self.cancel_btn.setEnabled(False)
        self.result = result
        self.log_line("")
        self.log_line(result.summary())
        try:
            self.panel.show_figure(
                plot_training_curve(result.history, result.chance_level,
                                    title=f"{result.model_name.upper()} 训练曲线"))
        except Exception as exc:                # noqa: BLE001
            self.log_line(f"【绘图失败】{exc}")

        self.window().broadcast_result(result)
        self.set_status(f"{result.model_name.upper()} 完成："
                        f"准确率 {result.accuracy:.1%}"
                        f"（随机 {result.chance_level:.0%}）")

    def on_enter(self) -> None:
        page = self.window().page("data")
        if page is not None and getattr(page, "ds", None) is None:
            self.set_status("还没有数据 —— 请先到“① 数据”页生成")
