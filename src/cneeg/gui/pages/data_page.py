# -*- coding: utf-8 -*-
"""数据页：选数据源 → 生成/导入 → 波形预览。"""

from __future__ import annotations

from PyQt5.QtWidgets import (QComboBox, QDoubleSpinBox, QFileDialog,
                             QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QSpinBox, QVBoxLayout)

from ...config import PRESET_LABELS, PRESETS
from ...data.loaders import SUPPORTED_FORMATS
from ...plots import plot_erp, plot_waveform
from ..workers import GenerateWorker, LoadFileWorker
from .base import FigurePanel, PageBase


class DataPage(PageBase):

    title = "① 数据"

    def __init__(self, cfg, parent=None):
        super().__init__(cfg, parent)
        self.ds_raw = None            # 预处理前的数据（预处理页要用）

        root = QVBoxLayout(self)

        # ---- 数据源选择 ----
        src_form = QFormLayout()
        self.source_combo = QComboBox()
        self.source_combo.addItems(["仿真数据（内置生成器）", "真实脑电文件（MNE）"])
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        src_form.addRow("数据源", self.source_combo)

        self.preset_combo = QComboBox()
        for key, label in PRESET_LABELS.items():
            self.preset_combo.addItem(f"{label}　（{len(PRESETS[key])} 类）", key)
        src_form.addRow("仿真预设", self.preset_combo)

        file_row = QHBoxLayout()
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText(
            "选择脑电文件，支持 " + " ".join(sorted(SUPPORTED_FORMATS)))
        browse = QPushButton("浏览…")
        browse.clicked.connect(self._browse)
        file_row.addWidget(self.file_edit)
        file_row.addWidget(browse)
        src_form.addRow("脑电文件", file_row)

        # ---- 仿真参数 ----
        self.n_subj = QSpinBox()
        self.n_subj.setRange(1, 50)
        self.n_subj.setValue(cfg.data.synthetic.n_subjects)
        self.n_subj.setToolTip("受试者越多，跨受试者评估越有意义，生成也越慢")
        src_form.addRow("受试者数", self.n_subj)

        self.n_trials = QSpinBox()
        self.n_trials.setRange(10, 500)
        self.n_trials.setValue(cfg.data.synthetic.trials_per_class)
        src_form.addRow("每类试次数", self.n_trials)

        self.snr = QDoubleSpinBox()
        self.snr.setRange(-30, 20)
        self.snr.setSingleStep(1)
        self.snr.setDecimals(1)
        self.snr.setValue(cfg.data.synthetic.snr_db)
        self.snr.setSuffix(" dB")
        self.snr.setToolTip("越低越难。默认值经过标定，两个模型都能明显高于随机水平")
        src_form.addRow("信噪比", self.snr)

        self.seed = QSpinBox()
        self.seed.setRange(0, 2 ** 31 - 1)
        self.seed.setValue(cfg.seed)
        self.seed.setToolTip("固定种子保证结果完全可复现 —— 换一个数就是一套新数据")
        src_form.addRow("随机种子", self.seed)

        root.addLayout(src_form)

        # ---- 按钮 ----
        btns = QHBoxLayout()
        self.gen_btn = QPushButton("生成数据")
        self.gen_btn.setDefault(True)
        self.gen_btn.clicked.connect(self.generate)
        self.refresh_btn = QPushButton("重新绘制")
        self.refresh_btn.clicked.connect(self.redraw)
        btns.addWidget(self.gen_btn)
        btns.addWidget(self.refresh_btn)
        btns.addStretch(1)
        root.addLayout(btns)

        self.progress = self.make_progress()
        root.addWidget(self.progress)
        self.status = QLabel("就绪")
        self.status.setStyleSheet("color:#2b5d8a;font-size:12px;")
        root.addWidget(self.status)

        # ---- 图 + 日志 ----
        self.panel = FigurePanel(figsize=(9, 4.2))
        root.addWidget(self.panel, stretch=3)

        self.log = self.make_log(110)
        root.addWidget(self.log, stretch=1)

        self._on_source_changed(0)

    # ------------------------------------------------------------------

    def _on_source_changed(self, idx: int) -> None:
        synthetic = idx == 0
        for w in (self.preset_combo, self.n_subj, self.n_trials, self.snr, self.seed):
            w.setEnabled(synthetic)
        self.file_edit.setEnabled(not synthetic)
        self.gen_btn.setText("生成数据" if synthetic else "导入文件")

    def _browse(self) -> None:
        exts = " ".join(f"*{e}" for e in sorted(SUPPORTED_FORMATS))
        path, _ = QFileDialog.getOpenFileName(
            self, "选择脑电文件", "", f"脑电数据 ({exts});;所有文件 (*)")
        if path:
            self.file_edit.setText(path)

    # ------------------------------------------------------------------

    def _apply_cfg(self) -> None:
        """把界面上的参数写回配置对象（其它页读同一份 cfg）。"""
        c = self.cfg
        c.seed = int(self.seed.value())
        c.data.synthetic.preset = self.preset_combo.currentData()
        c.data.synthetic.n_subjects = int(self.n_subj.value())
        c.data.synthetic.trials_per_class = int(self.n_trials.value())
        c.data.synthetic.snr_db = float(self.snr.value())

    def generate(self) -> None:
        self._apply_cfg()
        self.log_line("─" * 46)

        if self.source_combo.currentIndex() == 1:
            path = self.file_edit.text().strip()
            if not path:
                self.log_line("【提示】请先选择一个脑电文件")
                return
            self.cfg.data.source = "file"
            self.cfg.data.file.path = path
            self.log_line(f"读取真实数据：{path}")
            worker = LoadFileWorker(self.cfg, path)
        else:
            self.cfg.data.source = "synthetic"
            self.log_line(
                f"生成仿真数据：{self.preset_combo.currentText()}　"
                f"{self.n_subj.value()} 受试者 × {self.n_trials.value()} 试次/类　"
                f"SNR {self.snr.value():+.1f} dB　种子 {self.seed.value()}")
            worker = GenerateWorker(self.cfg)

        self.start_worker(worker, self._on_data_ready, label="数据准备")

    def _on_data_ready(self, result) -> None:
        ds, report, ds_raw = result
        self.ds = ds
        self.preprocess_report = report
        self.ds_raw = ds_raw          # 预处理前的数据，预处理页做前后对比用
        self.log_line(ds.summary())
        if report is not None:
            self.log_line("")
            self.log_line(report.summary())
        self.redraw()
        self.window().broadcast_dataset_changed()

    def redraw(self) -> None:
        if self.ds is None:
            self.log_line("【提示】还没有数据，先点“生成数据”")
            return
        try:
            self.panel.show_figure(plot_waveform(self.ds, n_channels=8))
            self.set_status(f"当前数据：{self.ds.n_trials} 试次，"
                            f"{self.ds.n_classes} 分类")
        except Exception as exc:                # noqa: BLE001
            self.log_line(f"【绘图失败】{exc}")

    def on_dataset_changed(self) -> None:
        self.redraw()
