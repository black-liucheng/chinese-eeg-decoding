# -*- coding: utf-8 -*-
"""预处理页：调滤波/参考/基线参数，即时看前后对比波形。"""

from __future__ import annotations

from PyQt5.QtWidgets import (QComboBox, QDoubleSpinBox, QFormLayout,
                             QHBoxLayout, QLabel, QPushButton, QVBoxLayout)

from ...data.epoching import preprocess_dataset
from ...plots import plot_preprocess_compare
from .base import FigurePanel, PageBase


class PreprocessPage(PageBase):

    title = "② 预处理"

    def __init__(self, cfg, parent=None):
        super().__init__(cfg, parent)
        self.ds_raw = None

        root = QVBoxLayout(self)
        root.addWidget(self.hint(
            "预处理参数会直接影响后面的模型结果。改完点“应用并预览”看效果 —— "
            "上图是原始波形，下图是处理后的。伪迹剔除会把峰峰值超限的试次整段丢掉，"
            "所以下图试次数可能变少。"))

        form = QFormLayout()

        self.l_freq = QDoubleSpinBox()
        self.l_freq.setRange(0, 40)
        self.l_freq.setSingleStep(0.1)
        self.l_freq.setDecimals(2)
        self.l_freq.setValue(cfg.preprocess.l_freq)
        self.l_freq.setSuffix(" Hz")
        self.l_freq.setToolTip("高通。太低会留下漂移，太高会削掉慢成分（如 N400 尾部）")
        form.addRow("带通下限", self.l_freq)

        self.h_freq = QDoubleSpinBox()
        self.h_freq.setRange(1, 128)
        self.h_freq.setSingleStep(1)
        self.h_freq.setDecimals(1)
        self.h_freq.setValue(cfg.preprocess.h_freq)
        self.h_freq.setSuffix(" Hz")
        self.h_freq.setToolTip("低通。40 Hz 是 ERP 分析常用上限，能滤掉大部分肌电")
        form.addRow("带通上限", self.h_freq)

        self.notch = QDoubleSpinBox()
        self.notch.setRange(0, 120)
        self.notch.setSingleStep(10)
        self.notch.setDecimals(0)
        self.notch.setValue(cfg.preprocess.notch_freq or 0)
        self.notch.setSuffix(" Hz")
        self.notch.setSpecialValueText("关闭")
        self.notch.setToolTip("中国市电 50 Hz。填 0 表示不做陷波")
        form.addRow("陷波", self.notch)

        self.reref = QComboBox()
        # 只列 rereference() 真正支持的两种 —— 之前这里还挂了个 "avg"，
        # 一选就抛 "未知的重参考方式"。宁可不给，也不能给了用不了。
        for val, label in (("car", "共同平均参考（CAR）"),
                           ("none", "不重参考")):
            self.reref.addItem(label, val)
        i = self.reref.findData(cfg.preprocess.reref)
        self.reref.setCurrentIndex(max(i, 0))
        form.addRow("重参考", self.reref)

        self.tmin = QDoubleSpinBox()
        self.tmin.setRange(-2, 0)
        self.tmin.setSingleStep(0.05)
        self.tmin.setDecimals(2)
        self.tmin.setValue(cfg.preprocess.tmin)
        self.tmin.setSuffix(" s")
        form.addRow("窗口起点", self.tmin)

        self.tmax = QDoubleSpinBox()
        self.tmax.setRange(0.1, 4)
        self.tmax.setSingleStep(0.1)
        self.tmax.setDecimals(2)
        self.tmax.setValue(cfg.preprocess.tmax)
        self.tmax.setSuffix(" s")
        self.tmax.setToolTip("汉语声调/语义效应通常在 300–800 ms，窗口太短会截掉")
        form.addRow("窗口终点", self.tmax)

        self.reject = QDoubleSpinBox()
        self.reject.setRange(0, 1000)
        self.reject.setSingleStep(10)
        self.reject.setDecimals(0)
        self.reject.setValue(cfg.preprocess.reject_uv or 0)
        self.reject.setSuffix(" µV")
        self.reject.setSpecialValueText("关闭")
        self.reject.setToolTip("峰峰值超过此值的试次被整段剔除。设太严会丢掉大量试次")
        form.addRow("伪迹阈值", self.reject)

        self.normalize = QComboBox()
        for val, label in (("zscore", "逐通道 z-score"),
                           ("none", "不标准化")):
            self.normalize.addItem(label, val)
        j = self.normalize.findData(cfg.preprocess.normalize)
        self.normalize.setCurrentIndex(max(j, 0))
        form.addRow("标准化", self.normalize)

        root.addLayout(form)

        btns = QHBoxLayout()
        self.apply_btn = QPushButton("应用并预览")
        self.apply_btn.setDefault(True)
        self.apply_btn.clicked.connect(self.apply_preprocess)
        btns.addWidget(self.apply_btn)
        btns.addStretch(1)
        root.addLayout(btns)

        self.progress = self.make_progress()
        self.progress.setVisible(False)         # 预处理很快，进度条意义不大
        root.addWidget(self.progress)
        self.status = QLabel("就绪")
        self.status.setStyleSheet("color:#2b5d8a;font-size:12px;")
        root.addWidget(self.status)

        self.panel = FigurePanel(figsize=(9, 4.5))
        root.addWidget(self.panel, stretch=3)
        self.log = self.make_log(95)
        root.addWidget(self.log, stretch=1)

    # ------------------------------------------------------------------

    def _apply_cfg(self) -> None:
        p = self.cfg.preprocess
        p.l_freq = float(self.l_freq.value())
        p.h_freq = float(self.h_freq.value())
        p.notch_freq = float(self.notch.value()) or 0.0
        p.reref = self.reref.currentData()
        p.tmin = float(self.tmin.value())
        p.tmax = float(self.tmax.value())
        p.reject_uv = float(self.reject.value()) or 0.0
        p.normalize = self.normalize.currentData()

    def apply_preprocess(self, broadcast: bool = True) -> None:
        """重跑预处理并刷新预览。

        ``broadcast=False`` 用于响应别页的数据变更 —— 否则
        ``on_dataset_changed → apply_preprocess → broadcast_dataset_changed``
        会无限递归。
        """
        if self.ds_raw is None:
            self.log_line("【提示】还没有数据。请先到“① 数据”页生成或导入数据。")
            return
        if self.tmin.value() >= self.tmax.value():
            self.log_line("【错误】窗口起点必须早于终点")
            return

        self._apply_cfg()
        self.log_line("─" * 46)
        self.log_line(f"重跑预处理：{self.l_freq.value():g}–{self.h_freq.value():g} Hz"
                      f"，参考 {self.reref.currentText()}，"
                      f"窗口 {self.tmin.value():g}~{self.tmax.value():g} s，"
                      f"伪迹阈值 {self.reject.value():g} µV")
        try:
            ds, report = preprocess_dataset(self.ds_raw, self.cfg)
        except Exception as exc:                # noqa: BLE001
            self.log_line(f"【预处理失败】{exc}")
            return

        self.ds = ds
        self.preprocess_report = report
        self.log_line(report.summary())

        if report.reject_ratio > 0.5:
            self.log_line("【注意】超过一半的试次被判为伪迹，阈值可能设得太严了。")

        try:
            ch = "Fz" if "Fz" in ds.ch_names else ds.ch_names[0]
            self.panel.show_figure(
                plot_preprocess_compare(self.ds_raw, ds, channel=ch))
            self.set_status(f"保留 {report.n_trials_out}/{report.n_trials_in} 试次"
                            f"（剔除 {report.n_rejected}）")
        except Exception as exc:                # noqa: BLE001
            self.log_line(f"【绘图失败】{exc}")

        if broadcast:
            self.window().broadcast_dataset_changed()

    def on_dataset_changed(self) -> None:
        # 数据页生成了新数据 → 本页同步一份原始数据并自动预览（不再向外广播）
        page = self.window().page("data")
        if page is not None and getattr(page, "ds_raw", None) is not None:
            self.ds_raw = page.ds_raw
            self.ds = page.ds
            self.preprocess_report = page.preprocess_report
            self.apply_preprocess(broadcast=False)

    def on_enter(self) -> None:
        page = self.window().page("data")
        if self.ds_raw is None and page is not None:
            self.ds_raw = getattr(page, "ds_raw", None)
