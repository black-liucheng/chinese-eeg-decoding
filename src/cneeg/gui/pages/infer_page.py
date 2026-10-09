# -*- coding: utf-8 -*-
"""推理页：载入一段记录 → 逐试次预测 → 类别置信度 + 空间拓扑。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QTableWidget,
                             QTableWidgetItem, QVBoxLayout)

from ...config import CLASS_LABELS
from ...plots import plot_topomap
from .base import FigurePanel, PageBase


class InferPage(PageBase):

    title = "⑤ 推理"

    def __init__(self, cfg, parent=None):
        super().__init__(cfg, parent)
        self.model = None
        self.class_names = []

        root = QVBoxLayout(self)
        root.addWidget(self.hint(
            "推理用的是当前内存里训练好的模型。如果还没训练过，先去“③ 训练”页跑一次；"
            "训练完成后模型会自动送过来。载入数据后点“开始预测”，"
            "左边是逐试次结果，右边是该试次的空间拓扑分布 —— "
            "拓扑上亮起来的位置就是模型认为“有信息”的电极。"))

        # ---- 数据来源 ----
        src = QHBoxLayout()
        self.src_combo = QComboBox()
        self.src_combo.addItem("使用当前数据集（① 数据页的数据）", "current")
        self.src_combo.addItem("载入脑电文件", "file")
        self.src_combo.currentIndexChanged.connect(self._on_src_changed)
        src.addWidget(QLabel("数据"))
        src.addWidget(self.src_combo, stretch=1)
        root.addLayout(src)

        file_row = QHBoxLayout()
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("选择要预测的脑电文件")
        self.file_edit.setEnabled(False)
        browse = QPushButton("浏览…")
        browse.clicked.connect(self._browse)
        browse.setEnabled(False)
        self._browse_btn = browse
        file_row.addWidget(self.file_edit)
        file_row.addWidget(browse)
        root.addLayout(file_row)

        # ---- 模型状态 ----
        self.model_lab = QLabel("模型：尚未载入 —— 请先在“③ 训练”页训练一个模型")
        self.model_lab.setStyleSheet("color:#b03030;font-size:12px;")
        root.addWidget(self.model_lab)

        # ---- 按钮 ----
        btns = QHBoxLayout()
        self.run_btn = QPushButton("开始预测")
        self.run_btn.setDefault(True)
        self.run_btn.clicked.connect(self.run_predict)
        self.run_btn.setEnabled(False)
        btns.addWidget(self.run_btn)
        btns.addStretch(1)
        root.addLayout(btns)

        self.progress = self.make_progress()
        root.addWidget(self.progress)
        self.status = QLabel("就绪")
        self.status.setStyleSheet("color:#2b5d8a;font-size:12px;")
        root.addWidget(self.status)

        # ---- 左右分栏：概率表 + 拓扑图 ----
        body = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(QLabel("逐试次预测（点一行看它的拓扑）"))
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["试次", "预测", "置信度"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        left.addWidget(self.table, stretch=1)
        body.addLayout(left, stretch=1)

        right = QVBoxLayout()
        right.addWidget(QLabel("该试次的空间分布"))
        self.panel = FigurePanel(figsize=(4.6, 3.6))
        right.addWidget(self.panel, stretch=1)
        body.addLayout(right, stretch=1)
        root.addLayout(body, stretch=1)

        self.log = self.make_log(84)
        root.addWidget(self.log)

        self._pred = None

    # ------------------------------------------------------------------

    def _on_src_changed(self, idx: int) -> None:
        is_file = self.src_combo.currentData() == "file"
        self.file_edit.setEnabled(is_file)
        self._browse_btn.setEnabled(is_file)

    def _browse(self) -> None:
        from ...data.loaders import SUPPORTED_FORMATS
        exts = " ".join(f"*{e}" for e in sorted(SUPPORTED_FORMATS))
        path, _ = QFileDialog.getOpenFileName(
            self, "选择脑电文件", "", f"脑电数据 ({exts});;所有文件 (*)")
        if path:
            self.file_edit.setText(path)

    # ------------------------------------------------------------------
    #  模型接收
    # ------------------------------------------------------------------

    def set_model(self, model, class_names) -> None:
        """训练页训完模型后调用（主窗口转发）。"""
        self.model = model
        self.class_names = list(class_names)
        labels = "、".join(CLASS_LABELS.get(c, c) for c in self.class_names)
        self.model_lab.setText(f"模型：已载入　可分类 {len(self.class_names)} 类 —— {labels}")
        self.model_lab.setStyleSheet("color:#2f7d54;font-size:12px;")
        self.run_btn.setEnabled(True)
        self.log_line(f"模型已就绪，输出类别：{labels}")

    # ------------------------------------------------------------------
    #  预测
    # ------------------------------------------------------------------

    def run_predict(self) -> None:
        if self.model is None:
            self.log_line("【提示】还没有模型，请先到“③ 训练”页训练。")
            return

        ds = self._load_input()
        if ds is None:
            return

        self.log_line("─" * 46)
        self.log_line(f"开始预测：{ds.n_trials} 试次 × {ds.n_channels} 通道")
        try:
            proba = np.asarray(self.model.predict_proba(ds.X))
        except Exception as exc:                # noqa: BLE001
            self.log_line(f"【预测失败】{exc}")
            return

        pred = proba.argmax(1)
        y_true = getattr(ds, "y", None)
        rows = []
        for i in range(len(pred)):
            name = self.class_names[pred[i]] if pred[i] < len(self.class_names) \
                else str(pred[i])
            rows.append((i + 1, name, float(proba[i, pred[i]]),
                         (None if y_true is None else int(y_true[i]))))

        self._pred = {"proba": proba, "pred": pred, "ds": ds, "rows": rows}
        self._fill_table(rows)
        if rows:
            self.table.selectRow(0)

        if y_true is not None and len(y_true) == len(pred):
            acc = float((pred == np.asarray(y_true)).mean())
            self.log_line(f"预测完成，与真实标签一致率 {acc:.1%}（{len(pred)} 试次）")
            self.set_status(f"预测完成：一致率 {acc:.1%}（部署模型自评，非性能指标）")
            # 这个数字**不是性能指标**：本页用的是在全量数据上重训的部署模型
            # （见技术文档 7.5 节）。默认输入又是"当前数据集"，也就是它训练时
            # 用过的同一批数据 —— 那等于考自己做过的题，一致率能到 100%。
            # 不写清楚的话，演示时屏幕上跳出一个接近满分的一致率，很容易被
            # 当成模型性能，那是最不该犯的错。
            self.log_line("【重要】上面的一致率不是模型性能 —— 本页用的是在全量数据上"
                          "重训的部署模型，若预测的正是训练用过的数据，它等于在背答案。")
            self.log_line("        模型的真实性能请看“④ 评估”页的交叉验证结果。")
        else:
            self.log_line(f"预测完成（{len(pred)} 试次，无真实标签可比对）")
            self.set_status(f"预测完成：{len(pred)} 试次")

    def _load_input(self):
        if self.src_combo.currentData() == "file":
            from ..workers import LoadFileWorker
            path = self.file_edit.text().strip()
            if not path:
                self.log_line("【提示】请先选择一个脑电文件")
                return None
            if not Path(path).exists():
                self.log_line(f"【错误】文件不存在：{path}")
                return None
            self.log_line(f"读取文件：{path}")
            # 文件读取可能很慢（几百 MB 的 .mff），但为了下面绘图方便这里同步读；
            # 读取失败会抛异常并被上层捕获。
            try:
                from ...data.epoching import preprocess_raw
                from ...data.loaders import load_recording
                rec = load_recording(path, self.cfg)
                ds, _ = preprocess_raw(
                    rec.data, rec.sfreq, rec.events, rec.event_codes,
                    rec.ch_names, rec.class_names, self.cfg,
                    rec.is_synthetic, rec.meta)
                return ds
            except Exception as exc:            # noqa: BLE001
                self.log_line(f"【读取失败】{exc}")
                return None

        page = self.window().page("data")
        ds = getattr(page, "ds", None) if page else None
        if ds is None:
            self.log_line("【提示】当前还没有数据。请先到“① 数据”页生成或导入。")
        return ds

    def _fill_table(self, rows) -> None:
        self.table.setRowCount(len(rows))
        for r, (idx, name, conf, true) in enumerate(rows):
            label = CLASS_LABELS.get(name, name)
            if true is not None:
                ok = (self.class_names.index(name) == true) \
                    if name in self.class_names else False
                label = ("✔ " if ok else "✘ ") + label
            cells = [f"#{idx}", label, f"{conf:.1%}"]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                if true is not None and c == 1:
                    item.setForeground(Qt.darkGreen if text.startswith("✔")
                                       else Qt.red)
                self.table.setItem(r, c, item)

    # ------------------------------------------------------------------

    def _on_row_selected(self) -> None:
        if self._pred is None:
            return
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        i = rows[0].row()
        ds = self._pred["ds"]
        try:
            # 用"该试次各类别的后验概率"在电极上做加权投影：取试次波形与
            # 各类平均模板的相关系数当空间权重，比单看一个时间点稳定。
            t = ds.times
            win = (t >= 0.25) & (t <= 0.60)      # ERP 主成分所在的时段
            seg = ds.X[i][:, win]
            weights = np.abs(seg).mean(axis=1)
            fig = plot_topomap(
                weights, ds.ch_names,
                title=f"试次 #{i + 1}　预测 {CLASS_LABELS.get(self.class_names[self._pred['pred'][i]], '')}"
                      f"（{self._pred['proba'][i, self._pred['pred'][i]]:.0%}）")
            self.panel.show_figure(fig)
        except Exception as exc:                # noqa: BLE001
            self.log_line(f"【绘图失败】{exc}")

    # ------------------------------------------------------------------

    def on_enter(self) -> None:
        if self.model is None:
            self.log_line("【提示】还没有模型 —— 请先到“③ 训练”页训练一个。")
