# -*- coding: utf-8 -*-
"""评估页：汇总各模型结果、混淆矩阵、指标表、置换检验、导出 HTML 报告。"""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QHeaderView,
                             QLabel, QPushButton, QSplitter, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget)

from ...plots import (plot_confusion, plot_model_comparison, plot_permutation,
                      plot_training_curve)
from ...report import build_report
from .base import FigurePanel, PageBase


class EvaluatePage(PageBase):

    title = "④ 评估"

    def __init__(self, cfg, parent=None):
        super().__init__(cfg, parent)
        # 模型名 → EvaluationResult。训练页每训完一个就广播过来
        self.results = {}

        root = QVBoxLayout(self)
        root.addWidget(self.hint(
            "每训练完一个模型，结果会自动汇总到这里。两个模型都训完之后，"
            "对比图会显示谁更好、好多少。最后可以一键导出 HTML 报告 —— "
            "单个文件，图片已内嵌，直接发给老师就能打开。"))

        # ---- 顶部：选择看哪个 ----
        top = QHBoxLayout()
        top.addWidget(QLabel("查看"))
        self.result_combo = QComboBox()
        self.result_combo.addItem("模型对比（全部）", "__all__")
        self.result_combo.currentIndexChanged.connect(self.refresh)
        top.addWidget(self.result_combo, stretch=1)

        top.addWidget(QLabel("图"))
        self.view_combo = QComboBox()
        for key, label in (("confusion", "混淆矩阵"),
                           ("curve", "训练曲线"),
                           ("perm", "置换检验"),
                           ("compare", "模型对比")):
            self.view_combo.addItem(label, key)
        self.view_combo.currentIndexChanged.connect(self.refresh)
        top.addWidget(self.view_combo)
        root.addLayout(top)

        # ---- 按钮 ----
        btns = QHBoxLayout()
        self.compare_btn = QPushButton("训练两个模型并对比")
        self.compare_btn.setToolTip("依次训练 SVM 与 CNN 再汇总 —— 离席几分钟即可")
        self.compare_btn.clicked.connect(self.run_comparison)
        btns.addWidget(self.compare_btn)

        self.export_btn = QPushButton("导出 HTML 报告")
        self.export_btn.clicked.connect(self.export_report)
        btns.addWidget(self.export_btn)

        self.csv_btn = QPushButton("导出指标 CSV")
        self.csv_btn.clicked.connect(self.export_csv)
        btns.addWidget(self.csv_btn)
        btns.addStretch(1)
        root.addLayout(btns)

        self.progress = self.make_progress()
        self.progress.setVisible(False)
        root.addWidget(self.progress)
        self.status = QLabel("还没有结果 —— 请先到“③ 训练”页训练一个模型")
        self.status.setStyleSheet("color:#2b5d8a;font-size:12px;")
        root.addWidget(self.status)

        # ---- 图 + 表 ----
        split = QSplitter(Qt.Horizontal)
        self.panel = FigurePanel(figsize=(6.5, 5))
        split.addWidget(self.panel)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(QLabel("指标汇总"))
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["模型", "准确率", "宏 F1", "Cohen κ", "高出随机", "p 值"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        rl.addWidget(self.table, stretch=1)

        rl.addWidget(QLabel("逐类指标"))
        self.per_class = QTableWidget(0, 4)
        self.per_class.setHorizontalHeaderLabels(["类别", "精确率", "召回率", "F1"])
        self.per_class.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.per_class.verticalHeader().setVisible(False)
        self.per_class.setEditTriggers(QTableWidget.NoEditTriggers)
        rl.addWidget(self.per_class, stretch=1)
        split.addWidget(right)
        split.setSizes([620, 480])
        root.addWidget(split, stretch=1)

        self.log = self.make_log(88)
        root.addWidget(self.log)

    # ------------------------------------------------------------------
    #  结果汇入
    # ------------------------------------------------------------------

    def add_result(self, result) -> None:
        """训练页广播过来的结果。同名模型覆盖旧的。"""
        self.results[result.model_name] = result
        self._rebuild_combo()
        self.refresh()

    def _rebuild_combo(self) -> None:
        cur = self.result_combo.currentData()
        self.result_combo.blockSignals(True)
        self.result_combo.clear()
        self.result_combo.addItem("模型对比（全部）", "__all__")
        for name in sorted(self.results):
            r = self.results[name]
            self.result_combo.addItem(
                f"{name.upper()}　准确率 {r.accuracy:.1%}", name)
        i = self.result_combo.findData(cur)
        self.result_combo.setCurrentIndex(max(i, 0))
        self.result_combo.blockSignals(False)

    def current_result(self):
        """当前选中的结果；选的是"全部"时返回准确率最高的那个。"""
        name = self.result_combo.currentData()
        if name in self.results:
            return self.results[name]
        if not self.results:
            return None
        return max(self.results.values(), key=lambda r: r.accuracy)

    # ------------------------------------------------------------------
    #  刷新
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        if not self.results:
            return
        self._fill_table()
        self._fill_per_class()
        self._draw()
        if len(self.results) >= 2 and self.view_combo.currentData() != "compare":
            self.log_line("【提示】两个模型都有结果了 —— 把上面的“图”切成"
                          "“模型对比”可以看对比柱状图和显著性")

    def _draw(self) -> None:
        view = self.view_combo.currentData()
        res = self.current_result()
        try:
            if view == "compare" and len(self.results) >= 2:
                fig = plot_model_comparison(self.results)
            elif view == "curve" and res is not None:
                fig = plot_training_curve(
                    res.history, res.chance_level,
                    title=f"{res.model_name.upper()} 训练曲线")
            elif view == "perm" and res is not None and res.permutation is not None:
                fig = plot_permutation(res.permutation, res.chance_level)
            elif res is not None:
                fig = plot_confusion(res)
            else:
                return
            self.panel.show_figure(fig)
        except Exception as exc:                # noqa: BLE001
            self.log_line(f"【绘图失败】{exc}")

    def _fill_table(self) -> None:
        self.table.setRowCount(len(self.results))
        for row, name in enumerate(sorted(self.results)):
            r = self.results[name]
            p = r.permutation.p_value if r.permutation is not None else None
            cells = [name.upper(),
                     f"{r.accuracy:.1%} ± {r.accuracy_std:.1%}",
                     f"{r.macro_f1:.3f}",
                     f"{r.kappa:.3f}",
                     f"{r.gain_over_chance:+.1%}",
                     ("—" if p is None else
                      ("p < 0.001" if p < 0.001 else f"p = {p:.3f}"))]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                if col == 4 and r.gain_over_chance > 0:
                    item.setForeground(Qt.darkGreen)
                self.table.setItem(row, col, item)

    def _fill_per_class(self) -> None:
        res = self.current_result()
        if res is None or not res.per_class:
            self.per_class.setRowCount(0)
            return
        from ...config import CLASS_LABELS
        items = list(res.per_class.items())
        self.per_class.setRowCount(len(items))
        for row, (name, m) in enumerate(items):
            label = CLASS_LABELS.get(name, name)
            for col, text in enumerate([label,
                                        f"{m.get('precision', float('nan')):.3f}",
                                        f"{m.get('recall', float('nan')):.3f}",
                                        f"{m.get('f1', float('nan')):.3f}"]):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                self.per_class.setItem(row, col, item)

    # ------------------------------------------------------------------
    #  批量对比
    # ------------------------------------------------------------------

    def run_comparison(self) -> None:
        page = self.window().page("train")
        if page is None:
            return
        ds = getattr(page, "ds", None) or getattr(
            self.window().page("data"), "ds", None)
        if ds is None:
            self.log_line("【提示】还没有数据。请先到“① 数据”页生成或导入。")
            return
        if self.busy:
            self.log_line("【提示】已有任务在运行，请等它结束")
            return

        # 把训练页的评估设置读过来，保证两边口径一致
        page._apply_cfg()
        self.log_line("═" * 46)
        self.log_line("开始 SVM vs CNN 完整对比（依次训练两个模型）")
        self.progress.setVisible(True)
        self.compare_btn.setEnabled(False)
        self._run_next_model(["svm", "cnn"], ds)

    def _run_next_model(self, queue, ds) -> None:
        if not queue:
            self.compare_btn.setEnabled(True)
            self.progress.setVisible(False)
            self.log_line("对比完成 —— 切到“模型对比”看图")
            self.refresh()
            return
        name = queue[0]
        from ..workers import TrainWorker
        self.log_line(f"── 训练 {name.upper()} …")
        worker = TrainWorker(self.cfg, ds, name,
                             group_by_subject=self.cfg.train.group_by_subject,
                             permutation=self.cfg.train.permutation_test)
        self.start_worker(
            worker,
            lambda res, rest=queue[1:]: self._on_compare_step(res, rest, ds),
            label=f"训练 {name.upper()}")

    def _on_compare_step(self, result, rest, ds) -> None:
        self.add_result(result)
        self.log_line(result.summary())
        self._run_next_model(rest, ds)

    # ------------------------------------------------------------------
    #  导出
    # ------------------------------------------------------------------

    def export_report(self) -> None:
        if not self.results:
            self.log_line("【提示】还没有结果可导出")
            return
        page = self.window().page("data")
        ds = getattr(page, "ds", None) if page else None
        rep = getattr(page, "preprocess_report", None) if page else None

        default = str(self.cfg.sub_dir("reports") / "分析报告.html")
        path, _ = QFileDialog.getSaveFileName(
            self, "导出 HTML 报告", default, "HTML 文件 (*.html)")
        if not path:
            return
        try:
            out = build_report(self.cfg, ds, self.results,
                               preprocess_report=rep,
                               out_path=path)
        except Exception as exc:                # noqa: BLE001
            self.log_line(f"【导出失败】{exc}")
            return
        self.log_line(f"报告已导出：{out}")
        self.set_status(f"报告已导出到 {out}")

    def export_csv(self) -> None:
        if not self.results:
            self.log_line("【提示】还没有结果可导出")
            return
        default = str(self.cfg.sub_dir("reports") / "指标汇总.csv")
        path, _ = QFileDialog.getSaveFileName(
            self, "导出指标 CSV", default, "CSV 文件 (*.csv)")
        if not path:
            return
        try:
            import csv
            with open(path, "w", newline="", encoding="utf-8-sig") as fh:
                w = csv.writer(fh)
                w.writerow(["模型", "准确率", "标准差", "宏F1", "CohenKappa",
                            "随机水平", "高出随机", "折数", "划分方式",
                            "特征维度", "p值"])
                for name in sorted(self.results):
                    r = self.results[name]
                    w.writerow([
                        name.upper(), f"{r.accuracy:.6f}", f"{r.accuracy_std:.6f}",
                        f"{r.macro_f1:.6f}", f"{r.kappa:.6f}",
                        f"{r.chance_level:.6f}", f"{r.gain_over_chance:.6f}",
                        len(r.folds), r.split, r.feature_dim,
                        "" if r.permutation is None else
                        f"{r.permutation.p_value:.6f}"])
        except Exception as exc:                # noqa: BLE001
            self.log_line(f"【导出失败】{exc}")
            return
        self.log_line(f"指标已导出：{path}")
        self.set_status(f"CSV 已导出到 {path}")

    # ------------------------------------------------------------------

    def on_dataset_changed(self) -> None:
        # 数据换了，之前的结果就失效了（否则报告里会把两套数据的数字混在一起）
        if self.results:
            self.results.clear()
            self._rebuild_combo()
            self.table.setRowCount(0)
            self.per_class.setRowCount(0)
            self.panel.canvas.clear()
            self.panel.canvas.draw()
            self.log_line("【注意】数据集已更换，之前的评估结果已清空，请重新训练。")
        self.set_status("还没有结果 —— 请先到“③ 训练”页训练一个模型")
