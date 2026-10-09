# -*- coding: utf-8 -*-
"""
页面基类与公共控件
==================

五个标签页共享的东西都放这里：matplotlib 画布、日志框、进度条、
参数行辅助函数。各页面只写自己特有的部分。

.. note::
   所有 matplotlib 画布都用 ``FigureCanvasQTAgg`` 内嵌。**不要**在 GUI 里调用
   ``plt.show()`` —— 那会弹出独立窗口并和 Qt 的事件循环打架，严重时整个程序
   卡死。统一用 ``Figure`` + canvas.draw()。
"""

from __future__ import annotations

from typing import Optional

from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QGroupBox, QHBoxLayout, QLabel, QPlainTextEdit,
                             QProgressBar, QSizePolicy, QVBoxLayout, QWidget)

from ...config import Config
from ...plots import CJK_FONT     # 导入即触发中文字体设置


class MplCanvas(FigureCanvas):
    """带工具栏的 matplotlib 画布，可直接塞进任何布局。"""

    def __init__(self, figsize=(7, 4.5), dpi=100, parent=None):
        self.fig = Figure(figsize=figsize, dpi=dpi, facecolor="white")
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()

    def clear(self):
        self.fig.clear()

    def draw_safe(self):
        """重绘。用 tight_layout 兜住"图太小导致标签重叠"的告警。"""
        try:
            self.fig.tight_layout()
        except Exception:                     # noqa: BLE001 — 布局失败不该中断界面
            pass
        self.draw()


class FigurePanel(QWidget):
    """画布 + 工具栏的组合控件。"""

    def __init__(self, figsize=(7, 4.5), parent=None):
        super().__init__(parent)
        self.canvas = MplCanvas(figsize, parent=self)
        self.toolbar = NavigationToolbar(self.canvas, self)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas)

    def show_figure(self, fig: Figure) -> None:
        """把外部生成的 ``Figure`` 搬到本画布上显示。

        ``clear()`` + 逐轴重画比较麻烦，直接把原图的 axes 搬过来又容易因
        figure 生命周期出问题；这里采取最稳妥的做法：把原图存成内存 PNG 再
        读回来画（多花几毫秒，但绝不会出现"画布空白"这种演示事故）。
        """
        import io

        import matplotlib.image as mpimg

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                    facecolor="white")
        buf.seek(0)
        img = mpimg.imread(buf)

        import matplotlib.pyplot as plt
        plt.close(fig)

        self.canvas.clear()
        ax = self.canvas.fig.add_subplot(111)
        ax.imshow(img)
        ax.axis("off")
        self.canvas.draw()


class LogBox(QPlainTextEdit):
    """只读日志框。界面上所有"发生了什么"都往这里写，方便演示时回看。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(4000)       # 防止长时间运行吃光内存
        self.setStyleSheet(
            "QPlainTextEdit{background:#fbfcfe;border:1px solid #dde5ee;"
            "border-radius:6px;font-family:Consolas,'Microsoft YaHei',monospace;"
            "font-size:12px;}")

    def append_line(self, text: str) -> None:
        self.appendPlainText(text)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())


class PageBase(QWidget):
    """所有标签页的基类。

    提供：``cfg``、日志框、进度条、``busy`` 状态管理（运行中禁用按钮，
    防止用户连点导致重复启动后台线程）。
    """

    #: 标签页标题（主窗口取这个当 tab 名）
    title: str = "页面"
    #: 是否是耗时页（主窗口据此决定切换时是否提示）
    heavy: bool = False

    def __init__(self, cfg: Config, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.ds = None                 # 当前数据集（EEGDataset）
        self.preprocess_report = None
        self.worker = None             # 正在跑的后台线程

    # -- 给子类用的小工具 --
    def make_log(self, height: int = 110) -> LogBox:
        box = LogBox()
        box.setFixedHeight(height)
        return box

    def make_progress(self) -> QProgressBar:
        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setTextVisible(True)
        return bar

    def group(self, title: str, layout) -> QGroupBox:
        box = QGroupBox(title)
        box.setLayout(layout)
        return box

    @staticmethod
    def hint(text: str) -> QLabel:
        lab = QLabel(text)
        lab.setWordWrap(True)
        lab.setStyleSheet("color:#5b6b7f;font-size:12px;")
        return lab

    # -- 运行状态 --
    @property
    def busy(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    def set_running(self, running: bool) -> None:
        """切换界面到"运行中/空闲"。子类可重写以额外禁用某些按钮。"""
        for w in self.findChildren(QWidget):
            if getattr(w, "_always_enabled", False):
                continue
            if w.__class__.__name__ in ("QPushButton", "QComboBox", "QSpinBox",
                                        "QDoubleSpinBox", "QLineEdit"):
                w.setEnabled(not running)

    def start_worker(self, worker, on_ok, on_fail=None, label: str = "任务"):
        """启动后台线程并接好所有信号。

        ``on_ok(result)`` 在**主线程**里被调用（Qt 信号跨线程自动排队），
        所以可以放心在里面更新界面。
        """
        if self.busy:
            self.log_line("【提示】已有任务在运行，请等它结束")
            return
        self.worker = worker
        self.set_running(True)

        worker.progress.connect(self._on_progress)
        worker.log.connect(self.log_line)
        worker.finished_ok.connect(lambda r: self._on_done(r, on_ok))
        worker.failed.connect(lambda n, d: self._on_fail(n, d, on_fail, label))
        worker.finished.connect(lambda: self.set_running(False))
        worker.start()

    def _on_progress(self, pct: int, text: str) -> None:
        bar = getattr(self, "progress", None)
        if bar is not None:
            bar.setValue(max(0, min(100, int(pct))))
        if text:
            self.set_status(text)

    def _on_done(self, result, on_ok) -> None:
        self.worker = None
        try:
            on_ok(result)
        except Exception as exc:                # noqa: BLE001 — 界面回调出错要看得见
            import traceback
            self.log_line(f"【错误】处理结果时出错：{exc}")
            self.log_line(traceback.format_exc())

    def _on_fail(self, name, detail, on_fail, label) -> None:
        self.worker = None
        self.log_line(f"【{label}失败】{name}: {detail.splitlines()[0]}")
        if on_fail is not None:
            on_fail(name, detail)
        else:
            from PyQt5.QtWidgets import QMessageBox
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle(f"{label}失败")
            box.setText(f"{name}: {detail.splitlines()[0]}")
            box.setDetailedText(detail)
            box.exec_()

    # -- 子类通常要覆盖 --
    def log_line(self, text: str) -> None:
        box = getattr(self, "log", None)
        if box is not None:
            box.append_line(text)

    def set_status(self, text: str) -> None:
        lab = getattr(self, "status", None)
        if lab is not None:
            lab.setText(text)

    def on_enter(self) -> None:
        """切到本页时调用（主窗口触发）。默认什么都不做。"""

    def on_dataset_changed(self) -> None:
        """数据集换了以后刷新本页内容（主窗口广播）。"""
