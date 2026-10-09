# -*- coding: utf-8 -*-
"""
主窗口
======

把五个标签页串成一个程序。除了搭界面，它还负责页面之间的**数据流**：

- 数据页生成/导入数据 → ``broadcast_dataset_changed()`` 通知所有页刷新
- 训练页训完一个模型 → ``broadcast_result()`` 把结果送到评估页、把模型送到推理页

页面之间不互相直接引用（除了通过 ``page(key)`` 取当前数据），全部走主窗口转发，
避免出现"改了一个页、另一页悄悄失效"的情况。
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import fields

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QAction, QApplication, QLabel, QMainWindow,
                             QMessageBox, QStatusBar, QTabWidget, QVBoxLayout,
                             QWidget)

from ..config import Config, load_config
from .pages.data_page import DataPage
from .pages.evaluate_page import EvaluatePage
from .pages.infer_page import InferPage
from .pages.preprocess_page import PreprocessPage
from .pages.train_page import TrainPage

#: 标签页顺序即使用顺序 —— 界面上也是按这个顺序从左到右排的
PAGE_ORDER = [
    ("data", DataPage),
    ("preprocess", PreprocessPage),
    ("train", TrainPage),
    ("evaluate", EvaluatePage),
    ("infer", InferPage),
]

_STYLE = """
QMainWindow, QWidget { background: #f6f8fb; color: #1c2431; }
QTabWidget::pane { border: 1px solid #dde5ee; border-radius: 6px;
                   background: #ffffff; top: -1px; }
QTabBar::tab { background: #e9eff6; color: #43536a; padding: 8px 20px;
               margin-right: 2px; border-top-left-radius: 6px;
               border-top-right-radius: 6px; font-size: 13px; }
QTabBar::tab:selected { background: #ffffff; color: #2b5d8a;
                        font-weight: bold; border-bottom: 2px solid #2b5d8a; }
QTabBar::tab:hover:!selected { background: #dfe8f2; }
QPushButton { background: #ffffff; border: 1px solid #c3d2e2; border-radius: 5px;
              padding: 5px 14px; font-size: 13px; }
QPushButton:hover { background: #eef4fb; border-color: #7ba7cd; }
QPushButton:default { background: #2b5d8a; color: #ffffff;
                      border: 1px solid #2b5d8a; font-weight: bold; }
QPushButton:default:hover { background: #36719f; }
QPushButton:disabled { background: #f0f2f5; color: #a8b2bf;
                       border-color: #dde5ee; }
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
    background: #ffffff; border: 1px solid #c3d2e2; border-radius: 4px;
    padding: 4px 6px; font-size: 13px; }
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {
    border-color: #2b5d8a; }
QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled,
QLineEdit:disabled { background: #f0f2f5; color: #a8b2bf; }
QProgressBar { border: 1px solid #dde5ee; border-radius: 5px;
               background: #f0f2f5; text-align: center; height: 20px;
               font-size: 12px; }
QProgressBar::chunk { background: #4a90c4; border-radius: 4px; }
QTableWidget { background: #ffffff; border: 1px solid #dde5ee;
               border-radius: 6px; gridline-color: #eef2f7;
               font-size: 12px; }
QHeaderView::section { background: #eef4fb; border: none;
                       border-bottom: 1px solid #dde5ee; padding: 5px;
                       font-size: 12px; color: #43536a; }
QGroupBox { border: 1px solid #dde5ee; border-radius: 6px; margin-top: 10px;
            padding-top: 8px; font-size: 13px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px;
                   color: #2b5d8a; font-weight: bold; }
QSplitter::handle { background: #e4eaf2; }
"""


class MainWindow(QMainWindow):
    """主窗口 —— 五个标签页 + 跨页数据流。"""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.pages = {}

        self.setWindowTitle("汉语脑电信号分类系统　v1.0")
        self.resize(1280, 880)
        self.setStyleSheet(_STYLE)

        central = QWidget()
        lay = QVBoxLayout(central)
        lay.setContentsMargins(10, 8, 10, 6)
        lay.setSpacing(6)

        banner = QLabel(
            "北京交通大学 大学生创新训练计划　·　基于深度学习的汉语脑电信号建模与分析")
        banner.setStyleSheet("color:#5b6b7f;font-size:12px;padding:2px 4px;")
        lay.addWidget(banner)

        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self._on_tab_changed)
        for key, cls in PAGE_ORDER:
            try:
                page = cls(cfg)
            except Exception as exc:            # noqa: BLE001 — 单页崩了别拖垮整个程序
                page = self._error_page(key, cls, exc)
            self.pages[key] = page
            self.tabs.addTab(page, getattr(page, "title", key))
        lay.addWidget(self.tabs, stretch=1)
        self.setCentralWidget(central)

        self.setStatusBar(QStatusBar())
        self._build_menu()

        data_summary = self._data_banner()
        self.statusBar().showMessage(data_summary)

    # ------------------------------------------------------------------
    #  构建
    # ------------------------------------------------------------------

    def _error_page(self, key: str, cls, exc: Exception):
        """某一页构造失败时，用一页错误说明顶替，而不是让程序起不来。"""
        w = QWidget()
        lay = QVBoxLayout(w)
        lab = QLabel(f"页面“{cls.title}”加载失败：\n\n{type(exc).__name__}: {exc}"
                     f"\n\n{traceback.format_exc()}")
        lab.setWordWrap(True)
        lab.setStyleSheet("color:#b03030;font-size:12px;")
        lay.addWidget(lab)
        w.title = getattr(cls, "title", key)
        return w

    def _build_menu(self) -> None:
        m = self.menuBar()

        file_menu = m.addMenu("文件(&F)")
        act = QAction("重新加载配置(&R)", self)
        act.triggered.connect(self._reload_cfg)
        file_menu.addAction(act)
        file_menu.addSeparator()
        act = QAction("退出(&Q)", self)
        act.setShortcut("Ctrl+Q")
        act.triggered.connect(self.close)
        file_menu.addAction(act)

        run_menu = m.addMenu("运行(&R)")
        for key, label in (("data", "① 数据"),
                           ("preprocess", "② 预处理"),
                           ("train", "③ 训练"),
                           ("evaluate", "④ 评估"),
                           ("infer", "⑤ 推理")):
            act = QAction(label, self)
            act.triggered.connect(lambda _c, k=key: self.goto(k))
            run_menu.addAction(act)

        help_menu = m.addMenu("帮助(&H)")
        act = QAction("关于(&A)", self)
        act.triggered.connect(self._about)
        help_menu.addAction(act)

    def _data_banner(self) -> str:
        if getattr(self.cfg.data, "source", "synthetic") == "synthetic":
            return ("数据源：仿真数据（方法学验证用）　·　"
                    "真实数据接入后改 configs/default.yaml 的 data.source 即可")
        return f"数据源：真实脑电文件　{self.cfg.data.file.path}"

    def _reload_cfg(self) -> None:
        """从 yaml 重读配置。

        **就地更新** ``self.cfg`` 的字段而不是替换这个对象 —— 所有页面在构造时
        都持有同一个 cfg 引用，替换对象会让它们继续读旧配置。
        """
        try:
            fresh = load_config()
        except Exception as exc:                # noqa: BLE001
            QMessageBox.warning(self, "重载失败", str(exc))
            return
        for f in fields(self.cfg):
            setattr(self.cfg, f.name, getattr(fresh, f.name))
        self.statusBar().showMessage(
            "配置已重新加载 —— 已打开的页面里控件上的值不会自动同步，"
            "需要手动改回或重启程序")

    def _about(self) -> None:
        QMessageBox.information(
            self, "关于",
            "汉语脑电信号分类系统　v1.0\n\n"
            "基于深度学习的汉语脑电信号建模与分析\n"
            "北京交通大学 大学生创新训练计划\n\n"
            "模型：STFT-SVM 基线　/　ShallowConvNet（Schirrmeister et al., 2017）\n"
            f"数据源：{'仿真数据（方法学验证）' if self.cfg.data.source == 'synthetic' else '真实脑电'}\n\n"
            "【注意】仿真数据仅用于验证软件流水线与模型对比方法，\n"
            "不构成真实汉语脑电的实验结论。")

    # ------------------------------------------------------------------
    #  页面访问
    # ------------------------------------------------------------------

    def page(self, key: str):
        """按 key 取页面 —— ``"data"`` / ``"preprocess"`` / ``"train"`` …"""
        return self.pages.get(key)

    def goto(self, key: str) -> None:
        page = self.pages.get(key)
        if page is not None:
            self.tabs.setCurrentWidget(page)

    def _on_tab_changed(self, idx: int) -> None:
        w = self.tabs.widget(idx)
        hook = getattr(w, "on_enter", None)
        if callable(hook):
            try:
                hook()
            except Exception as exc:            # noqa: BLE001
                # 页面的 on_enter 出错不该让切换标签失败
                print(f"[gui] {getattr(w, 'title', '?')} on_enter 出错: {exc}",
                      file=sys.stderr)

    # ------------------------------------------------------------------
    #  跨页数据流
    # ------------------------------------------------------------------

    def broadcast_dataset_changed(self) -> None:
        """数据集变了 —— 通知所有页（发起页自己会收到，需自行幂等）。"""
        for key, page in self.pages.items():
            hook = getattr(page, "on_dataset_changed", None)
            if callable(hook):
                try:
                    hook()
                except Exception as exc:        # noqa: BLE001
                    self.statusBar().showMessage(
                        f"{getattr(page, 'title', key)} 刷新失败：{exc}")

    def broadcast_result(self, result) -> None:
        """一次训练完成 —— 结果送评估页，训练好的模型送推理页。"""
        ev = self.pages.get("evaluate")
        if ev is not None and hasattr(ev, "add_result"):
            try:
                ev.add_result(result)
            except Exception as exc:            # noqa: BLE001
                self.statusBar().showMessage(f"评估页接收结果失败：{exc}")

        model = getattr(result, "full_model", None)
        inf = self.pages.get("infer")
        if inf is not None and model is not None and hasattr(inf, "set_model"):
            try:
                inf.set_model(model, result.class_names)
            except Exception as exc:            # noqa: BLE001
                self.statusBar().showMessage(f"推理页接收模型失败：{exc}")

    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:        # noqa: N802 — Qt 约定
        """退出前确认没有后台任务在跑 —— 强行退出可能留下半个模型文件。"""
        busy = [getattr(p, "title", k) for k, p in self.pages.items()
                if getattr(p, "busy", False)]
        if busy:
            r = QMessageBox.question(
                self, "还有任务在运行",
                "以下页面仍有任务在运行：\n　" + "\n　".join(busy) +
                "\n\n现在退出会中断它们。确定退出吗？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if r != QMessageBox.Yes:
                event.ignore()
                return
            for p in self.pages.values():
                w = getattr(p, "worker", None)
                if w is not None:
                    w.cancel()
                    w.wait(3000)
        event.accept()


def run(cfg: Config) -> int:
    """启动 GUI（``cli.cmd_gui`` 调这个）。"""
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("汉语脑电信号分类系统")

    win = MainWindow(cfg)
    win.show()
    return app.exec_()
