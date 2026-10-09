# -*- coding: utf-8 -*-
"""离屏构建主窗口，抓取五个标签页的截图 → data/screenshots/。

跑法（项目根目录下）：

    python scripts/shoot_gui.py                 # 快速自检：只截空界面（几秒）
    python scripts/shoot_gui.py --populate      # ★ 答辩用：跑真实流程，界面里有数据
    python scripts/shoot_gui.py --populate --with-cnn   # 额外训 CNN，多出一张模型对比图

用途有三个：

1. **开发自检**（不带参数）：确认改动没把界面弄坏，几秒出图。
2. **答辩截图**（``--populate``）：这是默认该用的模式。它把界面当成一个真人来用 ——
   生成数据 → 预览预处理 → 训练 SVM → 看评估页 → 推理一条试次 ——
   所以截出来的每张图里都有真实内容（波形、混淆矩阵、指标表、拓扑图）。
   不这么做的话，截出来的是一堆空画布：界面确实没坏，但答辩现场一旦要拿截图
   顶替演示，那些空图是没用的。
3. **两种模式截的图同名同路径**，后跑的覆盖先跑的，不会堆出一堆文件。

**用 ``WA_DontShowOnScreen`` 而不是 ``QT_QPA_PLATFORM=offscreen``** ——
后者在 Windows 上不加载字体，截出来的图里所有中文都是空白，
看起来像界面彻底坏了，其实是截图方式的问题，排查起来很费时间。

退出码非 0 表示有页面构建失败或流程没跑完（主窗口会为失败页生成占位页，程序仍能启动）。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PyQt5.QtCore import Qt                                  # noqa: E402
from PyQt5.QtWidgets import QApplication                     # noqa: E402

from cneeg.config import load_config                         # noqa: E402
from cneeg.gui.main_window import PAGE_ORDER, MainWindow     # noqa: E402


def _robust_console() -> None:
    """控制台遇到打不出的字符时用 ``?`` 顶替，而不是抛异常终止。

    Chinese Windows 的控制台是 GBK；脚本里出现 GBK 之外的字符（数学减号、
    对勾等）会直接 ``UnicodeEncodeError``。本项目已经在这上面栽过两次。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def _pump(app: QApplication, seconds: float = 0.05) -> None:
    """跑一小段事件循环，让 Qt 把排队的信号和重绘处理掉。"""
    end = time.monotonic() + seconds
    while True:
        app.processEvents()
        if time.monotonic() >= end:
            return
        time.sleep(0.01)


def _wait_idle(app: QApplication, page, timeout: float, label: str) -> bool:
    """等页面的后台线程跑完。

    ``page.busy`` 由 ``PageBase`` 依据 ``worker.isRunning()`` 算出来，
    而 ``worker`` 在完成回调里被置回 ``None`` —— 所以这里等的是一个
    **状态**，不是某个具体线程对象，不会因为中途换过 worker 而误判。
    """
    end = time.monotonic() + timeout
    t0 = time.monotonic()
    while time.monotonic() < end:
        app.processEvents()
        if not page.busy:
            print(f"    {label} 完成，耗时 {time.monotonic() - t0:.1f} s")
            return True
        time.sleep(0.02)
    print(f"    [警告] {label} 超过 {timeout:.0f} s 仍未结束，放弃等待")
    return False


def _populate(app: QApplication, win: MainWindow, with_cnn: bool) -> list[str]:
    """把界面当成真人来用一遍，好让截图里有真实内容。返回失败的步骤名。"""
    failed: list[str] = []

    # ① 数据：真的生成一份（会走后台线程）
    print("  [1/5] 生成数据 …")
    dp = win.page("data")
    win.goto("data")
    dp.generate()
    if not _wait_idle(app, dp, 900, "数据生成"):
        failed.append("data")
    _pump(app, 0.3)

    # ② 预处理：数据变了的广播已经触发过一次预览，这里只在它没跑时补一次
    print("  [2/5] 预处理预览 …")
    pp = win.page("preprocess")
    win.goto("preprocess")
    if pp.ds is None:
        pp.apply_preprocess(broadcast=False)
    _pump(app, 0.3)

    # ③ 训练：SVM 快（约 1 分钟），CNN 慢（数分钟，默认不跑）
    print("  [3/5] 训练 SVM …")
    tp = win.page("train")
    win.goto("train")
    tp.model_combo.setCurrentIndex(max(tp.model_combo.findData("svm"), 0))
    tp.start_training()
    if not _wait_idle(app, tp, 1800, "SVM 训练"):
        failed.append("train")
    if with_cnn:
        print("  [3/5] 训练 CNN（较慢）…")
        tp.model_combo.setCurrentIndex(max(tp.model_combo.findData("cnn"), 0))
        tp.start_training()
        if not _wait_idle(app, tp, 5400, "CNN 训练"):
            failed.append("train-cnn")
    _pump(app, 0.3)

    # ④ 评估：训完的结果已由主窗口转送到本页并自动画好
    print("  [4/5] 评估页 …")
    ev = win.page("evaluate")
    win.goto("evaluate")
    if with_cnn and len(ev.results) >= 2:
        ev.view_combo.setCurrentIndex(max(ev.view_combo.findData("compare"), 0))
    ev.refresh()
    _pump(app, 0.4)

    # ⑤ 推理：模型已由主窗口送到本页，预测当前数据集
    print("  [5/5] 推理页 …")
    ip = win.page("infer")
    win.goto("infer")
    if getattr(ip, "model", None) is None:
        print("    [警告] 推理页没收到模型，跳过预测")
        failed.append("infer-model")
    else:
        ip.run_predict()
        _pump(app, 0.5)

    return failed


def main() -> int:
    _robust_console()
    ap = argparse.ArgumentParser(description="抓取界面截图")
    ap.add_argument("--populate", action="store_true",
                    help="跑真实流程后再截图（答辩用；约 1.5 分钟）")
    ap.add_argument("--with-cnn", action="store_true",
                    help="连 CNN 一起训，多出一张模型对比图（另加数分钟）")
    args = ap.parse_args()
    if args.with_cnn and not args.populate:
        print("[提示] --with-cnn 只在 --populate 下有意义，已自动打开 --populate")
        args.populate = True

    app = QApplication(sys.argv)
    cfg = load_config()

    win = MainWindow(cfg)
    win.setAttribute(Qt.WA_DontShowOnScreen, True)
    win.resize(1400, 900)
    win.show()
    for _ in range(8):
        app.processEvents()

    failed: list[str] = []
    if args.populate:
        print("跑真实流程（这一步会真的训练模型）…")
        failed += _populate(app, win, args.with_cnn)

    out = cfg.sub_dir("screenshots")
    broken: list[str] = []
    for i, (key, cls) in enumerate(PAGE_ORDER, 1):
        win.goto(key)
        for _ in range(5):
            app.processEvents()

        page = win.page(key)
        # 主窗口在页面构造失败时会塞一个普通 QWidget 占位 —— 那就是坏了
        if page.__class__.__name__ == "QWidget" and cls.__name__ != "QWidget":
            broken.append(key)
            print(f"  [失败] {key} 是错误占位页")
            continue

        ok = page.grab().save(str(out / f"{i}_{key}.png"))
        print(f"  [{'ok' if ok else '失败'}] {key:11s} -> {i}_{key}.png  ({cls.__name__})")

    print(f"\n输出目录：{out}")
    print("模式：", "含真实内容（--populate）" if args.populate else "空界面自检")
    print("结果：", "全部正常" if not broken and not failed
          else f"有问题的页：{broken or '无'}；没跑完的步骤：{failed or '无'}")
    return 1 if (broken or failed) else 0


if __name__ == "__main__":
    sys.exit(main())
