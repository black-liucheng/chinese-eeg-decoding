# -*- coding: utf-8 -*-
"""
后台任务线程
============

界面卡死是实验软件最常见的毛病 —— 训练一个 CNN 要几分钟，如果放在主线程里，
窗口会变成"未响应"，演示的时候非常难看。

这里把所有耗时操作都包成 ``QThread`` 子类，通过 Qt 信号把进度和结果回传主线程：

- 信号只能由主线程更新界面（Qt 的硬性规定），所以 worker 里**绝不碰任何控件**。
- 一个 worker 同一时刻只跑一件事；``run()`` 结束就发 ``finished`` 让界面恢复按钮。

.. note::
   PyTorch 本身会用满所有 CPU 核心，再叠一个 QThread 并不会更快，但也不会更慢
   —— 这里用线程主要是为了**不阻塞界面**，不是为了并行。
"""

from __future__ import annotations

import time
import traceback
from typing import Any, Dict, List, Optional

from PyQt5.QtCore import QThread, pyqtSignal

from ..config import Config


class Worker(QThread):
    """通用后台任务线程。

    子类只需要实现 :meth:`work` 并把结果 ``return`` 出来，异常会自动捕获并
    通过 ``failed`` 信号回传 —— 不让后台线程里的异常静默吞掉，否则界面会
    一直停在"运行中"不动，用户完全不知道出了什么事。
    """

    #: 进度：(百分比 0-100, 说明文字)
    progress = pyqtSignal(int, str)
    #: 阶段性日志文字
    log = pyqtSignal(str)
    #: 成功完成，携带 work() 的返回值
    finished_ok = pyqtSignal(object)
    #: 失败，携带 (异常名, 详细信息) —— 详细信息含完整 traceback，方便排查
    failed = pyqtSignal(str, str)

    def __init__(self, cfg: Config, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self._cancelled = False

    # -- 子类实现 --
    def work(self) -> Any:
        raise NotImplementedError

    def cancel(self) -> None:
        """请求取消。任务在下一个检查点自行退出（不会强杀线程）。"""
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def check_cancel(self) -> None:
        """在耗时循环里周期性调用，被取消时抛异常结束任务。"""
        if self._cancelled:
            raise InterruptedError("用户取消了任务")

    # -- 运行 --
    def run(self) -> None:                        # noqa: D102 — Qt 约定
        try:
            result = self.work()
        except InterruptedError as exc:
            self.failed.emit("已取消", str(exc))
        except Exception as exc:                  # noqa: BLE001 — 必须兜住，否则线程静默死掉
            self.failed.emit(type(exc).__name__,
                             f"{exc}\n\n{traceback.format_exc()}")
        else:
            self.finished_ok.emit(result)


# ---------------------------------------------------------------------------
#  具体任务
# ---------------------------------------------------------------------------


class GenerateWorker(Worker):
    """生成（仿真）数据集 + 预处理。"""

    def work(self):
        from ..data.epoching import preprocess_dataset
        from ..data.synthetic import generate_dataset

        cfg = self.cfg
        self.progress.emit(5, "正在生成仿真脑电信号 ...")
        t0 = time.time()
        ds_raw = generate_dataset(cfg)
        self.check_cancel()
        self.log.emit(f"生成完成：{ds_raw.n_trials} 试次 × {ds_raw.n_channels} 通道 "
                      f"× {ds_raw.n_times} 采样点（{time.time() - t0:.1f} s）")

        self.progress.emit(55, "正在预处理（滤波 / 重参考 / 去伪迹 / 标准化）...")
        # preprocess_dataset 返回新对象、不改原对象，所以 ds_raw 仍是预处理前的数据，
        # 预处理页拿它做"前后对比"。不需要额外深拷贝。
        ds, report = preprocess_dataset(ds_raw, cfg)
        self.check_cancel()
        self.progress.emit(100, "完成")
        return ds, report, ds_raw


class LoadFileWorker(Worker):
    """读取真实脑电文件（MNE）并预处理。"""

    def __init__(self, cfg: Config, path: str, parent=None):
        super().__init__(cfg, parent)
        self.path = path

    def work(self):
        from ..data.epoching import preprocess_raw
        from ..data.loaders import load_recording

        self.progress.emit(10, "正在读取脑电文件 ...")
        rec = load_recording(self.path, self.cfg)
        self.log.emit(rec.summary())
        self.check_cancel()

        self.progress.emit(60, "正在预处理 ...")
        ds, report = preprocess_raw(
            rec.data, rec.sfreq, rec.events, rec.event_codes, rec.ch_names,
            rec.class_names, self.cfg, rec.is_synthetic, rec.meta)
        self.progress.emit(100, "完成")
        return ds, report, None      # 真实数据的分段结果即是"预处理后"，无独立原始集


class TrainWorker(Worker):
    """训练与评估一个模型（SVM 或 CNN），可选做置换检验。"""

    def __init__(self, cfg: Config, ds, model_name: str,
                 group_by_subject: bool = False, permutation: bool = False,
                 parent=None):
        super().__init__(cfg, parent)
        self.ds = ds
        self.model_name = model_name
        self.group_by_subject = group_by_subject
        self.do_permutation = permutation
        self.model_path: Optional[str] = None

    def work(self):
        from ..models import build_model
        from ..train import cross_validate, permutation_test

        cfg, ds = self.cfg, self.ds
        t0 = time.time()

        def _cb(fold, n_splits, epoch, total, f1):
            self.check_cancel()
            if fold == 0:                          # 折跑完了，进入部署模型训练
                # 这一步在 CNN 上要跑几分钟。不给提示的话进度条会停在
                # "第 N/N 折完成" 一动不动，看起来像卡死了。
                pct = 80 if self.do_permutation else 100
                self.progress.emit(pct, "交叉验证完成，正在训练部署模型 ...")
            elif epoch is None:                    # 一折结束
                pct = int(100 * fold / max(n_splits, 1) * (0.8 if self.do_permutation else 1.0))
                self.progress.emit(pct, f"第 {fold}/{n_splits} 折完成")
            else:                                  # CNN 的一个 epoch
                frac = fold - 1 + epoch / max(total, 1)
                pct = int(100 * frac / max(n_splits, 1) * (0.8 if self.do_permutation else 1.0))
                self.progress.emit(pct, f"第 {fold} 折　轮 {epoch}/{total}　"
                                        f"验证宏F1={f1:.3f}")

        self.progress.emit(1, f"开始训练 {self.model_name.upper()} ...")
        result = cross_validate(cfg, ds, model_name=self.model_name,
                                group_by_subject=self.group_by_subject,
                                progress_callback=_cb)
        self.log.emit(result.summary())

        if self.do_permutation and not self.cancelled:
            n = cfg.train.n_permutations
            self.progress.emit(82, f"置换检验（{n} 次打乱标签）...")
            perm = permutation_test(
                cfg, ds, model_name=self.model_name,
                observed=result.accuracy, n_permutations=n,
                progress_callback=lambda i, tot: (
                    self.check_cancel(),
                    self.progress.emit(82 + int(18 * i / max(tot, 1)),
                                       f"置换检验 {i}/{tot}"),
                ))
            result.permutation = perm
            self.log.emit(perm.summary())

        self.progress.emit(100, f"完成，耗时 {time.time() - t0:.1f} s")
        return result


class PermutationWorker(Worker):
    """单独跑置换检验（对已经训练好的结果补做）。"""

    def __init__(self, cfg: Config, ds, model_name: str, observed: float,
                 n_permutations: Optional[int] = None, parent=None):
        super().__init__(cfg, parent)
        self.ds = ds
        self.model_name = model_name
        self.observed = observed
        self.n_permutations = n_permutations or cfg.train.n_permutations

    def work(self):
        from ..train import permutation_test

        return permutation_test(
            self.cfg, self.ds, model_name=self.model_name,
            observed=self.observed, n_permutations=self.n_permutations,
            progress_callback=lambda i, tot: (
                self.check_cancel(),
                self.progress.emit(int(100 * i / max(tot, 1)),
                                   f"置换检验 {i}/{tot}"),
            ))


class PredictWorker(Worker):
    """对一段记录做逐试次预测，返回每个试次的类别概率。"""

    def __init__(self, cfg: Config, model, class_names: List[str], ds,
                 parent=None):
        super().__init__(cfg, parent)
        self.model = model
        self.class_names = class_names
        self.ds = ds

    def work(self):
        import numpy as np

        self.progress.emit(20, "正在计算 ...")
        proba = self.model.predict_proba(self.ds.X)
        pred = np.asarray(proba).argmax(1)
        self.progress.emit(100, "完成")
        return {
            "proba": np.asarray(proba),
            "pred": pred,
            "y_true": self.ds.y,
            "class_names": list(self.class_names),
            "accuracy": float((pred == self.ds.y).mean()) if len(self.ds.y) else float("nan"),
        }


class SNRSweepWorker(Worker):
    """信噪比扫描 —— 找出"多难的数据下 CNN 才跑赢 SVM"。"""

    def __init__(self, cfg: Config, snrs: List[float], parent=None):
        super().__init__(cfg, parent)
        self.snrs = snrs

    def work(self):
        from ..data.epoching import preprocess_dataset
        from ..data.synthetic import generate_dataset
        from ..train import holdout_evaluate

        import copy

        rows: List[Dict[str, float]] = []
        for i, snr in enumerate(self.snrs):
            self.check_cancel()
            c = copy.deepcopy(self.cfg)
            c.data.synthetic.snr_db = float(snr)
            self.progress.emit(int(100 * i / len(self.snrs)),
                               f"SNR {snr:+.0f} dB：生成数据 ...")
            ds = generate_dataset(c)
            ds, _ = preprocess_dataset(ds, c)
            row = {"snr": float(snr)}
            for name in ("svm", "cnn"):
                self.check_cancel()
                self.progress.emit(
                    int(100 * (i + 0.5) / len(self.snrs)),
                    f"SNR {snr:+.0f} dB：训练 {name.upper()} ...")
                c.model.name = name
                row[name] = holdout_evaluate(c, ds, model_name=name).accuracy
            rows.append(row)
            self.log.emit(f"SNR {snr:+.0f} dB　SVM {row['svm']:.1%}　"
                          f"CNN {row['cnn']:.1%}")
        self.progress.emit(100, "完成")
        return rows
