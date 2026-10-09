# -*- coding: utf-8 -*-
"""
命令行入口
==========

让整条流水线不用点界面也能一键复现 —— 答辩现场演示、老师验收、
或者你自己批量跑实验时用它。

.. code-block:: text

    python -m cneeg gen-data                          # 生成仿真数据
    python -m cneeg train  --model svm                # 训练 + 交叉验证
    python -m cneeg train  --model cnn --group        # 留一受试者评估
    python -m cneeg compare                           # SVM vs CNN 对比
    python -m cneeg sweep                             # 信噪比扫描
    python -m cneeg report                            # 生成 HTML 报告
    python -m cneeg infer  --file 被试01.mff          # 用真实数据推理
    python -m cneeg gui                               # 打开图形界面

所有子命令都支持 ``--set a.b.c=value`` 覆盖配置，例如::

    python -m cneeg train --model cnn --set model.cnn.n_epochs=120
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .config import PRESET_LABELS, PRESETS, Config, dump_config, load_config

# ---------------------------------------------------------------------------
#  小工具
# ---------------------------------------------------------------------------


def _parse_overrides(pairs) -> dict:
    """``["model.cnn.lr=0.001", "seed=7"]`` → ``{"model.cnn.lr": 0.001, ...}``"""
    out: dict = {}
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"--set 参数格式应为 键=值，收到：{item!r}")
        k, v = item.split("=", 1)
        k, v = k.strip(), v.strip()
        # 按 YAML 标量解析，这样 0.5 / true / null / [1,2] 都能用
        import yaml
        try:
            out[k] = yaml.safe_load(v)
        except Exception:
            out[k] = v
    return out


def _banner(title: str) -> None:
    print()
    print("=" * 68)
    print(f"  {title}")
    print("=" * 68)


def _is_real_source(cfg: Config) -> bool:
    return cfg.data.source in ("file", "real", "mff")


def _load_dataset(cfg: Config, path: str | None = None):
    """按配置取数据 —— 仿真走生成器，真实走 MNE 适配器，之后共用同样的预处理。

    两条路都**必定**做预处理：下游（交叉验证、特征、画图、报告）一律拿处理后的
    数据，没有哪个调用方想要未处理的数据。这里原先有个 ``preprocess`` 参数，
    但从来没有人传过 ``False``，末尾还跟着一句 "if not preprocess" 的分支 ——
    两个分支返回的东西一模一样。留着一个不起作用的开关，只会让人以为存在
    这种用法，所以删掉。
    """
    from .data.epoching import preprocess_dataset

    report = None
    if path or _is_real_source(cfg):
        from .data.loaders import load_any

        src = path or cfg.data.file.path
        if not src:
            raise SystemExit(
                "配置里 data.source 是真实数据，但没有给出文件路径。\n"
                "  请用 --file 指定，或设置 data.file.path。"
            )
        print(f"[数据] 读取真实记录：{src}")
        rec = load_any(src, cfg)
        print(rec.summary())
        from .data.epoching import preprocess_raw

        ds, report = preprocess_raw(
            rec.data, rec.sfreq, rec.events, rec.event_codes, rec.ch_names,
            rec.class_names, cfg, rec.is_synthetic, rec.meta,
        )
    else:
        from .data.synthetic import generate_dataset

        print(f"[数据] 生成仿真数据（预设 {cfg.data.synthetic.preset}，"
              f"SNR {cfg.data.synthetic.snr_db:+.1f} dB，"
              f"种子 {cfg.seed}）")
        ds = generate_dataset(cfg)
        ds, report = preprocess_dataset(ds, cfg)

    return ds, report


def _save_dataset(ds, cfg: Config, tag: str = "dataset") -> Path:
    """存数据集；内容与已有文件完全相同时直接复用，不重复占盘。

    种子固定，同样的配置重跑会生成逐字节相同的 47 MB 文件。而一键脚本
    每次都先跑 gen-data —— 不查重的话，跑三次就躺三份一模一样的数据。
    """
    from .data.base import EEGDataset

    out_dir = cfg.sub_dir("processed")
    fp = ds.fingerprint()
    candidates = sorted(out_dir.glob(f"{tag}_*.npz"))

    def _reuse(old: Path) -> Path:
        print(f"[跳过] 已有内容完全相同的数据集，直接复用：{old.name}")
        print("       （同样配置重跑不会重复占盘，也不会重复同步到 OneDrive）")
        return old

    # 第一遍走便宜的路：只读内嵌指纹，不解压波形。正常情况（文件是现版本写的）
    # 在这一遍就命中并返回。
    for old in candidates:
        if EEGDataset.stored_fingerprint(old) == fp:
            return _reuse(old)

    # 第二遍处理老文件：它们生成于 fingerprint 字段存在之前，内嵌指纹读出来是
    # 空串，而空串永远不等于真指纹 —— 查重会**静默失效**，每次 gen-data 都再写
    # 一份一模一样的 47 MB，正是本函数要防的事（磁盘上曾经就因此躺了两份）。
    # 所以对这种文件退回去按内容算一次指纹，代价是解压一遍波形（47 MB 约 1 秒）。
    for old in candidates:
        if EEGDataset.stored_fingerprint(old):
            continue                               # 上面已经比过，别再解压
        try:
            if EEGDataset.load(old).fingerprint() == fp:
                return _reuse(old)
        except Exception:                          # noqa: BLE001
            continue                               # 读不动的文件不参与比对

    out = out_dir / f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}.npz"
    ds.save(out)
    return out


def _load_saved_dataset(cfg: Config, path: str | None):
    """``--data`` 指定了就用缓存的 npz，否则现生成。"""
    from .data.base import EEGDataset

    if path:
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"找不到数据文件：{p}")
        print(f"[数据] 从缓存载入：{p}")
        return EEGDataset.load(p), None
    return _load_dataset(cfg)


# ---------------------------------------------------------------------------
#  子命令
# ---------------------------------------------------------------------------


def cmd_gen_data(args, cfg: Config) -> int:
    _banner("生成数据集")
    ds, rep = _load_dataset(cfg)
    print()
    print(ds.summary())
    if rep is not None:
        print()
        print(rep.summary())
    out = _save_dataset(ds, cfg, "dataset")
    print(f"\n[完成] 数据已保存：{out}")
    print(f"       体积 {out.stat().st_size / 1e6:.1f} MB")

    # 顺手出几张检查图，方便肉眼确认数据长得对不对
    from . import plots

    figdir = cfg.sub_dir("figures")
    plots.save(plots.plot_waveform(ds), figdir / "01_原始波形.png")
    plots.save(plots.plot_erp(ds), figdir / "02_各类ERP.png")
    plots.save(plots.plot_discriminability(ds), figdir / "03_类别可分性拓扑.png")
    print(f"[完成] 检查图已保存到：{figdir}")
    return 0


def _save_model(cfg: Config, res) -> Path | None:
    """把交叉验证后在全量数据上重训的模型存成 joblib，供 ``infer`` 使用。

    存的是 ``res.full_model``：它见过全部数据，**不参与任何评估指标**，
    正因如此才适合拿去部署推理（见 train.py 里那段注释）。

    若 ``full_model`` 没能训出来，这里明确说一声而不是写个空文件 ——
    ``infer`` 拿到坏文件要到预测那一刻才炸，报错位置离真正的原因很远。
    """
    model = getattr(res, "full_model", None)
    if model is None:
        print("[警告] 本次训练没有产出可供推理的模型，未保存。\n"
              "        训练过程中的各项指标仍然有效，报告照常可看。")
        return None

    import joblib

    out = cfg.sub_dir("models") / "model.joblib"
    joblib.dump({
        "model": model,
        "class_names": list(res.class_names),
        "model_name": res.model_name,
        "is_synthetic": bool(getattr(res, "is_synthetic", True)),
    }, out)
    print(f"\n[完成] 模型已保存：{out}")
    print(f"       体积 {out.stat().st_size / 1e6:.1f} MB")
    print("       下一步：python -m cneeg infer --file <脑电文件>")
    return out


def cmd_train(args, cfg: Config) -> int:
    model = args.model or cfg.model.name
    group = args.group
    _banner(f"训练与评估 —— {model.upper()}"
            f"{'（留一受试者）' if group else '（分层交叉验证）'}")

    from .train import cross_validate

    ds, rep = _load_saved_dataset(cfg, args.data)
    print(f"\n[数据] {ds.n_trials} 试次 × {ds.n_channels} 通道 × {ds.n_times} 采样点"
          f"　类别 {ds.class_names}")

    last = {"fold": 0}

    def _progress(fold, n_splits, epoch, total, f1):
        if fold == 0:                      # 折跑完了，进入部署模型训练
            print("  交叉验证完成，正在用全量数据训练部署模型"
                  "（仅供推理，不参与上面的指标）...", flush=True)
        elif epoch is None:
            last["fold"] = fold
            print(f"  第 {fold}/{n_splits} 折完成", flush=True)
        elif epoch % 10 == 0 or epoch == total:
            print(f"    轮 {epoch:>3}/{total}　验证宏F1 = {f1:.3f}", flush=True)

    t0 = time.time()
    res = cross_validate(cfg, ds, model_name=model, group_by_subject=group,
                         progress_callback=_progress)
    print()
    print(res.summary())
    print(f"\n[完成] 总耗时 {time.time() - t0:.1f} s")

    # 训练一结束就存模型：置换检验要跑好几分钟，让人先拿到模型更合理
    if getattr(args, "save_model", False):
        _save_model(cfg, res)

    if args.permutation:
        from .train import permutation_test

        n = args.n_permutations or cfg.train.n_permutations
        _banner(f"置换检验（{n} 次打乱标签）")
        perm = permutation_test(cfg, ds, model_name=model,
                                observed=res.accuracy, n_permutations=n)
        print()
        print(perm.summary())
        res.permutation = perm

    if args.report:
        _write_report(cfg, ds, {model: res}, rep, tag=model)
    return 0


def cmd_compare(args, cfg: Config) -> int:
    _banner("SVM vs ShallowCNN 分类性能对比")
    from .train import compare_models, comparison_table

    ds, rep = _load_saved_dataset(cfg, args.data)
    print(f"\n[数据] {ds.n_trials} 试次，{ds.n_classes} 分类，"
          f"随机水平 {ds.chance_level:.1%}"
          f"{'（仿真）' if ds.is_synthetic else '（真实）'}")

    def _cb(name, payload):
        """进度回调。payload 为 None 表示换模型了，否则是训练进度元组。

        CNN 一次交叉验证十几分钟，必须让用户看见它在动。
        """
        if payload is None:
            print(f"\n>>> 正在评估 {name.upper()} ...", flush=True)
            return
        fold, n_splits, epoch, total, f1 = payload
        if fold == 0:                      # 折跑完了，进入部署模型训练
            print(f"    {name.upper()} 交叉验证完成，正在训练部署模型 ...", flush=True)
        elif epoch is None:
            print(f"    第 {fold}/{n_splits} 折完成", flush=True)
        elif epoch % 20 == 0 or epoch == total:
            print(f"      {name.upper()} 轮 {epoch:>3}/{total}　验证宏F1 = {f1:.3f}",
                  flush=True)

    results = compare_models(cfg, ds, models=("svm", "cnn"),
                             group_by_subject=args.group,
                             progress_callback=_cb)
    print()
    print(comparison_table(results))

    # 均值差多少说明不了"差距稳不稳定" —— 两条流水线跑的是同一批折划分，
    # 逐折配对才看得出 CNN 是不是每折都赢。这一节是结论能不能立住的依据。
    from .train import paired_model_comparison
    paired = paired_model_comparison(results, "svm", "cnn")
    if paired:
        print()
        print(paired)

    _write_report(cfg, ds, results, rep, tag="compare")
    return 0


def cmd_sweep(args, cfg: Config) -> int:
    """信噪比扫描 —— 本项目最有价值的实验：多难的数据下 CNN 才跑赢 SVM。"""
    _banner("信噪比扫描")
    from .data.epoching import preprocess_dataset
    from .data.synthetic import generate_dataset
    from .train import holdout_evaluate

    # 默认区间与技术文档 8.1 的标定实验一致（scripts/calibrate_snr.py 用的就是
    # -12 ~ +4）。这里原先默认到 -16 dB —— 那个点两个模型都在随机水平，文档里
    # 也没有，白等一轮训练。
    snrs = args.snrs or [-12.0, -8.0, -4.0, 0.0, 4.0]
    print(f"扫描点：{snrs} dB，每个点 {cfg.data.synthetic.n_subjects} 受试者 × "
          f"{cfg.data.synthetic.trials_per_class} 试次/类")
    print("（耗时随上面的数据量线性增长；技术文档 8.1 那张表用的是 "
          "4 受试者 × 40 试次/类，可用 --set 调小）\n")
    print(f"{'SNR(dB)':>9} | {'SVM':>8} | {'CNN':>8}")
    print("-" * 34)

    rows = []
    for snr in snrs:
        c = load_config(args.config, _parse_overrides(args.set))
        c.data.synthetic.snr_db = float(snr)
        ds = generate_dataset(c)
        ds, _ = preprocess_dataset(ds, c)
        accs = {}
        for name in ("svm", "cnn"):
            c.model.name = name
            accs[name] = holdout_evaluate(c, ds, model_name=name).accuracy
        rows.append((snr, accs["svm"], accs["cnn"]))
        print(f"{snr:>+9.1f} | {accs['svm']:>7.1%} | {accs['cnn']:>7.1%}", flush=True)

    from . import plots

    fig = plots.plot_snr_sweep([{"snr_db": r[0], "svm": r[1], "cnn": r[2]}
                                for r in rows])
    out = plots.save(fig, cfg.sub_dir("figures") / "信噪比扫描.png")
    print(f"\n[完成] 曲线已保存：{out}")

    csv = cfg.sub_dir("reports") / "snr_sweep.csv"
    with open(csv, "w", encoding="utf-8") as fh:
        fh.write("snr_db,svm_acc,cnn_acc\n")
        for r in rows:
            fh.write(f"{r[0]},{r[1]:.4f},{r[2]:.4f}\n")
    print(f"[完成] 数据已保存：{csv}")
    return 0


def cmd_report(args, cfg: Config) -> int:
    _banner("生成 HTML 分析报告")
    from .train import cross_validate

    ds, rep = _load_saved_dataset(cfg, args.data)
    models = [m.strip() for m in (args.models or "svm,cnn").split(",") if m.strip()]
    results = {}
    for name in models:
        print(f"\n>>> 评估 {name.upper()} ...", flush=True)
        results[name] = cross_validate(cfg, ds, model_name=name,
                                       group_by_subject=args.group)
        print(results[name].summary())
    _write_report(cfg, ds, results, rep, tag="full")
    return 0


def cmd_infer(args, cfg: Config) -> int:
    """用训好的模型对一段记录做预测 —— 演示时的"实时分类"。"""
    _banner("推理")
    import numpy as np

    from .data.epoching import preprocess_raw
    from .data.loaders import load_any
    from .models import build_model

    model_path = Path(args.model_path or (cfg.sub_dir("models") / "model.joblib"))
    if not model_path.exists():
        raise SystemExit(
            f"找不到已训练模型：{model_path}\n"
            "  请先运行：python -m cneeg train --model svm --save-model"
        )
    import joblib

    blob = joblib.load(model_path)
    model, class_names = blob["model"], blob["class_names"]
    print(f"[模型] {blob.get('model_name', '?').upper()}　"
          f"类别 {class_names}　训练于 {'仿真' if blob.get('is_synthetic') else '真实'}数据")

    # 没给 --file 时 load_any 会塞一段 120 秒的演示录音进来（20 试次左右）。
    # 它和训练数据毫无关系，模型在这上面表现差是正常的 —— 但屏幕上跳出一个
    # 低于随机水平的一致率，看起来就像模型坏了。所以这里必须把"用的是哪份
    # 数据"说在前面，而不是只报一个数字。
    is_demo = args.file in (None, "")
    rec = load_any(args.file, cfg)
    print(f"[数据] {rec.summary()}")
    if is_demo:
        print("[数据] 本次未指定 --file，上面是系统自带的演示录音，与训练数据无关。")
    ds, _ = preprocess_raw(rec.data, rec.sfreq, rec.events, rec.event_codes,
                           rec.ch_names, rec.class_names, cfg,
                           rec.is_synthetic, rec.meta)
    if args.limit:
        ds = ds.subset(np.arange(min(args.limit, ds.n_trials)))

    proba = model.predict_proba(ds.X)
    pred = proba.argmax(1)

    print(f"\n逐试次预测（共 {ds.n_trials} 试次）：")
    print(f"  {'试次':>5}  {'预测':<14}{'置信度':>9}   真实")
    for i in range(min(ds.n_trials, 25)):
        mark = "OK" if pred[i] == ds.y[i] else "错误"
        print(f"  {i+1:>5}  {class_names[pred[i]]:<14}"
              f"{proba[i, pred[i]]:>8.1%}   {class_names[ds.y[i]]}  {mark}")
    if ds.n_trials > 25:
        print(f"  ...（其余 {ds.n_trials - 25} 试次省略）")

    acc = float((pred == ds.y).mean())
    # 这里是**一致率**，不是准确率，两者不能混着说：模型是在全量数据上重训的
    # 部署模型（见技术文档 7.5 节），它见过训练时的每一个试次；而按 --file
    # 传进来的记录也完全可能正是当初训练用的那一份。一旦如此，这个数字就是
    # 背答案背出来的。早先这里写的是"准确率：xx%（随机 25%）"，
    # 拿一个含义待定的数字去和随机水平比，是最容易被当成性能指标的一种说法。
    # 随机水平用 ds.chance_level（多数类占比），不要用 1/类别数 —— 那段演示录音
    # 里各类别的试次数并不相等，1/类别数会低估基线。（base.py 里已经改过同一处。）
    print(f"\n[结果] 与真实标签的一致率：{acc:.1%}（{ds.n_trials} 试次，"
          f"{len(class_names)} 类随机水平 {ds.chance_level:.1%}）")
    if is_demo:
        print("       【注意】这不是模型性能。上面那段是**演示录音**，"
              "它和训练数据没关系，也不代表真实场景；")
        print("              模型在这上面报低是正常的。看性能请用 compare 的"
              "交叉验证结果，对真实文件推理请加 --file。")
    else:
        print("       【注意】这里用的是在全量数据上重训的部署模型（见技术文档 7.5 节）。"
              "只有当这段记录")
        print("              没有参与过训练时，上面的一致率才是有意义的测试表现。")
    return 0


def cmd_gui(args, cfg: Config) -> int:
    _banner("启动图形界面")
    try:
        from .gui.main_window import run
    except ImportError as exc:
        print(f"[错误] 无法启动界面：{exc}\n"
              "  PyQt5 未安装？执行：pip install PyQt5")
        return 1
    return run(cfg) or 0


# ---------------------------------------------------------------------------
#  公共
# ---------------------------------------------------------------------------


def _write_report(cfg: Config, ds, results, rep, tag: str = "report") -> Path:
    from . import plots
    from .report import build_report
    from .train import results_to_dict

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = cfg.sub_dir("reports")

    # 除 HTML 之外再落一份 JSON：逐折指标、置换零分布这些要拿去做配对比较
    # 或重新绘图的东西，必须机器可读，HTML 里的表格做不了这件事。
    try:
        import json
        js = out_dir / f"results_{tag}_{stamp}.json"
        js.write_text(json.dumps(results_to_dict(results), ensure_ascii=False,
                                 indent=2), encoding="utf-8")
        print(f"[完成] 指标数据：{js}")
    except Exception as exc:                   # 报告本身不能因为导出失败而丢
        print(f"[警告] 指标 JSON 导出失败（不影响 HTML 报告）：{exc}")

    out = out_dir / f"report_{tag}_{stamp}.html"
    path = build_report(cfg, ds, results, preprocess_report=rep, out_path=out)
    print(f"\n[完成] HTML 报告：{path}")
    print("       双击即可用浏览器打开（图片已内嵌，单文件可直接发给老师）")
    return path


# ---------------------------------------------------------------------------
#  参数解析
# ---------------------------------------------------------------------------


def _add_common_args(sp: argparse.ArgumentParser) -> None:
    """把 ``--config/--set/--file`` 也挂到子命令上。

    直觉上大家会写 ``python -m cneeg gen-data --set ...``，但 argparse 默认
    只认**子命令之前**的全局选项。两边都挂一份，两种写法就都能用。

    ``default=SUPPRESS`` 是关键：否则子解析器会用 ``None`` 覆盖掉父解析器
    已经解析出来的值 —— 这是 argparse 的经典坑，表现为"写在前面反而不生效"。
    """
    sp.add_argument("--config", "-c", default=argparse.SUPPRESS,
                    help="配置文件路径（默认 configs/default.yaml）")
    # 存到 set_sub 而不是 set —— 子解析器是在一个**全新 namespace** 上解析的，
    # 若复用同名 dest，"子命令前的 --set" 会被"子命令后的 --set"整个覆盖掉。
    # 两个都写时由 main() 合并。
    sp.add_argument("--set", action="append", metavar="键=值", dest="set_sub",
                    default=argparse.SUPPRESS,
                    help="覆盖配置项，可重复。例：--set model.cnn.n_epochs=120")
    sp.add_argument("--file", "-f", default=argparse.SUPPRESS,
                    help="真实脑电文件路径（.mff/.vhdr/.edf/.bdf/.cnt/.set/.fif）")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m cneeg",
        description="汉语脑电信号分类软件系统 —— 命令行入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", "-c", default=None,
                   help="配置文件路径（默认 configs/default.yaml）")
    p.add_argument("--set", action="append", metavar="键=值",
                   help="覆盖配置项，可重复。例：--set model.cnn.n_epochs=120")
    p.add_argument("--file", "-f", default=None,
                   help="真实脑电文件路径（.mff/.vhdr/.edf/.bdf/.cnt/.set/.fif）")

    sub = p.add_subparsers(dest="command", metavar="命令")

    sp = sub.add_parser("gen-data", help="生成数据集并出检查图")

    _add_common_args(sp)
    sp.set_defaults(func=cmd_gen_data)

    sp = sub.add_parser("train", help="训练并评估一个模型")

    _add_common_args(sp)
    sp.add_argument("--model", "-m", choices=["svm", "cnn"], default=None)
    sp.add_argument("--data", "-d", default=None, help="复用已保存的 .npz 数据")
    sp.add_argument("--group", action="store_true",
                    help="留一受试者交叉验证（更严格，测试受试者训练时没见过）")
    sp.add_argument("--permutation", action="store_true", help="额外做置换检验")
    sp.add_argument("--n-permutations", type=int, default=None)
    sp.add_argument("--save-model", action="store_true", help="保存模型供推理使用")
    sp.add_argument("--report", action="store_true", help="同时生成 HTML 报告")
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("compare", help="SVM vs CNN 对比")

    _add_common_args(sp)
    sp.add_argument("--data", "-d", default=None)
    sp.add_argument("--group", action="store_true")
    sp.set_defaults(func=cmd_compare)

    sp = sub.add_parser("sweep", help="信噪比扫描")

    _add_common_args(sp)
    sp.add_argument("--snrs", type=float, nargs="+", default=None,
                    help="要扫描的 SNR（dB）列表，默认 "
                         "-12 -8 -4 0 4（与技术文档 8.1 的标定区间一致）")
    sp.set_defaults(func=cmd_sweep)

    sp = sub.add_parser("report", help="生成 HTML 分析报告")

    _add_common_args(sp)
    sp.add_argument("--data", "-d", default=None)
    sp.add_argument("--models", default="svm,cnn")
    sp.add_argument("--group", action="store_true")
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser("infer", help="用已训练模型对新记录做预测")

    _add_common_args(sp)
    sp.add_argument("--model-path", default=None)
    sp.add_argument("--limit", type=int, default=0, help="只跑前 N 个试次")
    # 这里曾经有个隐藏的 --save-model，但它是空实现（推理本来就不保存模型）。
    # 一个"接受了却什么都不做"的开关比没有这个开关更糟 —— 用户会以为存下了。
    sp.set_defaults(func=cmd_infer)

    sp = sub.add_parser("gui", help="启动 PyQt5 图形界面")

    _add_common_args(sp)
    sp.set_defaults(func=cmd_gui)

    return p


def _make_console_robust() -> None:
    """让控制台遇到打不出的字符时用 ``?`` 顶替，而不是抛异常终止。

    Chinese Windows 的控制台是 GBK（cp936）。数学减号 U+2212、乘号 U+00D7、
    对勾 U+2713 这类字符都不在 GBK 里，一旦打印出去就是
    ``UnicodeEncodeError`` —— 而且往往发生在命令**跑完所有计算、正要打印
    结果**的那一刻，几十分钟的运算白做，用户看到的却只是一个编码错误。

    这里保留控制台原本的编码（中文才能正常显示），只把错误处理改成替换。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError, OSError):
            # 被重定向到文件、或某些 IDE 的假 stdout 不支持 reconfigure
            pass


def main(argv=None) -> int:
    _make_console_robust()
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()

    # `python -m cneeg`（不带子命令）默认打开界面，双击 run.bat 的体验最顺。
    # 判据是"argv 里有没有出现任何已知子命令"，不能只看 argv[0] —— 否则
    # `--set seed=7 gen-data` 会被误判成"没给子命令"而插进一个 gui。
    commands = {"gen-data", "train", "compare", "sweep", "report", "infer", "gui"}
    if not any(a in commands for a in argv) and not any(
            a in argv for a in ("--help", "-h", "--version")):
        argv = ["gui"] + argv

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0

    # 合并"子命令前"和"子命令后"两处 --set（见 _add_common_args）
    args.set = list(getattr(args, "set", None) or []) + \
        list(getattr(args, "set_sub", None) or [])

    try:
        cfg = load_config(args.config, _parse_overrides(args.set))
    except Exception as exc:
        print(f"[配置错误] {exc}")
        return 2

    print(f"汉语脑电信号分类软件系统　配置：{args.config or 'configs/default.yaml'}"
          f"　种子：{cfg.seed}")

    try:
        return args.func(args, cfg)
    except KeyboardInterrupt:
        print("\n[中断] 用户取消")
        return 130
    except SystemExit:
        raise
    except Exception as exc:                      # noqa: BLE001 — 顶层兜底，给人看的
        import traceback
        print(f"\n[运行出错] {type(exc).__name__}: {exc}\n")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
