# -*- coding: utf-8 -*-
"""
HTML 实验报告生成
=================

把一次完整实验的所有产物（数据概况、预处理、模型配置、指标、图表、
置换检验）汇总成**单个自包含的 HTML 文件** —— 图片以 base64 内嵌，
不依赖任何外部文件，直接发给老师或上传到系统都能正常打开。

.. important::
   报告顶部会根据数据来源显示横幅：仿真数据是**红色警示**，真实数据是
   绿色。这是有意的 —— 仿真结果绝不能被误当成真实实验结论。
"""

from __future__ import annotations

import base64
import html
import io
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from matplotlib.figure import Figure

from .config import CLASS_LABELS, PRESET_LABELS, Config
from .data.base import EEGDataset
from .train import EvaluationResult, comparison_table

# ---------------------------------------------------------------------------
#  样式
# ---------------------------------------------------------------------------

_CSS = """
:root {
  --ink:#1c2431; --muted:#5b6b7f; --line:#dde5ee; --bg:#f6f8fb;
  --card:#ffffff; --accent:#2b5d8a; --warn:#b03030; --ok:#2f7d54;
}
* { box-sizing: border-box; }
body {
  margin:0; padding:32px 20px 64px; background:var(--bg); color:var(--ink);
  font-family:"Microsoft YaHei","PingFang SC","Hiragino Sans GB",system-ui,sans-serif;
  line-height:1.75; font-size:15px;
}
.wrap { max-width:1080px; margin:0 auto; }
h1 { font-size:26px; margin:0 0 6px; letter-spacing:.4px; }
h2 {
  font-size:19px; margin:38px 0 14px; padding-left:11px;
  border-left:4px solid var(--accent);
}
h3 { font-size:16px; margin:24px 0 10px; color:var(--accent); }
.sub { color:var(--muted); font-size:13px; margin-bottom:22px; }
.card {
  background:var(--card); border:1px solid var(--line); border-radius:10px;
  padding:18px 22px; margin:14px 0;
}
.banner {
  border-radius:10px; padding:14px 20px; margin:0 0 22px;
  font-weight:600; border:1px solid;
}
.banner.warn { background:#fdf0f0; border-color:#f0c4c4; color:var(--warn); }
.banner.ok   { background:#eef8f2; border-color:#bfe3ce; color:var(--ok); }
.banner small { display:block; font-weight:400; margin-top:5px; opacity:.85; }
table { border-collapse:collapse; width:100%; margin:10px 0; font-size:14px; }
th,td { border:1px solid var(--line); padding:8px 12px; text-align:left; }
th { background:#eef3f9; font-weight:600; }
td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
tr:nth-child(even) td { background:#fafcfe; }
.metric-row { display:flex; flex-wrap:wrap; gap:12px; margin:12px 0; }
.metric {
  flex:1 1 150px; background:var(--card); border:1px solid var(--line);
  border-radius:10px; padding:14px 16px;
}
.metric .k { font-size:12px; color:var(--muted); }
.metric .v { font-size:23px; font-weight:700; font-variant-numeric:tabular-nums; }
.metric .v small { font-size:13px; font-weight:400; color:var(--muted); }
figure { margin:18px 0; text-align:center; }
figure img { max-width:100%; border:1px solid var(--line); border-radius:8px; }
figcaption { font-size:13px; color:var(--muted); margin-top:7px; }
pre {
  background:#f0f4f9; border:1px solid var(--line); border-radius:8px;
  padding:13px 16px; overflow-x:auto; font-size:13px; line-height:1.6;
}
.good { color:var(--ok); font-weight:600; }
.bad  { color:var(--warn); font-weight:600; }
footer {
  margin-top:48px; padding-top:16px; border-top:1px solid var(--line);
  font-size:12px; color:var(--muted);
}
"""


def _fig_to_base64(fig: Figure, dpi: int = 130) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="white")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return f"data:image/png;base64,{b64}"


def _figure_block(fig: Figure, caption: str, dpi: int = 130) -> str:
    return (f"<figure><img src='{_fig_to_base64(fig, dpi)}' alt='{html.escape(caption)}'>"
            f"<figcaption>{html.escape(caption)}</figcaption></figure>")


def _metric(k: str, v: str, sub: str = "") -> str:
    small = f"<small>{html.escape(sub)}</small>" if sub else ""
    return (f"<div class='metric'><div class='k'>{html.escape(k)}</div>"
            f"<div class='v'>{html.escape(v)}{small}</div></div>")


def _class_label(name: str) -> str:
    return CLASS_LABELS.get(name, name)


# ---------------------------------------------------------------------------
#  报告正文
# ---------------------------------------------------------------------------


def build_report(
    cfg: Config,
    ds: EEGDataset,
    results: Dict[str, EvaluationResult],
    preprocess_report=None,
    out_path: Optional[str | Path] = None,
    figures: Optional[List[tuple]] = None,
    title: Optional[str] = None,
    extra_notes: Optional[str] = None,
) -> Path:
    """生成一份自包含的 HTML 报告，返回写出的文件路径。

    ``figures`` 可以传 ``[(Figure, 说明文字), ...]`` 追加自定义图。
    """
    import matplotlib.pyplot as plt

    from . import plots
    from .config import PRESET_LABELS

    title = title or cfg.output.report_title
    out_path = Path(out_path or (cfg.sub_dir("reports") /
                                 f"report_{datetime.now():%Y%m%d_%H%M%S}.html"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    is_syn = ds.is_synthetic
    banner = (
        "<div class='banner warn'>【仿真数据】本报告基于程序生成的仿真脑电数据"
        "<small>仿真数据仅用于验证软件流水线与模型对比方法，<b>不构成真实汉语脑电的"
        "实验结论</b>。真实数据接入后请重新运行本报告，届时结论才具科学意义。</small></div>"
        if is_syn else
        "<div class='banner ok'>【真实数据】本报告基于实验室实际采集的脑电数据。</div>"
    )

    parts: List[str] = [
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title><style>{_CSS}</style></head><body>",
        "<div class='wrap'>",
        f"<h1>{html.escape(title)}</h1>",
        f"<div class='sub'>生成时间：{datetime.now():%Y-%m-%d %H:%M}　|　"
        f"数据预设：{html.escape(PRESET_LABELS.get(ds.meta.get('preset', ''), ds.meta.get('preset', '—')))}"
        f"　|　随机种子：{cfg.seed}</div>",
        banner,
    ]

    # ---- 1. 核心指标 ----
    parts.append("<h2>一、核心结果</h2>")
    if results:
        best_name = max(results, key=lambda k: results[k].accuracy)
        best = results[best_name]
        parts.append("<div class='metric-row'>")
        parts.append(_metric("最佳模型", best_name.upper()))
        parts.append(_metric("准确率", f"{best.accuracy:.1%}",
                             f"± {best.accuracy_std:.1%}"))
        parts.append(_metric("相对随机提升", f"{best.gain_over_chance:+.1%}",
                             f"随机 {best.chance_level:.0%}"))
        parts.append(_metric("宏平均 F1", f"{best.macro_f1:.3f}"))
        parts.append(_metric("Cohen's κ", f"{best.kappa:.3f}"))
        parts.append("</div>")

        parts.append("<h3>模型对比</h3>")
        parts.append("<table><tr><th>模型</th><th class='num'>准确率</th>"
                     "<th class='num'>宏平均 F1</th><th class='num'>κ</th>"
                     "<th class='num'>耗时</th></tr>")
        for name, r in results.items():
            mark = " ★" if name == best_name else ""
            parts.append(
                f"<tr><td>{name.upper()}{mark}</td>"
                f"<td class='num'>{r.accuracy:.1%} ± {r.accuracy_std:.1%}</td>"
                f"<td class='num'>{r.macro_f1:.3f} ± {r.macro_f1_std:.3f}</td>"
                f"<td class='num'>{r.kappa:.3f}</td>"
                f"<td class='num'>{r.elapsed_s:.1f} s</td></tr>")
        parts.append("</table>")

        if len(results) > 1:
            parts.append(_figure_block(
                plots.plot_model_comparison(results),
                "图 1　SVM 基线与 ShallowCNN 的分类性能对比"))

    # ---- 2. 数据 ----
    parts.append("<h2>二、数据概况</h2>")
    parts.append("<div class='card'><pre>" + html.escape(ds.summary()) + "</pre></div>")

    parts.append("<h3>各类别试次数</h3><table><tr><th>类别</th><th>含义</th>"
                 "<th class='num'>试次数</th></tr>")
    for i, name in enumerate(ds.class_names):
        parts.append(f"<tr><td>{html.escape(str(name))}</td>"
                     f"<td>{html.escape(_class_label(str(name)))}</td>"
                     f"<td class='num'>{int((ds.y == i).sum())}</td></tr>")
    parts.append("</table>")

    parts.append(_figure_block(plots.plot_erp(ds),
                               "图 2　各类别在关键导联上的平均 ERP —— 波形分离程度反映任务的可分性"))

    # ---- 3. 预处理 ----
    if preprocess_report is not None:
        parts.append("<h2>三、预处理</h2>")
        parts.append("<div class='card'><pre>"
                     + html.escape(preprocess_report.summary()) + "</pre></div>")

    # ---- 4. 方法与配置 ----
    parts.append("<h2>四、方法与配置</h2>")
    parts.append("<h3>模型配置</h3><table><tr><th>项目</th><th>取值</th></tr>")
    if "svm" in results:
        s = cfg.model.svm
        parts.append(f"<tr><td>SVM</td><td>{html.escape(s.kernel)} 核，"
                     f"C={s.C}，gamma={html.escape(str(s.gamma))}"
                     f"{f'，PCA {s.pca_components} 维' if s.pca_components else ''}</td></tr>")
    if "cnn" in results:
        c = cfg.model.cnn
        parts.append(f"<tr><td>ShallowConvNet</td><td>时间卷积 {c.n_filters_time}×"
                     f"{c.filter_time_length}，空间卷积 {c.n_filters_spat}，"
                     f"池化 {c.pool_time_length}/{c.pool_time_stride}，"
                     f"dropout {c.dropout}</td></tr>")
        parts.append(f"<tr><td>训练</td><td>Adam lr={c.lr}，batch={c.batch_size}，"
                     f"最多 {c.n_epochs} 轮，早停耐心 {c.early_stop_patience}</td></tr>")
    if results:
        anyr = next(iter(results.values()))
        parts.append(f"<tr><td>评估方式</td><td>{html.escape(anyr.split)}"
                     f"{'（按受试者分组，训练时未见过测试受试者）' if anyr.group_by_subject else ''}"
                     f"</td></tr>")
    f = cfg.features
    parts.append(f"<tr><td>特征（SVM）</td><td>{html.escape(f.method)}，"
                 f"频带 {'/'.join(f.bands.keys())}</td></tr>")
    parts.append("</table>")

    p = cfg.preprocess
    parts.append("<h3>预处理参数</h3><table><tr><th>项目</th><th>取值</th></tr>")
    for k, v in [("带通", f"{p.l_freq:g} – {p.h_freq:g} Hz"),
                 ("陷波", f"{p.notch_freq:g} Hz" if p.notch_freq else "关闭"),
                 ("重参考", "共同平均参考" if p.reref == "car" else p.reref),
                 ("分段窗口", f"{p.tmin:g} – {p.tmax:g} s"),
                 ("基线", f"{p.baseline[0]:g} – {p.baseline[1]:g} s" if p.baseline else "无"),
                 ("伪迹剔除", f"峰峰值 > {p.reject_uv:g} µV" if p.reject_uv else "关闭"),
                 ("标准化", "逐通道 z-score" if p.normalize == "zscore" else p.normalize)]:
        parts.append(f"<tr><td>{k}</td><td>{html.escape(str(v))}</td></tr>")
    parts.append("</table>")

    # ---- 5. 详细结果 ----
    for idx, (name, r) in enumerate(results.items(), start=1):
        parts.append(f"<h2>五.{idx}　{name.upper()} 详细结果</h2>")
        parts.append(f"<div class='card'><pre>{html.escape(r.summary())}</pre></div>")

        parts.append(_figure_block(
            plots.plot_confusion(r),
            f"图 {10 + idx}　{name.upper()} 归一化混淆矩阵（对角线越深越好）"))

        parts.append("<h3>逐类指标</h3>")
        parts.append("<table><tr><th>类别</th><th>含义</th><th class='num'>精确率</th>"
                     "<th class='num'>召回率</th><th class='num'>F1</th></tr>")
        for cname, m in r.per_class.items():
            parts.append(
                f"<tr><td>{html.escape(str(cname))}</td>"
                f"<td>{html.escape(_class_label(str(cname)))}</td>"
                f"<td class='num'>{m['precision']:.3f}</td>"
                f"<td class='num'>{m['recall']:.3f}</td>"
                f"<td class='num'>{m['f1']:.3f}</td></tr>")
        parts.append("</table>")

        if r.history:
            parts.append(_figure_block(
                plots.plot_training_curve(r.history, r.chance_level),
                f"图 {20 + idx}　{name.upper()} 训练曲线"
                + (f"（早停于第 {r.best_epoch} 轮）" if r.best_epoch else "")))

        if r.permutation is not None:
            parts.append("<h3>置换检验</h3>")
            sig = r.permutation.p_value < 0.05
            parts.append(
                f"<div class='card'><pre>{html.escape(r.permutation.summary())}</pre>"
                f"<p>结论：<span class='{'good' if sig else 'bad'}'>"
                f"{'模型的解码能力显著高于随机水平（p < 0.05），结果可信。' if sig else '未能证明模型显著优于随机水平，结果存疑。'}"
                f"</span></p></div>")
            parts.append(_figure_block(
                plots.plot_permutation(r.permutation, r.chance_level),
                f"图 {30 + idx}　置换检验零分布（橙色线为实际结果，远超灰分布说明效应真实）"))

    # ---- 6. 附加图 ----
    if figures:
        parts.append("<h2>六、补充图表</h2>")
        for fig, cap in figures:
            parts.append(_figure_block(fig, cap))

    # ---- 7. 结论与局限 ----
    parts.append("<h2>七、结论与局限</h2><div class='card'>")
    if results:
        best_name = max(results, key=lambda k: results[k].accuracy)
        best = results[best_name]
        parts.append(
            f"<p>在本数据集上，<b>{best_name.upper()}</b> 取得最高准确率 "
            f"<b>{best.accuracy:.1%}</b>（随机水平 {best.chance_level:.0%}，"
            f"相对提升 {best.gain_over_chance:+.1%}），"
            f"宏平均 F1 = {best.macro_f1:.3f}，Cohen's κ = {best.kappa:.3f}。</p>")
    parts.append(
        "<p><b>局限与后续工作：</b></p><ul>"
        + ("<li><b>本报告使用仿真数据</b>，不具备科学结论效力，仅证明软件"
           "流水线与评估方法可正常工作。</li>" if is_syn else "")
        + "<li>声调/词汇的神经表征存在显著个体差异，跨受试者泛化通常明显低于"
          "受试者内结果，建议按受试者分别报告。</li>"
          "<li>当前窗口长度与频带设置为通用选择，未做逐受试者优化。</li>"
          "<li>真实数据接入后，应先用已知的显著效应（如 P300 或 N400）验证"
          "事件对齐是否正确，再评估解码性能。</li>"
        + "</ul></div>")

    parts.append(
        f"<footer>由「汉语脑电信号分类软件系统」自动生成　|　"
        f"Python {sys.version.split()[0]}　|　{platform.system()} "
        f"{platform.release()}　|　配置：{html.escape(str(cfg.data.source))} 数据源，"
        f"种子 {cfg.seed}</footer>")
    parts.append("</div></body></html>")

    if extra_notes:
        parts.insert(len(parts) - 3, f"<div class='card'>{extra_notes}</div>")

    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path
