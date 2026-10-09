# -*- coding: utf-8 -*-
"""标定默认信噪比：找出 SVM 与 CNN 都明显高于随机水平的 snr_db。

跑法（项目根目录下）：
    python scripts/calibrate_snr.py
结果同时打印到控制台与 scripts/snr_sweep.csv。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cneeg.config import load_config                      # noqa: E402
from cneeg.data.synthetic import generate_dataset         # noqa: E402
from cneeg.data.epoching import preprocess_dataset        # noqa: E402
from cneeg.train import holdout_evaluate                  # noqa: E402

SNRS = [-12.0, -8.0, -4.0, 0.0, 4.0]
N_SUBJECTS = 4
TRIALS = 40

rows = []
print(f"标定数据集：{N_SUBJECTS} 受试者 × {TRIALS} 试次/类 × 4 类 "
      f"= {N_SUBJECTS * TRIALS * 4} 试次，留出法 30%", flush=True)
print(f"{'SNR(dB)':>8} | {'SVM':>9} | {'CNN':>9} | 耗时", flush=True)
print("-" * 52, flush=True)

for snr in SNRS:
    cfg = load_config(overrides={
        "data.synthetic.snr_db": snr,
        "data.synthetic.n_subjects": N_SUBJECTS,
        "data.synthetic.trials_per_class": TRIALS,
        "model.cnn.n_epochs": 60,
        "model.cnn.early_stop_patience": 12,
    })
    ds = generate_dataset(cfg)
    ds, rep = preprocess_dataset(ds, cfg)

    t0 = time.time()
    out = []
    for name in ("svm", "cnn"):
        cfg.model.name = name
        r = holdout_evaluate(cfg, ds, model_name=name)
        out.append(r.accuracy)
        if name == "cnn":
            best_ep = r.best_epoch
    dt = time.time() - t0
    rows.append((snr, out[0], out[1], best_ep, rep.n_rejected))
    print(f"{snr:>+8.1f} | {out[0]:>8.1%} | {out[1]:>8.1%} | "
          f"{dt:>4.0f}s（CNN 最佳轮次 {best_ep}，剔除 {rep.n_rejected} 试次）",
          flush=True)

csv = ROOT / "scripts" / "snr_sweep.csv"
with open(csv, "w", encoding="utf-8") as fh:
    fh.write("snr_db,svm_acc,cnn_acc,cnn_best_epoch,n_rejected\n")
    for r in rows:
        fh.write(f"{r[0]},{r[1]:.4f},{r[2]:.4f},{r[3]},{r[4]}\n")
print(f"\n已写出 {csv}", flush=True)
