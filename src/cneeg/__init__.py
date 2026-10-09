# -*- coding: utf-8 -*-
"""
cneeg —— 汉语脑电信号分类软件系统
=================================

大创项目《基于深度学习的汉语脑电信号建模与分析研究》的软件实现。

流水线：数据加载 → 预处理 → 特征提取 → 模型（SVM 基线 / ShallowCNN）→ 评估 → 报告

::

    from cneeg.config import load_config
    from cneeg.data.synthetic import generate_dataset

    cfg = load_config()
    ds = generate_dataset(cfg)
    print(ds.summary())

命令行入口见 ``python -m cneeg --help``。

.. warning::
   当前默认数据源为**仿真数据**（真实采集尚未开始）。仿真数据仅用于验证
   软件流水线与模型对比方法，**不作为真实汉语脑电的实验结论**。
"""

import os as _os

# ---------------------------------------------------------------------------
#  OpenMP 运行库冲突（anaconda + pip 版 torch）
# ---------------------------------------------------------------------------
#  anaconda 的 MKL 和 pip 安装的 PyTorch 各自打包了一份 libiomp5md.dll，
#  同进程加载两份会导致进程直接以 OMP Error #15 退出。必须在导入 torch **之前**
#  设置这个开关。放在包的最顶部，保证 `import cneeg` 先于 `import torch` 生效。
#  （run.bat / run_cli.bat 里也设了一份，双保险。）
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ---------------------------------------------------------------------------
#  目录枚举缓存失效
# ---------------------------------------------------------------------------
#  本项目放在 OneDrive 同步文件夹里。Windows 下 Python 的 FileFinder 会缓存
#  目录列表，只在目录 mtime 变化时重新枚举；同步驱动更新目录内容时并不总是
#  推进目录 mtime，刚新增的模块偶尔会临时找不到。invalidate_caches() 强制下次
#  查找时重新枚举一次，开销可忽略，放在包入口让每个入口自动生效。
import importlib as _importlib
_importlib.invalidate_caches()

__version__ = "1.0.0"
__all__ = ["__version__"]
