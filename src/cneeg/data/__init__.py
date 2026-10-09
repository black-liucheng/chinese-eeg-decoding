# -*- coding: utf-8 -*-
"""数据层：仿真生成、真实数据适配、预处理与分段。"""

from .base import CH_NAMES_32, CH_POS_2D, EEGDataset, channel_positions

__all__ = ["EEGDataset", "CH_NAMES_32", "CH_POS_2D", "channel_positions"]
