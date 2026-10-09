#!/usr/bin/env bash
# 完整验收流水线 —— 用法：bash scripts/acceptance_run.sh [-d 数据集.npz]
#
# 两个坑，都踩过：
#  1) python 必须加 -u。重定向到文件后 stdout 是块缓冲的，攒够 8KB 才落盘，
#     tail -f 看起来像"卡死"，其实一直在跑。
#  2) 日志文件必须先删掉再重定向，且确保没有旧进程还持有它的写偏移 ——
#     否则旧进程会从很大的偏移继续写，文件开头留下一段空洞（null）。
cd "$(dirname "$0")/.."
export PYTHONPATH=src PYTHONIOENCODING=utf-8
PY="D:/anaconda/python.exe"
DATA="${2:-$(ls -t data/processed/*.npz | head -1)}"
echo "########## 数据集：$DATA"
echo "########## [1/4] SVM 5 折交叉验证 + 置换检验"
$PY -u -m cneeg train --model svm -d "$DATA" --permutation --set train.n_permutations=20 2>&1
echo "########## [2/4] CNN 5 折交叉验证"
$PY -u -m cneeg train --model cnn -d "$DATA" 2>&1
echo "########## [3/4] SVM vs CNN 对比 + HTML 报告"
$PY -u -m cneeg compare -d "$DATA" 2>&1
echo "########## [4/4] 全部结束"
