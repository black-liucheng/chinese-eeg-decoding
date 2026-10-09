# -*- coding: utf-8 -*-
"""一键运行脚本（``.bat``）的静态检查。

这个文件是被一个真实的坑逼出来的：``run.bat`` 的报错分支里原本有一行

    echo   Install Python 3.13 (64-bit), then run:

它位于 ``if errorlevel 1 ( ... )`` 块**内部**，而括号在 .bat 里是块分隔符 ——
于是 ``(64-bit)`` 里的括号提前把块闭合了，剩下的 ``, then run:`` 被当成命令，
cmd 报 ``then was unexpected at this time.`` 并直接退出。用户双击后只看到
窗口一闪而过，**连那句报错都读不到**（因为 `pause` 在崩溃点之后）。

更值得记住的是它当时没被发现的原因：改完只核对了"文件是纯 ASCII"，
**没有真的把 run.bat 跑一遍**。编码检查过了，执行没测 —— 而用户用的是后者。
所以这里的断言刻意盯住"能不能被 cmd 解析"这类问题，而不是文件长相。

注意：本文件只做**静态**检查。它能拦住括号这类解析陷阱，但拦不住
"脚本能解析、跑起来却失败"。真跑一遍要启动 GUI，不适合放进单测。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BAT_FILES = sorted(ROOT.glob("*.bat")) + sorted((ROOT / "scripts").glob("*.bat"))

# 块开启行：这些行末尾的括号是**结构性的**，不是错误
_BLOCK_OPENERS = (
    re.compile(r"^\s*if\b.*\($", re.IGNORECASE),
    re.compile(r".*\bdo\s*\($", re.IGNORECASE),
)


def _strip_quoted(s: str) -> str:
    """去掉双引号里的内容 —— 其中的括号不会参与块结构解析。"""
    return re.sub(r'"[^"]*"', "", s)


def _offending_parens(line: str) -> int:
    """数出一行里会破坏块结构的括号个数。

    转义过的 ``^(`` ``^)`` 是安全的（cmd 会当作普通字符输出），
    双引号内的也算安全。其余的在块内部出现就是解析错误。
    """
    s = line.replace("^(", "").replace("^)", "")
    s = _strip_quoted(s)
    return s.count("(") + s.count(")")


def test_bat_files_exist():
    """交付物清单里必须有这两个入口，否则"一键运行"就无从谈起。"""
    names = {p.name for p in BAT_FILES}
    assert "run.bat" in names, f"找不到 run.bat（实际找到：{sorted(names)}）"
    assert "run_cli.bat" in names
    assert "_find_python.bat" in names


@pytest.mark.parametrize("path", BAT_FILES, ids=lambda p: p.name)
def test_bat_is_ascii_only(path: Path):
    """``.bat`` 必须全是 ASCII。

    cmd.exe 按 OEM 代码页（中文 Windows 上是 936）解析 ``.bat``，
    文件里的 UTF-8 中文会被拆成乱码，运气不好还会拆出 ``&`` ``|`` 这类
    元字符，把一个普通参数变成一条命令。所以中文一律放在 Python 侧输出。
    """
    raw = path.read_bytes()
    bad = [(i, b) for i, b in enumerate(raw) if b > 0x7F]
    assert not bad, (
        f"{path.name} 含 {len(bad)} 个非 ASCII 字节，"
        f"第一个在第 {bad[0][0]} 字节（0x{bad[0][1]:02X}）"
    )


@pytest.mark.parametrize("path", BAT_FILES, ids=lambda p: p.name)
def test_no_unescaped_parens_inside_blocks(path: Path):
    """块内部的括号必须转义 —— 见本文件开头那段事故记录。

    逐个跟踪块深度：``if ... (`` / ``for ... do (`` 加深一层，单独一行
    ``)`` 减一层。只检查**块内部**的行，因为块外的括号是合法的。
    """
    depth = 0
    offenders: list[str] = []

    for lineno, raw in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        stripped = raw.strip()

        if stripped == ")":                     # 块闭合行，本身就是一个括号
            depth = max(0, depth - 1)
            continue

        if depth > 0 and not any(p.match(raw) for p in _BLOCK_OPENERS):
            if _offending_parens(raw) > 0:
                offenders.append(f"  第 {lineno} 行: {stripped}")

        if stripped.endswith("(") and not stripped.startswith("::"):
            depth += 1

    assert not offenders, (
        f"{path.name} 在 if/for 块内部出现了未转义的括号，"
        f"cmd 会在这里报 'xxx was unexpected at this time' 并直接退出：\n"
        + "\n".join(offenders)
        + "\n  （改法：写成 ^( ... ^)，或者干脆别用括号）"
    )


@pytest.mark.parametrize("name", ["run.bat", "run_cli.bat"])
def test_entry_launchers_use_dp0_not_hardcoded_path(name: str):
    """两个入口脚本必须用 ``%~dp0`` 定位自己，不能写死路径。

    项目可能被放在任何地方（本机就在一个含中文的 OneDrive 目录下），
    写死路径会在换机器/换目录后直接失效。
    ``_find_python.bat`` 不在此列：它是被上面两个脚本按绝对路径调用的
    辅助脚本，本来就要去探测固定的解释器位置。
    """
    text = (ROOT / name).read_text(encoding="ascii")
    assert "%~dp0" in text, f"{name} 没有用 %~dp0 定位自身目录"


def test_run_bat_sets_required_env():
    """``run.bat`` 必须设好两个变量，且都要能被 Python 侧用到。"""
    text = (ROOT / "run.bat").read_text(encoding="ascii")
    assert "PYTHONPATH=%~dp0src" in text, "src/ 布局没进 PYTHONPATH，import cneeg 会失败"
    assert "KMP_DUPLICATE_LIB_OK" in text, (
        "缺这个会在 anaconda + pip 版 PyTorch 上触发 OMP: Error #15"
    )
