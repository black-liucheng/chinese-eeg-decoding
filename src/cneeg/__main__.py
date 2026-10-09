# -*- coding: utf-8 -*-
"""让 ``python -m cneeg`` 可用（等价于 ``python -m cneeg.cli``）。"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
