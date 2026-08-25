"""PyInstaller用エントリポイント: vat serve を起動する。

配布パッケージではElectron側が vat-serve.exe を直接spawnする
（standalone/main.js の app.isPackaged 分岐を参照）。
"""

import sys

from vat.cli import main

if __name__ == "__main__":
    sys.exit(main(["serve"]))
