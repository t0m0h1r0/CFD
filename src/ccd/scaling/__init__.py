"""
行列スケーリングモジュール

高精度コンパクト差分法(CCD)用の行列スケーリング実装を提供します。
"""

# 標準ライブラリのインポート
import sys
import os

# モジュールレベルのインポート
from .plugin_manager import ScalingPluginManager

# 明示的にパスを追加して、importの問題を解決
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 簡単にアクセスできるようにプラグインマネージャーのインスタンスを作成
plugin_manager = ScalingPluginManager()

# 最初の呼び出し時にプラグインを読み込む
plugin_manager.discover_plugins()