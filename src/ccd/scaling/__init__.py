"""
行列スケーリングモジュール

高精度コンパクト差分法(CCD)用の行列スケーリング実装を提供します。
"""

import sys
import os

# 明示的にパスを追加して、importの問題を解決
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 事前にimport
try:
    from .base import BaseScaling
    from .no_scaling import NoScaling
    from .row_scaling import RowScaling
    from .column_scaling import ColumnScaling
    from .equilibration_scaling import EquilibrationScaling
    from .adaptive_scaling import AdaptiveScaling
except ImportError as e:
    print(f"Warning: Some scaling plugins couldn't be imported: {e}")

# プラグインマネージャーをインポート
from .plugin_manager import ScalingPluginManager

# 簡単にアクセスできるようにプラグインマネージャーのインスタンスを作成
plugin_manager: ScalingPluginManager = ScalingPluginManager()

# 最初の呼び出し時にプラグインを読み込む
plugin_manager.discover_plugins()