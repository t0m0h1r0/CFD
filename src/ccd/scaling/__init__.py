# scalingディレクトリをパッケージとして扱うためのファイル
from .plugin_manager import ScalingPluginManager
from typing import Optional

# 簡単にアクセスできるようにプラグインマネージャーのインスタンスを作成
plugin_manager: ScalingPluginManager = ScalingPluginManager()