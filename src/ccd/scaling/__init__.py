# scalingディレクトリをパッケージとして扱うためのファイル
from .base import BaseScaling
from .plugin_manager import ScalingPluginManager

# 簡単にアクセスできるようにプラグインマネージャーのインスタンスを作成
plugin_manager = ScalingPluginManager()