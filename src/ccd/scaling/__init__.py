# scalingディレクトリをパッケージとして扱うためのファイル
from .plugin_manager import ScalingPluginManager

# 簡単にアクセスできるようにプラグインマネージャーのインスタンスを作成
plugin_manager: ScalingPluginManager = ScalingPluginManager()