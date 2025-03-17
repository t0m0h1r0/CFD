"""
行列スケーリングモジュール

高精度コンパクト差分法(CCD)用の行列スケーリング実装を提供します。
"""

from .plugin_manager import ScalingPluginManager

# 簡単にアクセスできるようにプラグインマネージャーのインスタンスを作成
plugin_manager: ScalingPluginManager = ScalingPluginManager()