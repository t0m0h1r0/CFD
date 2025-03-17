"""
行列スケーリングモジュール

高精度コンパクト差分法(CCD)用の行列スケーリング実装を提供します。
"""

from .plugin_manager import ScalingPluginManager
from .array_utils import ArrayBackend

# 簡単にアクセスできるようにプラグインマネージャーのインスタンスを作成
plugin_manager: ScalingPluginManager = ScalingPluginManager()

# すべてのスケーリング操作のバックエンドを設定する関数
def set_backend(backend: str):
    """
    すべてのスケーリング操作の計算バックエンドを設定
    
    Args:
        backend: 'numpy', 'cupy', または 'jax'
    """
    plugin_manager.set_backend(backend)