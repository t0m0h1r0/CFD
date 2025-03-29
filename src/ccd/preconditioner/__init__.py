# preconditioner/__init__.py
"""
前処理器プラグインパッケージ

このパッケージは、反復法ソルバーの収束を加速するための
前処理行列を計算する機能を提供します。
"""

from .plugin_manager import PreconditionerPluginManager
from .base import BasePreconditioner

# プラグインマネージャーのインスタンス
plugin_manager = PreconditionerPluginManager()

# 簡単にアクセスできる関数
def get_preconditioner(name=None):
    """
    指定名の前処理器を取得
    
    Args:
        name: 前処理器名（Noneの場合はデフォルト）
        
    Returns:
        前処理器インスタンス
    """
    return plugin_manager.get_plugin(name)

def list_preconditioners():
    """
    利用可能な前処理器の一覧を取得
    
    Returns:
        前処理器名のリスト
    """
    return plugin_manager.list_plugins()

# 基底クラスもエクスポート
__all__ = ['BasePreconditioner', 'get_preconditioner', 'list_preconditioners']