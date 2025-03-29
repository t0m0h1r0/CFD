"""
テスター (tester) パッケージ

このパッケージは、CCDソルバーのテストと評価を行うための
クラスと機能を提供します。次元に依存せず一貫したインターフェースで
ソルバーのテストを可能にします。
"""

from .tester1d import CCDTester1D
from .tester2d import CCDTester2D
from .tester3d import CCDTester3D


def create_tester(grid):
    """
    グリッドの次元に応じたテスターを作成
    
    Args:
        grid: CCDグリッドオブジェクト
    
    Returns:
        適切な次元のテスター
    """
    if hasattr(grid, 'is_3d') and grid.is_3d:
        return CCDTester3D(grid)
    elif hasattr(grid, 'is_2d') and grid.is_2d:
        return CCDTester2D(grid)
    else:
        return CCDTester1D(grid)