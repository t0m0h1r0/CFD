"""
方程式セットの定義と管理を行うモジュール

このモジュールでは、CCD（Combined Compact Difference）法に使用される
様々な種類の方程式セットを定義し、次元に応じて適切に管理します。

このファイルは、リファクタリングされた複数のファイルをインポートし、
後方互換性を保つためのインターフェースを提供します。
"""

# 基底クラスのインポート
from core.base.base_equation_set import EquationSet, DimensionalEquationSetWrapper

# 1次元方程式セットのインポート
from equation_set.equation_set1d import (
    DerivativeEquationSet1D,
    PoissonEquationSet1D,
    PoissonEquationSet1D2
)

# 2次元方程式セットのインポート
from equation_set.equation_set2d import (
    DerivativeEquationSet2D,
    PoissonEquationSet2D,
    PoissonEquationSet2D2
)

from equation_set.equation_set3d import (
    DerivativeEquationSet3D,
    PoissonEquationSet3D,
    PoissonEquationSet3D2
)

# EquationSet クラスを拡張して、利用可能な方程式セットの辞書を提供
@classmethod
def get_available_sets(cls, dimension=None):
    """
    利用可能な方程式セットを返す
    
    Args:
        dimension: 1または2 (Noneの場合は両方)
        
    Returns:
        利用可能な方程式セットの辞書
    """
    all_sets = {
        "poisson": {"1d": PoissonEquationSet1D, "2d": PoissonEquationSet2D, "3d": PoissonEquationSet3D},
        "poisson2": {"1d": PoissonEquationSet1D2, "2d": PoissonEquationSet2D2, "3d": PoissonEquationSet3D2},
        "derivative": {"1d": DerivativeEquationSet1D, "2d": DerivativeEquationSet2D, "3d": DerivativeEquationSet3D},
    }
    
    if dimension == 1:
        return {key: value["1d"] for key, value in all_sets.items()}
    elif dimension == 2:
        return {key: value["2d"] for key, value in all_sets.items()}
    elif dimension == 3:
        return {key: value["3d"] for key, value in all_sets.items()}
    else:
        return all_sets

# クラスメソッドを上書き
EquationSet.get_available_sets = get_available_sets
