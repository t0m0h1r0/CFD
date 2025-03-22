from abc import ABC, abstractmethod
import numpy as np

class Equation(ABC):
    """差分方程式の基底クラス"""

    def __init__(self, grid=None):
        """初期化
        
        Args:
            grid: 計算格子オブジェクト
        """
        self.grid = grid

    def set_grid(self, grid):
        """グリッドを設定
        
        Args:
            grid: 計算格子オブジェクト
            
        Returns:
            self: メソッドチェーン用
        """
        self.grid = grid
        return self

    @abstractmethod
    def get_stencil_coefficients(self, i=None):
        """グリッド点iにおけるステンシル係数を返す
        
        Args:
            i: グリッド点のインデックス
            
        Returns:
            ステンシル係数の辞書
        """
        pass

    @abstractmethod
    def is_valid_at(self, i=None):
        """方程式がグリッド点iに適用可能かを判定
        
        Args:
            i: グリッド点のインデックス
            
        Returns:
            有効性を示すブール値
        """
        pass

    def __add__(self, other):
        return CombinedEquation(self, other, "+")

    def __sub__(self, other):
        return CombinedEquation(self, other, "-")

    def __mul__(self, scalar):
        return ScaledEquation(self, scalar)

    __rmul__ = __mul__


class CombinedEquation(Equation):
    """二つの方程式を組み合わせた方程式"""

    def __init__(self, eq1, eq2, operation="+", grid=None):
        """二つの方程式を組み合わせた方程式を初期化
        
        Args:
            eq1: 1つ目の方程式
            eq2: 2つ目の方程式
            operation: 演算子（"+"または"-"）
            grid: 計算格子オブジェクト（オプション）
        """
        # gridが指定されなかった場合、eq1またはeq2のgridを使用（存在すれば）
        if grid is None:
            if hasattr(eq1, 'grid') and eq1.grid is not None:
                grid = eq1.grid
            elif hasattr(eq2, 'grid') and eq2.grid is not None:
                grid = eq2.grid
                
        super().__init__(grid)
        self.eq1 = eq1
        self.eq2 = eq2
        self.operation = operation
        
        # 部分方程式にも同じgridを設定（ある場合）
        if self.grid is not None:
            if hasattr(eq1, 'set_grid'):
                eq1.set_grid(self.grid)
            if hasattr(eq2, 'set_grid'):
                eq2.set_grid(self.grid)

    def set_grid(self, grid):
        """グリッドを設定
        
        両方の部分方程式にも同じグリッドを設定する
        
        Args:
            grid: 計算格子オブジェクト
            
        Returns:
            self: メソッドチェーン用
        """
        super().set_grid(grid)
        
        # 部分方程式にもgridを設定
        if hasattr(self.eq1, 'set_grid'):
            self.eq1.set_grid(grid)
        if hasattr(self.eq2, 'set_grid'):
            self.eq2.set_grid(grid)
            
        return self

    def get_stencil_coefficients(self, i=None):
        """結合されたステンシル係数を返す
        
        Args:
            i: グリッド点のインデックス
            
        Returns:
            ステンシル係数の辞書
        """
        if self.grid is None:
            raise ValueError("gridが設定されていません。set_grid()で設定してください。")
        
        if i is None:
            raise ValueError("グリッド点のインデックスiを指定する必要があります。")
        
        # 両方の方程式のステンシル係数を取得
        coeffs1 = self.eq1.get_stencil_coefficients(i)
        coeffs2 = self.eq2.get_stencil_coefficients(i)
        
        # 結合
        combined_coeffs = {}
        all_offsets = set(list(coeffs1.keys()) + list(coeffs2.keys()))

        for offset in all_offsets:
            coeff1 = coeffs1.get(offset, np.zeros(4))
            coeff2 = coeffs2.get(offset, np.zeros(4))
            if self.operation == "+":
                combined_coeffs[offset] = coeff1 + coeff2
            else:
                combined_coeffs[offset] = coeff1 - coeff2

        return combined_coeffs

    def is_valid_at(self, i=None):
        """結合された方程式が有効かどうかを判定
        
        Args:
            i: グリッド点のインデックス
            
        Returns:
            有効性を示すブール値
        """
        if self.grid is None:
            raise ValueError("gridが設定されていません。set_grid()で設定してください。")
        
        if i is None:
            raise ValueError("グリッド点のインデックスiを指定する必要があります。")
        
        return self.eq1.is_valid_at(i) and self.eq2.is_valid_at(i)


class ScaledEquation(Equation):
    """スカラー倍された方程式"""

    def __init__(self, equation, scalar, grid=None):
        """スカラー倍された方程式を初期化
        
        Args:
            equation: 元の方程式
            scalar: スカラー倍する値
            grid: 計算格子オブジェクト（オプション）
        """
        # gridが指定されなかった場合、equationのgridを使用（存在すれば）
        if grid is None and hasattr(equation, 'grid'):
            grid = equation.grid
            
        super().__init__(grid)
        self.equation = equation
        self.scalar = scalar
        
        # 元の方程式にも同じgridを設定（ある場合）
        if self.grid is not None and hasattr(equation, 'set_grid'):
            equation.set_grid(self.grid)

    def set_grid(self, grid):
        """グリッドを設定
        
        元の方程式にも同じグリッドを設定する
        
        Args:
            grid: 計算格子オブジェクト
            
        Returns:
            self: メソッドチェーン用
        """
        super().set_grid(grid)
        
        # 元の方程式にもgridを設定
        if hasattr(self.equation, 'set_grid'):
            self.equation.set_grid(grid)
            
        return self

    def get_stencil_coefficients(self, i=None):
        """スケールされたステンシル係数を返す
        
        Args:
            i: グリッド点のインデックス
            
        Returns:
            ステンシル係数の辞書
        """
        if self.grid is None:
            raise ValueError("gridが設定されていません。set_grid()で設定してください。")
        
        if i is None:
            raise ValueError("グリッド点のインデックスiを指定する必要があります。")
        
        coeffs = self.equation.get_stencil_coefficients(i)
        return {offset: self.scalar * coeff for offset, coeff in coeffs.items()}

    def is_valid_at(self, i=None):
        """方程式が有効かどうかを判定
        
        Args:
            i: グリッド点のインデックス
            
        Returns:
            有効性を示すブール値
        """
        if self.grid is None:
            raise ValueError("gridが設定されていません。set_grid()で設定してください。")
        
        if i is None:
            raise ValueError("グリッド点のインデックスiを指定する必要があります。")
        
        return self.equation.is_valid_at(i)