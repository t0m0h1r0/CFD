"""
2次元方程式システムの定義モジュール

このモジュールは、CCD (Combined Compact Difference) 法で使用される
2次元問題の格子点方程式を管理し、線形方程式系の行列を構築するための機能を提供します。
"""

from core.base.base_equation_system import BaseEquationSystem

class EquationSystem2D(BaseEquationSystem):
    """2次元方程式システムを管理するクラス"""

    def _initialize_equations(self):
        """2D用の方程式コレクションを初期化"""
        # 領域ごとの方程式コレクション
        self.equations = {
            'interior': [],       # 内部点
            'left': [],           # 左境界
            'right': [],          # 右境界
            'bottom': [],         # 下境界
            'top': [],            # 上境界
            'left_bottom': [],    # 左下角
            'right_bottom': [],   # 右下角
            'left_top': [],       # 左上角
            'right_top': []       # 右上角
        }
    
    def _get_point_location(self, i, j=None, k=None):
        """
        格子点の位置タイプを判定
        
        Args:
            i: x方向のインデックス
            j: y方向のインデックス
            k: 未使用 (2Dでは無視)
            
        Returns:
            位置を表す文字列 ('interior', 'left', 'bottom', etc.)
        """
        if j is None:
            raise ValueError("2D格子では j インデックスが必要です")
            
        nx, ny = self.grid.nx_points, self.grid.ny_points
        
        # 内部点
        if 0 < i < nx - 1 and 0 < j < ny - 1:
            return 'interior'
        
        # 角点
        if i == 0 and j == 0:
            return 'left_bottom'
        elif i == nx - 1 and j == 0:
            return 'right_bottom'
        elif i == 0 and j == ny - 1:
            return 'left_top'
        elif i == nx - 1 and j == ny - 1:
            return 'right_top'
        
        # 境界エッジ（角を除く）
        if i == 0:
            return 'left'
        elif i == nx - 1:
            return 'right'
        elif j == 0:
            return 'bottom'
        elif j == ny - 1:
            return 'top'
        
        # 念のため
        return 'interior'
    
    def _assign_equations_2d(self, eq_by_type, i, j, k=None):
        """
        2D格子点における方程式の割り当てを決定
        
        Args:
            eq_by_type: 種類別に分類された方程式
            i: x方向のインデックス
            j: y方向のインデックス
            k: 未使用 (2Dでは無視)
            
        Returns:
            各行に割り当てる方程式のリスト [0行目, 1行目, ..., 6行目]
        """
        governing = eq_by_type["governing"]
        dirichlet = eq_by_type["dirichlet"]
        neumann_x = eq_by_type["neumann_x"]
        neumann_y = eq_by_type["neumann_y"]
        auxiliary = eq_by_type["auxiliary"].copy()  # コピーして操作
        
        # 方程式の不足時に使うデフォルト方程式
        fallback = governing
        
        # 割り当て結果
        assignments = [None] * 7
        
        # 方程式を割り当てる関数 (不足時はデフォルト)
        def assign_with_fallback(idx, primary, use_auxiliary=True, tertiary=None):
            if primary:
                assignments[idx] = primary
            elif use_auxiliary and auxiliary:
                assignments[idx] = auxiliary.pop(0)
            elif tertiary:
                assignments[idx] = tertiary
            else:
                assignments[idx] = fallback
        
        # 0行目: 支配方程式 (ψ)
        assignments[0] = governing
        
        # 1行目: x方向のディリクレ境界または補助方程式 (ψ_x)
        assign_with_fallback(1, dirichlet)
        
        # 2行目: x方向のノイマン境界または補助方程式 (ψ_xx)
        assign_with_fallback(2, neumann_x, True, dirichlet)
        
        # 3行目: 補助方程式 (ψ_xxx)
        assign_with_fallback(3, None, True, neumann_x or dirichlet)
        
        # 4行目: y方向のディリクレ境界または補助方程式 (ψ_y)
        assign_with_fallback(4, dirichlet if dirichlet and dirichlet != assignments[1] else None)
        
        # 5行目: y方向のノイマン境界または補助方程式 (ψ_yy)
        assign_with_fallback(5, neumann_y, True, dirichlet)
        
        # 6行目: 補助方程式 (ψ_yyy)
        assign_with_fallback(6, None, True, neumann_y or dirichlet)
        
        return assignments
    
    # 基底クラスの抽象メソッドを実装（1D, 3D用）
    def _assign_equations_1d(self, eq_by_type, i, j=None, k=None):
        # 2Dシステムでは呼び出されないはずなので例外を投げる
        raise NotImplementedError("2Dシステムで_assign_equations_1dが呼び出されました")
        
    def _assign_equations_3d(self, eq_by_type, i, j, k):
        # 2Dシステムでは呼び出されないはずなので例外を投げる
        raise NotImplementedError("2Dシステムで_assign_equations_3dが呼び出されました")