"""
1次元方程式システムの定義モジュール

このモジュールは、CCD (Combined Compact Difference) 法で使用される
1次元問題の格子点方程式を管理し、線形方程式系の行列を構築するための機能を提供します。
"""

from core.base.base_equation_system import BaseEquationSystem

class EquationSystem1D(BaseEquationSystem):
    """1次元方程式システムを管理するクラス"""

    def _initialize_equations(self):
        """1D用の方程式コレクションを初期化"""
        # 領域ごとの方程式コレクション
        self.equations = {
            'interior': [],   # 内部点
            'left': [],       # 左境界
            'right': [],      # 右境界
        }
        
    def _get_point_location(self, i, j=None, k=None):
        """
        格子点の位置タイプを判定
        
        Args:
            i: x方向のインデックス
            j: 未使用 (1Dでは無視)
            k: 未使用 (1Dでは無視)
            
        Returns:
            位置を表す文字列 ('interior', 'left', 'right')
        """
        n = self.grid.n_points
        if i == 0:
            return 'left'
        elif i == n - 1:
            return 'right'
        else:
            return 'interior'
    
    def _assign_equations_1d(self, eq_by_type, i, j=None, k=None):
        """
        1D格子点における方程式の割り当てを決定
        
        Args:
            eq_by_type: 種類別に分類された方程式
            i: x方向のインデックス
            j: 未使用 (1Dでは無視)
            k: 未使用 (1Dでは無視)
            
        Returns:
            各行に割り当てる方程式のリスト [0行目, 1行目, ..., 3行目]
        """
        governing = eq_by_type["governing"]
        dirichlet = eq_by_type["dirichlet"]
        neumann = eq_by_type["neumann"]
        auxiliary = eq_by_type["auxiliary"].copy()  # コピーして操作
        
        # 割り当て結果
        assignments = [None] * 4
        
        # 0行目: 支配方程式 (ψ)
        assignments[0] = governing
        
        # 1行目: ディリクレ境界または最初の補助方程式 (ψ')
        if dirichlet:
            assignments[1] = dirichlet
        elif auxiliary:
            assignments[1] = auxiliary.pop(0)
        else:
            raise ValueError(f"点 {i} の1行目に設定する方程式がありません")
        
        # 2行目: ノイマン境界または次の補助方程式 (ψ'')
        if neumann:
            assignments[2] = neumann
        elif auxiliary:
            assignments[2] = auxiliary.pop(0)
        else:
            raise ValueError(f"点 {i} の2行目に設定する方程式がありません")
        
        # 3行目: 残りの補助方程式 (ψ''')
        if auxiliary:
            assignments[3] = auxiliary.pop(0)
        else:
            raise ValueError(f"点 {i} の3行目に設定する方程式がありません")
        
        return assignments
    
    # 基底クラスの抽象メソッドを実装（2D, 3D用）
    def _assign_equations_2d(self, eq_by_type, i, j, k=None):
        # 1Dシステムでは呼び出されないはずなので例外を投げる
        raise NotImplementedError("1Dシステムで_assign_equations_2dが呼び出されました")
        
    def _assign_equations_3d(self, eq_by_type, i, j, k):
        # 1Dシステムでは呼び出されないはずなので例外を投げる
        raise NotImplementedError("1Dシステムで_assign_equations_3dが呼び出されました")