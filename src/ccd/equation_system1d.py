"""
1次元方程式システムの定義モジュール

このモジュールは、CCD (Combined Compact Difference) 法で使用される
1次元問題の格子点方程式を管理し、線形方程式系の行列を構築するための機能を提供します。
"""

import numpy as np
import scipy.sparse as sp_cpu

from base_equation_system import BaseEquationSystem

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
        
    def _get_point_location(self, i: int, j: int = None) -> str:
        """
        格子点の位置タイプを判定
        
        Args:
            i: x方向のインデックス
            j: 未使用 (1Dでは無視)
            
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
    
    def build_matrix_system(self):
        """
        1D用の行列システムを構築
        
        Returns:
            システム行列 (CPU CSR形式)
        """
        # 簡易的な検証
        self._validate_equations()
        
        n = self.grid.n_points
        var_per_point = 4  # 1点あたりの変数数 [ψ, ψ', ψ'', ψ''']
        system_size = n * var_per_point
        
        # 行列要素の蓄積用 (CPU処理)
        data = []
        row_indices = []
        col_indices = []
        
        # 各格子点について処理
        for i in range(n):
            base_idx = i * var_per_point
            location = self._get_point_location(i)
            location_equations = self.equations[location]
            
            # 方程式を種類別に分類
            eq_by_type = {"governing": None, "dirichlet": None, "neumann": None, "auxiliary": []}
            for eq in location_equations:
                eq_type = self._identify_equation_type(eq, i)
                if eq_type == "auxiliary":
                    eq_by_type["auxiliary"].append(eq)
                elif eq_type:  # Noneでない場合
                    eq_by_type[eq_type] = eq
            
            # 各種方程式の存在確認
            if not eq_by_type["governing"]:
                raise ValueError(f"点 {i} に支配方程式が設定されていません")
            
            # 各行に適切な方程式を配置
            assignments = [None] * var_per_point  # 各行の方程式
            
            # 0行目: 支配方程式 (ψ)
            assignments[0] = eq_by_type["governing"]
            
            # 1行目: ディリクレ境界または最初の補助方程式 (ψ')
            if eq_by_type["dirichlet"]:
                assignments[1] = eq_by_type["dirichlet"]
            elif eq_by_type["auxiliary"]:
                assignments[1] = eq_by_type["auxiliary"].pop(0)
            else:
                raise ValueError(f"点 {i} の1行目に設定する方程式がありません")
            
            # 2行目: ノイマン境界または次の補助方程式 (ψ'')
            if eq_by_type["neumann"]:
                assignments[2] = eq_by_type["neumann"]
            elif eq_by_type["auxiliary"]:
                assignments[2] = eq_by_type["auxiliary"].pop(0)
            else:
                raise ValueError(f"点 {i} の2行目に設定する方程式がありません")
            
            # 3行目: 残りの補助方程式 (ψ''')
            if eq_by_type["auxiliary"]:
                assignments[3] = eq_by_type["auxiliary"].pop(0)
            else:
                raise ValueError(f"点 {i} の3行目に設定する方程式がありません")
            
            # 各行の方程式から係数を行列に追加
            for row_offset, eq in enumerate(assignments):
                row = base_idx + row_offset
                
                # ステンシル係数を取得して行列に追加
                stencil_coeffs = eq.get_stencil_coefficients(i=i)
                for offset, coeffs in stencil_coeffs.items():
                    j = i + offset
                    if 0 <= j < n:  # グリッド範囲内チェック
                        col_base = j * var_per_point
                        for k, coeff in enumerate(coeffs):
                            if coeff != 0.0:  # 非ゼロ要素のみ追加
                                row_indices.append(row)
                                col_indices.append(col_base + k)
                                # CuPy配列があればNumPyに変換
                                data.append(float(self._to_numpy(coeff)))
        
        # SciPyを使用してCSR行列を構築
        A = sp_cpu.csr_matrix(
            (np.array(data), (np.array(row_indices), np.array(col_indices))), 
            shape=(system_size, system_size)
        )
        
        return A
