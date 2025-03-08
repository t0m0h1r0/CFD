# equation_system.py
import cupy as cp
import cupyx.scipy.sparse as sp
from typing import List, Tuple, Dict
from equation.base import Equation
from grid import Grid


class EquationSystem:
    """方程式システムを管理するクラス (スパース行列最適化版)"""

    def __init__(self, grid: Grid):
        """
        初期化

        Args:
            grid: 計算格子
        """
        self.grid = grid
        # 位置ごとに方程式をグループ化
        self.left_boundary_equations: List[Equation] = []
        self.interior_equations: List[Equation] = []
        self.right_boundary_equations: List[Equation] = []

    def add_left_boundary_equation(self, equation: Equation) -> None:
        """左境界用の方程式を追加"""
        self.left_boundary_equations.append(equation)

    def add_interior_equation(self, equation: Equation) -> None:
        """内部点用の方程式を追加"""
        self.interior_equations.append(equation)

    def add_right_boundary_equation(self, equation: Equation) -> None:
        """右境界用の方程式を追加"""
        self.right_boundary_equations.append(equation)

    def add_equation(self, equation: Equation) -> None:
        self.left_boundary_equations.append(equation)
        self.interior_equations.append(equation)
        self.right_boundary_equations.append(equation)

    def build_matrix_system(self) -> Tuple[sp.csr_matrix, cp.ndarray]:
        """
        スパース行列システムを構築
        
        Returns:
            Tuple[sp.csr_matrix, cp.ndarray]: スパース行列Aとベクトルbのタプル
        """
        n = self.grid.n_points

        # マトリックスサイズ: 各点に4つの値 (psi, psi', psi'', psi''') がある
        size = 4 * n
        
        # 疎行列用の座標形式データを格納するリスト
        data = []      # 非ゼロ値
        row_indices = []  # 行インデックス
        col_indices = []  # 列インデックス
        
        # 右辺ベクトル
        b = cp.zeros(size)

        # 各格子点での方程式を評価
        for i in range(n):
            equations_to_use = []

            if i == 0:  # 左境界
                equations_to_use = self.left_boundary_equations
            elif i == n - 1:  # 右境界
                equations_to_use = self.right_boundary_equations
            else:  # 内部点
                equations_to_use = self.interior_equations

            # この点での方程式がない場合はエラー
            if not equations_to_use:
                raise ValueError(f"点 {i} に対する方程式が設定されていません")

            # 4つの方程式（ψ, ψ', ψ'', ψ'''の各状態に対応）が必要
            if len(equations_to_use) != 4:
                raise ValueError(
                    f"点 {i} に対する方程式が4つではありません（{len(equations_to_use)}個）"
                )

            # 各方程式を行列に反映
            for k, eq in enumerate(equations_to_use):
                # この方程式がこの点に適用可能か確認
                if not eq.is_valid_at(self.grid, i):
                    raise ValueError(f"方程式 {k} は点 {i} に適用できません")

                # 方程式のステンシル係数を取得
                stencil_coeffs = eq.get_stencil_coefficients(self.grid, i)
                rhs_value = eq.get_rhs(self.grid, i)

                # 各ステンシル点の係数を行列に設定
                for offset, coeffs in stencil_coeffs.items():
                    j = i + offset  # ステンシル点のインデックス

                    # インデックスが有効範囲内かチェック
                    if 0 <= j < n:
                        # 係数を疎行列形式で格納
                        for l, coeff in enumerate(coeffs):
                            if coeff != 0.0:  # ゼロでない要素のみ格納
                                row_indices.append(i * 4 + k)
                                col_indices.append(j * 4 + l)
                                data.append(coeff)

                # 右辺ベクトルを設定
                b[i * 4 + k] = rhs_value

        # COO形式からCSR形式のスパース行列を構築
        A = sp.csr_matrix(
            (cp.array(data), (cp.array(row_indices), cp.array(col_indices))), 
            shape=(size, size)
        )

        return A, b

    def analyze_sparsity(self) -> Dict:
        """
        行列の疎性を分析し、統計情報を返す
        
        Returns:
            Dict: 疎性の統計情報を含む辞書
        """
        A, _ = self.build_matrix_system()
        size = A.shape[0]
        nnz = A.nnz  # 非ゼロ要素数
        density = nnz / (size * size)
        
        # ステンシル構造の分析（行ごとの非ゼロ要素数の平均と最大）
        row_nnz = cp.diff(A.indptr)
        avg_row_nnz = float(cp.mean(row_nnz))
        max_row_nnz = int(cp.max(row_nnz))
        
        # バンド幅の推定
        if nnz > 0:
            row_indices = cp.zeros(nnz, dtype=cp.int32)
            for i in range(size):
                row_indices[A.indptr[i]:A.indptr[i+1]] = i
            col_indices = A.indices
            bandwidth = int(cp.max(cp.abs(row_indices - col_indices))) + 1
        else:
            bandwidth = 0
            
        return {
            "size": size,
            "nonzeros": nnz,
            "density": density,
            "avg_nonzeros_per_row": avg_row_nnz,
            "max_nonzeros_per_row": max_row_nnz,
            "estimated_bandwidth": bandwidth,
            "memory_dense_MB": size * size * 8 / (1024 * 1024),  # 8バイト/要素と仮定
            "memory_sparse_MB": (nnz + size + 1) * 8 / (1024 * 1024)  # CSR形式の概算
        }