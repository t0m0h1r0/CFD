import cupy as cp
import cupyx.scipy.sparse as sp

class EquationSystem2D:
    """2次元方程式システムを管理するクラス"""
    
    def __init__(self, grid):
        """
        グリッドを指定して初期化
        
        Args:
            grid: Grid2D オブジェクト
        """
        self.grid = grid
        
        # 内部点の方程式
        self.interior_equations = []
        
        # 境界の方程式
        self.left_boundary_equations = []    # i = 0
        self.right_boundary_equations = []   # i = nx-1
        self.bottom_boundary_equations = []  # j = 0
        self.top_boundary_equations = []     # j = ny-1
    
    def add_interior_equation(self, equation):
        """内部点の方程式を追加"""
        self.interior_equations.append(equation)
    
    def add_left_boundary_equation(self, equation):
        """左境界の方程式を追加 (i=0)"""
        self.left_boundary_equations.append(equation)
    
    def add_right_boundary_equation(self, equation):
        """右境界の方程式を追加 (i=nx-1)"""
        self.right_boundary_equations.append(equation)
    
    def add_bottom_boundary_equation(self, equation):
        """下境界の方程式を追加 (j=0)"""
        self.bottom_boundary_equations.append(equation)
    
    def add_top_boundary_equation(self, equation):
        """上境界の方程式を追加 (j=ny-1)"""
        self.top_boundary_equations.append(equation)
    
    def is_left_boundary(self, i, j):
        """左境界かどうかを判定"""
        return i == 0
    
    def is_right_boundary(self, i, j):
        """右境界かどうかを判定"""
        nx = self.grid.nx_points
        return i == nx - 1
    
    def is_bottom_boundary(self, i, j):
        """下境界かどうかを判定"""
        return j == 0
    
    def is_top_boundary(self, i, j):
        """上境界かどうかを判定"""
        ny = self.grid.ny_points
        return j == ny - 1
    
    def is_interior(self, i, j):
        """内部点かどうかを判定"""
        nx, ny = self.grid.nx_points, self.grid.ny_points
        return 0 < i < nx - 1 and 0 < j < ny - 1
    
    def build_matrix_system(self):
        """行列システムを構築"""
        nx, ny = self.grid.nx_points, self.grid.ny_points
        
        # 7 unknowns per grid point: ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy
        n_unknowns_per_point = 7
        system_size = n_unknowns_per_point * nx * ny
        
        data = []
        row_indices = []
        col_indices = []
        b = cp.zeros(system_size)
        
        # 全グリッド点に対して
        for j in range(ny):
            for i in range(nx):
                # グリッド点の基本インデックス
                base_idx = (j * nx + i) * n_unknowns_per_point
                
                # この点で適用する方程式リスト
                applicable_equations = []
                
                # 点のタイプに基づいて適用する方程式を決定
                if self.is_interior(i, j):
                    # 内部点の場合は内部方程式のみ
                    applicable_equations = self.interior_equations
                else:
                    # 境界点の場合、適用される境界条件を収集
                    if self.is_left_boundary(i, j):
                        applicable_equations.extend(self.left_boundary_equations)
                    
                    if self.is_right_boundary(i, j):
                        applicable_equations.extend(self.right_boundary_equations)
                    
                    if self.is_bottom_boundary(i, j):
                        applicable_equations.extend(self.bottom_boundary_equations)
                    
                    if self.is_top_boundary(i, j):
                        applicable_equations.extend(self.top_boundary_equations)
                
                # 有効な方程式のカウンター
                eq_count = 0
                
                # 各方程式を適用
                for eq in applicable_equations:
                    if eq.is_valid_at(self.grid, i, j):
                        # 最大7つまで（オーバーフロー防止）
                        if eq_count >= n_unknowns_per_point:
                            break
                        
                        # 行インデックス
                        row = base_idx + eq_count
                        eq_count += 1
                        
                        # ステンシル係数を取得
                        stencil_coeffs = eq.get_stencil_coefficients(self.grid, i, j)
                        
                        # 行列に係数を追加
                        for (di, dj), coeffs in stencil_coeffs.items():
                            ni, nj = i + di, j + dj
                            if 0 <= ni < nx and 0 <= nj < ny:
                                col_base = (nj * nx + ni) * n_unknowns_per_point
                                for m, coeff in enumerate(coeffs):
                                    if coeff != 0.0:
                                        row_indices.append(row)
                                        col_indices.append(col_base + m)
                                        data.append(float(coeff))
                        
                        # 右辺値を設定
                        b[row] = eq.get_rhs(self.grid, i, j)
                
                # 方程式が不足している場合、単位行列で補完
                if eq_count < n_unknowns_per_point:
                    for k in range(eq_count, n_unknowns_per_point):
                        row = base_idx + k
                        row_indices.append(row)
                        col_indices.append(row)  # 同じインデックスで対角成分
                        data.append(1.0)  # 単位行列の要素
        
        # 疎行列を作成
        A = sp.csr_matrix(
            (cp.array(data), (cp.array(row_indices), cp.array(col_indices))),
            shape=(system_size, system_size)
        )
        
        return A, b
    
    def analyze_sparsity(self):
        """行列の疎性を分析"""
        A, _ = self.build_matrix_system()
        
        total_size = A.shape[0]
        nnz = A.nnz
        max_possible_nnz = total_size * total_size
        sparsity = 1.0 - (nnz / max_possible_nnz)
        
        memory_dense_MB = (total_size * total_size * 8) / (1024 * 1024)  # 8 bytes per double
        memory_sparse_MB = (nnz * 12) / (1024 * 1024)  # 8 bytes for value + 4 bytes for indices
        
        results = {
            "matrix_size": total_size,
            "non_zeros": nnz,
            "sparsity": sparsity,
            "memory_dense_MB": memory_dense_MB,
            "memory_sparse_MB": memory_sparse_MB
        }
        
        return results