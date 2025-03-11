import cupy as cp
import cupyx.scipy.sparse as sp

class EquationSystem:
    """1D/2D 両対応の方程式システムを管理するクラス"""

    def __init__(self, grid):
        """
        方程式システムを初期化
        
        Args:
            grid: Grid (1D/2D) オブジェクト
        """
        self.grid = grid
        
        # 1D or 2D mode判定
        self.is_2d = grid.is_2d
        
        if self.is_2d:
            # 2D specific variables
            # 内部点の方程式
            self.interior_equations = []
            
            # 境界の方程式
            self.left_boundary_equations = []    # i = 0
            self.right_boundary_equations = []   # i = nx-1
            self.bottom_boundary_equations = []  # j = 0
            self.top_boundary_equations = []     # j = ny-1
            
            # 角の方程式（より詳細な境界条件管理用）
            self.left_bottom_equations = []      # i = 0, j = 0
            self.right_bottom_equations = []     # i = nx-1, j = 0
            self.left_top_equations = []         # i = 0, j = ny-1
            self.right_top_equations = []        # i = nx-1, j = ny-1
        else:
            # 1D specific variables
            self.left_boundary_equations = []
            self.interior_equations = []
            self.right_boundary_equations = []

    def add_left_boundary_equation(self, equation):
        """
        左境界方程式を追加
        
        Args:
            equation: 方程式オブジェクト
        """
        # 方程式にグリッドを設定
        if hasattr(equation, 'set_grid'):
            equation.set_grid(self.grid)
            
        self.left_boundary_equations.append(equation)
        
        # 2Dの場合、左側の角にも同じ方程式を追加
        if self.is_2d:
            self.left_bottom_equations.append(equation)
            self.left_top_equations.append(equation)

    def add_interior_equation(self, equation):
        """
        内部方程式を追加
        
        Args:
            equation: 方程式オブジェクト
        """
        # 方程式にグリッドを設定
        if hasattr(equation, 'set_grid'):
            equation.set_grid(self.grid)
            
        self.interior_equations.append(equation)

    def add_right_boundary_equation(self, equation):
        """
        右境界方程式を追加
        
        Args:
            equation: 方程式オブジェクト
        """
        # 方程式にグリッドを設定
        if hasattr(equation, 'set_grid'):
            equation.set_grid(self.grid)
            
        self.right_boundary_equations.append(equation)
        
        # 2Dの場合、右側の角にも同じ方程式を追加
        if self.is_2d:
            self.right_bottom_equations.append(equation)
            self.right_top_equations.append(equation)

    def add_equation(self, equation):
        """
        全ての領域に方程式を追加
        
        Args:
            equation: 方程式オブジェクト
        """
        # 方程式にグリッドを設定
        if hasattr(equation, 'set_grid'):
            equation.set_grid(self.grid)
            
        self.left_boundary_equations.append(equation)
        self.interior_equations.append(equation)
        self.right_boundary_equations.append(equation)
        
        # 2Dの場合、その他の境界と角にも追加
        if self.is_2d:
            self.bottom_boundary_equations.append(equation)
            self.top_boundary_equations.append(equation)
            self.left_bottom_equations.append(equation)
            self.right_bottom_equations.append(equation)
            self.left_top_equations.append(equation)
            self.right_top_equations.append(equation)

    # 2D固有のメソッド群
    def add_interior_x_equation(self, equation):
        """内部点のx方向の方程式を追加 (2D only)"""
        if not self.is_2d:
            raise ValueError("2D専用のメソッドが1Dグリッドで呼び出されました")
            
        # 方程式にグリッドを設定
        if hasattr(equation, 'set_grid'):
            equation.set_grid(self.grid)
            
        self.interior_equations.append(equation)
        self.bottom_boundary_equations.append(equation)
        self.top_boundary_equations.append(equation)
    
    def add_interior_y_equation(self, equation):
        """内部点のy方向の方程式を追加 (2D only)"""
        if not self.is_2d:
            raise ValueError("2D専用のメソッドが1Dグリッドで呼び出されました")
            
        # 方程式にグリッドを設定
        if hasattr(equation, 'set_grid'):
            equation.set_grid(self.grid)
            
        self.interior_equations.append(equation)
        self.left_boundary_equations.append(equation)
        self.right_boundary_equations.append(equation)

    def add_bottom_boundary_equation(self, equation):
        """下境界の方程式を追加 (j=0) (2D only)"""
        if not self.is_2d:
            raise ValueError("2D専用のメソッドが1Dグリッドで呼び出されました")
            
        # 方程式にグリッドを設定
        if hasattr(equation, 'set_grid'):
            equation.set_grid(self.grid)
            
        self.bottom_boundary_equations.append(equation)
        # 下側の角にも同じ方程式を追加
        self.left_bottom_equations.append(equation)
        self.right_bottom_equations.append(equation)
    
    def add_top_boundary_equation(self, equation):
        """上境界の方程式を追加 (j=ny-1) (2D only)"""
        if not self.is_2d:
            raise ValueError("2D専用のメソッドが1Dグリッドで呼び出されました")
            
        # 方程式にグリッドを設定
        if hasattr(equation, 'set_grid'):
            equation.set_grid(self.grid)
            
        self.top_boundary_equations.append(equation)
        # 上側の角にも同じ方程式を追加
        self.left_top_equations.append(equation)
        self.right_top_equations.append(equation)

    # 位置判定メソッド（1D/2D両対応）
    def is_boundary_point(self, i, j=None):
        """境界点かどうかを判定"""
        return self.grid.is_boundary_point(i, j)
        
    def is_interior_point(self, i, j=None):
        """内部点かどうかを判定"""
        return self.grid.is_interior_point(i, j)

    # 2D用の位置判定メソッド
    def is_left_boundary(self, i, j):
        """左境界かどうかを判定（角を除く）(2D only)"""
        if not self.is_2d:
            raise ValueError("2D専用のメソッドが1Dグリッドで呼び出されました")
            
        return i == 0 and 0 < j < self.grid.ny_points - 1
    
    def is_right_boundary(self, i, j):
        """右境界かどうかを判定（角を除く）(2D only)"""
        if not self.is_2d:
            raise ValueError("2D専用のメソッドが1Dグリッドで呼び出されました")
            
        nx = self.grid.nx_points
        return i == nx - 1 and 0 < j < self.grid.ny_points - 1
    
    def is_bottom_boundary(self, i, j):
        """下境界かどうかを判定（角を除く）(2D only)"""
        if not self.is_2d:
            raise ValueError("2D専用のメソッドが1Dグリッドで呼び出されました")
            
        return j == 0 and 0 < i < self.grid.nx_points - 1
    
    def is_top_boundary(self, i, j):
        """上境界かどうかを判定（角を除く）(2D only)"""
        if not self.is_2d:
            raise ValueError("2D専用のメソッドが1Dグリッドで呼び出されました")
            
        ny = self.grid.ny_points
        return j == ny - 1 and 0 < i < self.grid.nx_points - 1
    
    def is_left_bottom_corner(self, i, j):
        """左下の角かどうかを判定 (2D only)"""
        if not self.is_2d:
            raise ValueError("2D専用のメソッドが1Dグリッドで呼び出されました")
            
        return i == 0 and j == 0
    
    def is_right_bottom_corner(self, i, j):
        """右下の角かどうかを判定 (2D only)"""
        if not self.is_2d:
            raise ValueError("2D専用のメソッドが1Dグリッドで呼び出されました")
            
        return i == self.grid.nx_points - 1 and j == 0
    
    def is_left_top_corner(self, i, j):
        """左上の角かどうかを判定 (2D only)"""
        if not self.is_2d:
            raise ValueError("2D専用のメソッドが1Dグリッドで呼び出されました")
            
        return i == 0 and j == self.grid.ny_points - 1
    
    def is_right_top_corner(self, i, j):
        """右上の角かどうかを判定 (2D only)"""
        if not self.is_2d:
            raise ValueError("2D専用のメソッドが1Dグリッドで呼び出されました")
            
        return i == self.grid.nx_points - 1 and j == self.grid.ny_points - 1

    def build_matrix_system(self):
        """
        行列システムを構築
        
        Returns:
            Tuple[sp.csr_matrix, cp.ndarray]: システム行列と右辺ベクトル
        """
        if self.is_2d:
            return self._build_matrix_system_2d()
        else:
            return self._build_matrix_system_1d()

    def _build_matrix_system_1d(self):
        """1D行列システムの構築"""
        n = self.grid.n_points
        size = 4 * n
        
        data = []
        row_indices = []
        col_indices = []
        b = cp.zeros(size)

        for i in range(n):
            if i == 0:
                equations = self.left_boundary_equations
            elif i == n - 1:
                equations = self.right_boundary_equations
            else:
                equations = self.interior_equations

            if len(equations) < 4:
                # 不足している場合の処理を追加する
                empty_count = 4 - len(equations)
                for _ in range(empty_count):
                    row = i * 4 + len(equations)
                    row_indices.append(row)
                    col_indices.append(row)  # 対角要素
                    data.append(1.0)  # 単位行列
                    b[row] = 0.0
                continue

            for k, eq in enumerate(equations[:4]):  # 最大4つの方程式を使用
                # 新しいインターフェースのみを使用
                stencil_coeffs = eq.get_stencil_coefficients(i=i)
                rhs_value = eq.get_rhs(i=i)

                for offset, coeffs in stencil_coeffs.items():
                    j = i + offset
                    if 0 <= j < n:
                        for m, coeff in enumerate(coeffs):
                            if coeff != 0.0:
                                row_indices.append(i * 4 + k)
                                col_indices.append(j * 4 + m)
                                data.append(float(coeff))

                b[i * 4 + k] = rhs_value

        A = sp.csr_matrix(
            (cp.array(data), (cp.array(row_indices), cp.array(col_indices))), 
            shape=(size, size)
        )

        return A, b

    def _build_matrix_system_2d(self):
        """2D行列システムの構築"""
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
                if self.is_interior_point(i, j):
                    # 内部点
                    applicable_equations = self.interior_equations
                elif self.is_left_bottom_corner(i, j):
                    # 左下の角
                    applicable_equations = self.left_bottom_equations
                elif self.is_right_bottom_corner(i, j):
                    # 右下の角
                    applicable_equations = self.right_bottom_equations
                elif self.is_left_top_corner(i, j):
                    # 左上の角
                    applicable_equations = self.left_top_equations
                elif self.is_right_top_corner(i, j):
                    # 右上の角
                    applicable_equations = self.right_top_equations
                elif self.is_left_boundary(i, j):
                    # 左境界（角を除く）
                    applicable_equations = self.left_boundary_equations
                elif self.is_right_boundary(i, j):
                    # 右境界（角を除く）
                    applicable_equations = self.right_boundary_equations
                elif self.is_bottom_boundary(i, j):
                    # 下境界（角を除く）
                    applicable_equations = self.bottom_boundary_equations
                elif self.is_top_boundary(i, j):
                    # 上境界（角を除く）
                    applicable_equations = self.top_boundary_equations
                
                # 有効な方程式のカウンター
                eq_count = 0
                
                # 各方程式を適用
                for eq in applicable_equations:
                    if eq.is_valid_at(i=i, j=j):
                        # 最大7つまで（オーバーフロー防止）
                        if eq_count >= n_unknowns_per_point:
                            break
                        
                        # 行インデックス
                        row = base_idx + eq_count
                        eq_count += 1
                        
                        # ステンシル係数を取得（新インターフェースのみ使用）
                        stencil_coeffs = eq.get_stencil_coefficients(i=i, j=j)
                        
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
                        
                        # 右辺値を設定（新インターフェースのみ使用）
                        b[row] = eq.get_rhs(i=i, j=j)
                
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