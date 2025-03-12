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

    def validate_equation_system(self):
        """方程式システムの妥当性を検証"""
        checks = []
        
        # 1D/2Dに応じて異なるチェック
        if not self.is_2d:
            checks = [
                (self.left_boundary_equations, "左境界方程式"),
                (self.interior_equations, "内部方程式"),
                (self.right_boundary_equations, "右境界方程式")
            ]
        else:
            checks = [
                (self.left_boundary_equations, "左境界方程式"),
                (self.right_boundary_equations, "右境界方程式"),
                (self.bottom_boundary_equations, "下境界方程式"),
                (self.top_boundary_equations, "上境界方程式"),
                (self.interior_equations, "内部方程式")
            ]
        
        for equations, description in checks:
            if not equations:
                raise ValueError(f"{description}が設定されていません。")

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
    def is_left_boundary(self, i, j=None):
        """左境界かどうかを判定（角を除く）(2D only)"""
        if not self.is_2d:
            raise ValueError("2D専用のメソッドが1Dグリッドで呼び出されました")
            
        return i == 0 and 0 < j < self.grid.ny_points - 1
    
    def is_right_boundary(self, i, j=None):
        """右境界かどうかを判定（角を除く）(2D only)"""
        if not self.is_2d:
            raise ValueError("2D専用のメソッドが1Dグリッドで呼び出されました")
            
        nx = self.grid.nx_points
        return i == nx - 1 and 0 < j < self.grid.ny_points - 1
    
    def is_bottom_boundary(self, i, j=None):
        """下境界かどうかを判定（角を除く）(2D only)"""
        if not self.is_2d:
            raise ValueError("2D専用のメソッドが1Dグリッドで呼び出されました")
            
        return j == 0 and 0 < i < self.grid.nx_points - 1
    
    def is_top_boundary(self, i, j=None):
        """上境界かどうかを判定（角を除く）(2D only)"""
        if not self.is_2d:
            raise ValueError("2D専用のメソッドが1Dグリッドで呼び出されました")
            
        ny = self.grid.ny_points
        return j == ny - 1 and 0 < i < self.grid.nx_points - 1
    
    def is_left_bottom_corner(self, i, j=None):
        """左下の角かどうかを判定 (2D only)"""
        if not self.is_2d:
            raise ValueError("2D専用のメソッドが1Dグリッドで呼び出されました")
            
        return i == 0 and j == 0
    
    def is_right_bottom_corner(self, i, j=None):
        """右下の角かどうかを判定 (2D only)"""
        if not self.is_2d:
            raise ValueError("2D専用のメソッドが1Dグリッドで呼び出されました")
            
        return i == self.grid.nx_points - 1 and j == 0
    
    def is_left_top_corner(self, i, j=None):
        """左上の角かどうかを判定 (2D only)"""
        if not self.is_2d:
            raise ValueError("2D専用のメソッドが1Dグリッドで呼び出されました")
            
        return i == 0 and j == self.grid.ny_points - 1
    
    def is_right_top_corner(self, i, j=None):
        """右上の角かどうかを判定 (2D only)"""
        if not self.is_2d:
            raise ValueError("2D専用のメソッドが1Dグリッドで呼び出されました")
            
        return i == self.grid.nx_points - 1 and j == self.grid.ny_points - 1

    def _build_matrix_system_1d(self):
        """1D行列システムの構築"""
        # 境界条件のタイプを判別するために必要なクラスをインポート
        from equation.poisson import PoissonEquation
        from equation.original import OriginalEquation
        from equation.boundary import (
            DirichletBoundaryEquation, NeumannBoundaryEquation
        )
        
        # システムの妥当性を検証
        self.validate_equation_system()
        
        n = self.grid.n_points
        size = 4 * n  # 1Dは従来通り4変数
        
        data = []
        row_indices = []
        col_indices = []
        b = cp.zeros(size)

        for i in range(n):
            # 基本インデックス
            base_idx = i * 4
            
            # この点で適用する方程式リスト
            if i == 0:
                applicable_equations = self.left_boundary_equations
            elif i == n - 1:
                applicable_equations = self.right_boundary_equations
            else:
                applicable_equations = self.interior_equations
            
            # 各タイプの方程式を格納するリスト
            governing_eq = None  # 支配方程式
            dirichlet_eq = None  # Dirichlet境界条件
            neumann_eq = None    # Neumann境界条件
            aux_equations = []   # 補助方程式
            
            # 方程式をタイプごとに分類
            for eq in applicable_equations:
                if eq.is_valid_at(i=i):
                    if isinstance(eq, PoissonEquation):
                        governing_eq = eq
                    elif isinstance(eq, OriginalEquation):
                        governing_eq = eq
                    elif isinstance(eq, DirichletBoundaryEquation):
                        dirichlet_eq = eq
                    elif isinstance(eq, NeumannBoundaryEquation):
                        neumann_eq = eq
                    else:
                        aux_equations.append(eq)
            
            # 方程式が見つからない場合は例外
            if not governing_eq:
                raise ValueError(f"点 {i} に支配方程式が見つかりません。方程式システムを確認してください。")
            
            # 1: ψ' の方程式
            if not dirichlet_eq and not aux_equations:
                raise ValueError(f"点 {i} にψ'の方程式が見つかりません。")
            
            # 2: ψ'' の方程式
            if not neumann_eq and not aux_equations:
                raise ValueError(f"点 {i} にψ''の方程式が見つかりません。")
            
            # 3: ψ''' の方程式
            if not aux_equations:
                raise ValueError(f"点 {i} にψ'''の方程式が見つかりません。")
            
            # 1D用に方程式を特定の順序で配置
            # 0: ψ - 常に支配方程式
            self._add_equation_to_matrix_1d(governing_eq, i, base_idx, 0, data, row_indices, col_indices, b)
            
            # 1: ψ' - ディリクレ境界または補助方程式
            if dirichlet_eq:
                self._add_equation_to_matrix_1d(dirichlet_eq, i, base_idx, 1, data, row_indices, col_indices, b)
            else:
                self._add_equation_to_matrix_1d(aux_equations.pop(0), i, base_idx, 1, data, row_indices, col_indices, b)
            
            # 2: ψ'' - ノイマン境界または補助方程式
            if neumann_eq:
                self._add_equation_to_matrix_1d(neumann_eq, i, base_idx, 2, data, row_indices, col_indices, b)
            else:
                self._add_equation_to_matrix_1d(aux_equations.pop(0), i, base_idx, 2, data, row_indices, col_indices, b)
            
            # 3: ψ''' - 補助方程式
            self._add_equation_to_matrix_1d(aux_equations.pop(0), i, base_idx, 3, data, row_indices, col_indices, b)

        A = sp.csr_matrix(
            (cp.array(data), (cp.array(row_indices), cp.array(col_indices))), 
            shape=(size, size)
        )

        return A, b

    def _build_matrix_system_2d(self):
        """2D行列システムの構築"""
        # 境界条件のタイプを判別するために必要なクラスをインポート
        from equation.poisson import PoissonEquation2D
        from equation.original import OriginalEquation2D
        from equation.boundary import (
            DirichletXBoundaryEquation2D, DirichletYBoundaryEquation2D,
            NeumannXBoundaryEquation2D, NeumannYBoundaryEquation2D
        )
        
        # システムの妥当性を検証
        self.validate_equation_system()
        
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
                
                # 各タイプの方程式を格納するリスト
                governing_eq = None  # 支配方程式
                dirichlet_x_eq = None  # Dirichlet X
                neumann_x_eq = None  # Neumann X
                dirichlet_y_eq = None  # Dirichlet Y
                neumann_y_eq = None  # Neumann Y
                aux_equations = []  # 補助方程式
                
                # 方程式をタイプごとに分類
                for eq in applicable_equations:
                    if eq.is_valid_at(i=i, j=j):
                        if isinstance(eq, PoissonEquation2D):
                            governing_eq = eq
                        elif isinstance(eq, OriginalEquation2D):
                            governing_eq = eq
                        elif isinstance(eq, DirichletXBoundaryEquation2D):
                            dirichlet_x_eq = eq
                        elif isinstance(eq, NeumannXBoundaryEquation2D):
                            neumann_x_eq = eq
                        elif isinstance(eq, DirichletYBoundaryEquation2D):
                            dirichlet_y_eq = eq
                        elif isinstance(eq, NeumannYBoundaryEquation2D):
                            neumann_y_eq = eq
                        else:
                            aux_equations.append(eq)
                
                # 方程式が見つからない場合は例外
                if not governing_eq:
                    raise ValueError(f"点 ({i}, {j}) に支配方程式が見つかりません。方程式システムを確認してください。")
                
                # 各位置に方程式を配置
                # 0: ψ - 常に支配方程式
                self._add_equation_to_matrix_2d(governing_eq, i, j, base_idx, 0, data, row_indices, col_indices, b)
                
                # 1: ψ_x - x方向のディリクレ境界または補助方程式
                if dirichlet_x_eq:
                    self._add_equation_to_matrix_2d(dirichlet_x_eq, i, j, base_idx, 1, data, row_indices, col_indices, b)
                else:
                    self._add_equation_to_matrix_2d(aux_equations.pop(0), i, j, base_idx, 1, data, row_indices, col_indices, b)
                
                # 2: ψ_xx - x方向のノイマン境界または補助方程式
                if neumann_x_eq:
                    self._add_equation_to_matrix_2d(neumann_x_eq, i, j, base_idx, 2, data, row_indices, col_indices, b)
                else:
                    self._add_equation_to_matrix_2d(aux_equations.pop(0), i, j, base_idx, 2, data, row_indices, col_indices, b)
                
                # 3: ψ_xxx - 補助方程式
                self._add_equation_to_matrix_2d(aux_equations.pop(0), i, j, base_idx, 3, data, row_indices, col_indices, b)
                
                # 4: ψ_y - y方向のディリクレ境界または補助方程式
                if dirichlet_y_eq:
                    self._add_equation_to_matrix_2d(dirichlet_y_eq, i, j, base_idx, 4, data, row_indices, col_indices, b)
                else:
                    self._add_equation_to_matrix_2d(aux_equations.pop(0), i, j, base_idx, 4, data, row_indices, col_indices, b)
                
                # 5: ψ_yy - y方向のノイマン境界または補助方程式
                if neumann_y_eq:
                    self._add_equation_to_matrix_2d(neumann_y_eq, i, j, base_idx, 5, data, row_indices, col_indices, b)
                else:
                    self._add_equation_to_matrix_2d(aux_equations.pop(0), i, j, base_idx, 5, data, row_indices, col_indices, b)
                
                # 6: ψ_yyy - 補助方程式
                self._add_equation_to_matrix_2d(aux_equations.pop(0), i, j, base_idx, 6, data, row_indices, col_indices, b)
        
        # 疎行列を作成
        A = sp.csr_matrix(
            (cp.array(data), (cp.array(row_indices), cp.array(col_indices))),
            shape=(system_size, system_size)
        )
        
        return A, b

    def _add_equation_to_matrix_1d(self, eq, i, base_idx, row_offset, data, row_indices, col_indices, b):
        """
        特定の行オフセットに1D方程式を追加
        
        Args:
            eq: 追加する方程式
            i: グリッド点インデックス
            base_idx: 現在のグリッド点の基本インデックス
            row_offset: グリッド点内の変数オフセット (0-3)
            data: 行列値のリスト
            row_indices: 行インデックスのリスト
            col_indices: 列インデックスのリスト
            b: 右辺ベクトル
        """
        # 行インデックス
        row = base_idx + row_offset
        
        # ステンシル係数を取得
        stencil_coeffs = eq.get_stencil_coefficients(i=i)
        
        # 行列に係数を追加
        n = self.grid.n_points
        for offset, coeffs in stencil_coeffs.items():
            j = i + offset
            if 0 <= j < n:
                col_base = j * 4
                for m, coeff in enumerate(coeffs):
                    if coeff != 0.0:
                        row_indices.append(row)
                        col_indices.append(col_base + m)
                        data.append(float(coeff))
        
        # 右辺値を設定
        b[row] = eq.get_rhs(i=i)

    def _add_equation_to_matrix_2d(self, eq, i, j, base_idx, row_offset, data, row_indices, col_indices, b):
        """
        特定の行オフセットに2D方程式を追加
        
        Args:
            eq: 追加する方程式
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
            base_idx: 現在のグリッド点の基本インデックス
            row_offset: グリッド点内の変数オフセット (0-6)
            data: 行列値のリスト
            row_indices: 行インデックスのリスト
            col_indices: 列インデックスのリスト
            b: 右辺ベクトル
        """
        # 行インデックス
        row = base_idx + row_offset
        
        # ステンシル係数を取得
        stencil_coeffs = eq.get_stencil_coefficients(i=i, j=j)
        
        # 行列に係数を追加
        nx = self.grid.nx_points
        for (di, dj), coeffs in stencil_coeffs.items():
            ni, nj = i + di, j + dj
            if 0 <= ni < nx and 0 <= nj < self.grid.ny_points:
                col_base = (nj * nx + ni) * 7
                for m, coeff in enumerate(coeffs):
                    if coeff != 0.0:
                        row_indices.append(row)
                        col_indices.append(col_base + m)
                        data.append(float(coeff))
        
        # 右辺値を設定
        b[row] = eq.get_rhs(i=i, j=j)

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