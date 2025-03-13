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
        self.is_2d = grid.is_2d
        
        # 方程式のコレクションを初期化
        if self.is_2d:
            # 2D specific variables
            self.interior_equations = []
            self.left_boundary_equations = []    # i = 0
            self.right_boundary_equations = []   # i = nx-1
            self.bottom_boundary_equations = []  # j = 0
            self.top_boundary_equations = []     # j = ny-1
            self.corner_equations = {
                'left_bottom': [],   # i = 0, j = 0
                'right_bottom': [],  # i = nx-1, j = 0
                'left_top': [],      # i = 0, j = ny-1
                'right_top': []      # i = nx-1, j = ny-1
            }
        else:
            # 1D specific variables
            self.left_boundary_equations = []
            self.interior_equations = []
            self.right_boundary_equations = []

    def validate_equation_system(self):
        """方程式システムの妥当性を検証"""
        # 1D/2Dに応じて異なるチェック
        if not self.is_2d:
            required_equations = [
                (self.left_boundary_equations, "左境界方程式"),
                (self.interior_equations, "内部方程式"),
                (self.right_boundary_equations, "右境界方程式")
            ]
        else:
            required_equations = [
                (self.left_boundary_equations, "左境界方程式"),
                (self.right_boundary_equations, "右境界方程式"),
                (self.bottom_boundary_equations, "下境界方程式"),
                (self.top_boundary_equations, "上境界方程式"),
                (self.interior_equations, "内部方程式")
            ]
        
        for equations, description in required_equations:
            if not equations:
                raise ValueError(f"{description}が設定されていません。")

    def _set_grid_to_equation(self, equation):
        """方程式にグリッドを設定（必要な場合）"""
        if hasattr(equation, 'set_grid'):
            equation.set_grid(self.grid)
        return equation

    def add_left_boundary_equation(self, equation):
        """左境界方程式を追加"""
        equation = self._set_grid_to_equation(equation)
        self.left_boundary_equations.append(equation)
        
        # 2Dの場合、左側の角にも同じ方程式を追加
        if self.is_2d:
            self.corner_equations['left_bottom'].append(equation)
            self.corner_equations['left_top'].append(equation)

    def add_interior_equation(self, equation):
        """内部方程式を追加"""
        equation = self._set_grid_to_equation(equation)
        self.interior_equations.append(equation)

    def add_right_boundary_equation(self, equation):
        """右境界方程式を追加"""
        equation = self._set_grid_to_equation(equation)
        self.right_boundary_equations.append(equation)
        
        # 2Dの場合、右側の角にも同じ方程式を追加
        if self.is_2d:
            self.corner_equations['right_bottom'].append(equation)
            self.corner_equations['right_top'].append(equation)

    def add_equation(self, equation):
        """全ての領域に方程式を追加"""
        equation = self._set_grid_to_equation(equation)
        
        # 基本の境界と内部領域に追加
        self.left_boundary_equations.append(equation)
        self.interior_equations.append(equation)
        self.right_boundary_equations.append(equation)
        
        # 2Dの場合、その他の境界と角にも追加
        if self.is_2d:
            self.bottom_boundary_equations.append(equation)
            self.top_boundary_equations.append(equation)
            for corner in self.corner_equations.values():
                corner.append(equation)

    # 2D固有のメソッド
    def add_bottom_boundary_equation(self, equation):
        """下境界の方程式を追加 (j=0) (2D only)"""
        if not self.is_2d:
            raise ValueError("2D専用のメソッドが1Dグリッドで呼び出されました")
            
        equation = self._set_grid_to_equation(equation)
        self.bottom_boundary_equations.append(equation)
        self.corner_equations['left_bottom'].append(equation)
        self.corner_equations['right_bottom'].append(equation)
    
    def add_top_boundary_equation(self, equation):
        """上境界の方程式を追加 (j=ny-1) (2D only)"""
        if not self.is_2d:
            raise ValueError("2D専用のメソッドが1Dグリッドで呼び出されました")
            
        equation = self._set_grid_to_equation(equation)
        self.top_boundary_equations.append(equation)
        self.corner_equations['left_top'].append(equation)
        self.corner_equations['right_top'].append(equation)

    def add_interior_x_equation(self, equation):
        """内部点のx方向の方程式を追加 (2D only)"""
        if not self.is_2d:
            raise ValueError("2D専用のメソッドが1Dグリッドで呼び出されました")
            
        equation = self._set_grid_to_equation(equation)
        self.interior_equations.append(equation)
        self.bottom_boundary_equations.append(equation)
        self.top_boundary_equations.append(equation)
    
    def add_interior_y_equation(self, equation):
        """内部点のy方向の方程式を追加 (2D only)"""
        if not self.is_2d:
            raise ValueError("2D専用のメソッドが1Dグリッドで呼び出されました")
            
        equation = self._set_grid_to_equation(equation)
        self.interior_equations.append(equation)
        self.left_boundary_equations.append(equation)
        self.right_boundary_equations.append(equation)

    # 位置判定ヘルパーメソッド
    def _get_point_type(self, i, j=None):
        """グリッド点のタイプを判定"""
        if not self.is_2d:
            if i == 0:
                return "left_boundary"
            elif i == self.grid.n_points - 1:
                return "right_boundary"
            else:
                return "interior"
                
        # 2Dの場合
        nx, ny = self.grid.nx_points, self.grid.ny_points
        
        # 角の判定
        if (i == 0 and j == 0):
            return "left_bottom_corner"
        elif (i == nx - 1 and j == 0):
            return "right_bottom_corner"
        elif (i == 0 and j == ny - 1):
            return "left_top_corner"
        elif (i == nx - 1 and j == ny - 1):
            return "right_top_corner"
        
        # 境界の判定
        if i == 0:
            return "left_boundary"
        elif i == nx - 1:
            return "right_boundary"
        elif j == 0:
            return "bottom_boundary"
        elif j == ny - 1:
            return "top_boundary"
        
        # 内部点
        return "interior"

    def _get_equations_for_point(self, i, j=None):
        """グリッド点に適用する方程式のリストを取得"""
        point_type = self._get_point_type(i, j)
        
        if not self.is_2d:
            if point_type == "left_boundary":
                return self.left_boundary_equations
            elif point_type == "right_boundary":
                return self.right_boundary_equations
            else:
                return self.interior_equations
        
        # 2Dの場合
        if point_type == "interior":
            return self.interior_equations
        elif point_type == "left_boundary":
            return self.left_boundary_equations
        elif point_type == "right_boundary":
            return self.right_boundary_equations
        elif point_type == "bottom_boundary":
            return self.bottom_boundary_equations
        elif point_type == "top_boundary":
            return self.top_boundary_equations
        elif point_type == "left_bottom_corner":
            return self.corner_equations['left_bottom']
        elif point_type == "right_bottom_corner":
            return self.corner_equations['right_bottom']
        elif point_type == "left_top_corner":
            return self.corner_equations['left_top']
        elif point_type == "right_top_corner":
            return self.corner_equations['right_top']
            
        return []

    def build_matrix_system(self):
        """行列システムを構築"""
        if self.is_2d:
            return self._build_matrix_system_2d()
        else:
            return self._build_matrix_system_1d()

    def _build_matrix_system_1d(self):
        """1D行列システムの構築"""
        # システムの妥当性を検証
        self.validate_equation_system()
        
        n = self.grid.n_points
        size = 4 * n  # 1Dは4変数 [ψ, ψ', ψ'', ψ''']
        
        data = []
        row_indices = []
        col_indices = []

        for i in range(n):
            # 基本インデックス
            base_idx = i * 4
            
            # この点で適用する方程式リスト
            applicable_equations = self._get_equations_for_point(i)
            
            # 各グリッド点で有効な4つの方程式を追加
            valid_equations = []
            for eq in applicable_equations:
                if eq.is_valid_at(i=i):
                    valid_equations.append(eq)
            
            # 4つの方程式が必要
            if len(valid_equations) < 4:
                raise ValueError(f"点 {i} に十分な方程式がありません。")
            
            # 最初の4つの方程式を使用
            for row_offset, eq in enumerate(valid_equations[:4]):
                # 行インデックス
                row = base_idx + row_offset
                
                # ステンシル係数を取得
                stencil_coeffs = eq.get_stencil_coefficients(i=i)
                
                # 行列に係数を追加
                for offset, coeffs in stencil_coeffs.items():
                    j = i + offset
                    if 0 <= j < n:
                        col_base = j * 4
                        for m, coeff in enumerate(coeffs):
                            if coeff != 0.0:
                                row_indices.append(row)
                                col_indices.append(col_base + m)
                                data.append(float(coeff))

        A = sp.csr_matrix(
            (cp.array(data), (cp.array(row_indices), cp.array(col_indices))), 
            shape=(size, size)
        )

        return A

    def _build_matrix_system_2d(self):
        """2D行列システムの構築"""
        # システムの妥当性を検証
        self.validate_equation_system()
        
        nx, ny = self.grid.nx_points, self.grid.ny_points
        n_unknowns = 7  # 7 unknowns per grid point: [ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy]
        system_size = n_unknowns * nx * ny
        
        data = []
        row_indices = []
        col_indices = []
        
        # 全グリッド点に対して
        for j in range(ny):
            for i in range(nx):
                # グリッド点の基本インデックス
                base_idx = (j * nx + i) * n_unknowns
                
                # この点で適用する方程式リスト
                applicable_equations = self._get_equations_for_point(i, j)
                
                # 各グリッド点で有効な7つの方程式を追加
                valid_equations = []
                for eq in applicable_equations:
                    if eq.is_valid_at(i=i, j=j):
                        valid_equations.append(eq)
                
                # 7つの方程式が必要
                if len(valid_equations) < 7:
                    raise ValueError(f"点 ({i}, {j}) に十分な方程式がありません。")
                
                # 最初の7つの方程式を使用
                for row_offset, eq in enumerate(valid_equations[:7]):
                    # 行インデックス
                    row = base_idx + row_offset
                    
                    # ステンシル係数を取得
                    stencil_coeffs = eq.get_stencil_coefficients(i=i, j=j)
                    
                    # 行列に係数を追加
                    for (di, dj), coeffs in stencil_coeffs.items():
                        ni, nj = i + di, j + dj
                        if 0 <= ni < nx and 0 <= nj < ny:
                            col_base = (nj * nx + ni) * n_unknowns
                            for m, coeff in enumerate(coeffs):
                                if coeff != 0.0:
                                    row_indices.append(row)
                                    col_indices.append(col_base + m)
                                    data.append(float(coeff))
        
        # 疎行列を作成
        A = sp.csr_matrix(
            (cp.array(data), (cp.array(row_indices), cp.array(col_indices))),
            shape=(system_size, system_size)
        )
        
        return A

    def analyze_sparsity(self):
        """行列の疎性を分析"""
        A = self.build_matrix_system()
        
        total_size = A.shape[0]
        nnz = A.nnz
        sparsity = 1.0 - (nnz / (total_size * total_size))
        
        memory_dense_MB = (total_size * total_size * 8) / (1024 * 1024)  # 8 bytes per double
        memory_sparse_MB = (nnz * 12) / (1024 * 1024)  # 8 bytes for value + 4 bytes for indices
        
        return {
            "matrix_size": total_size,
            "non_zeros": nnz,
            "sparsity": sparsity,
            "memory_dense_MB": memory_dense_MB,
            "memory_sparse_MB": memory_sparse_MB
        }