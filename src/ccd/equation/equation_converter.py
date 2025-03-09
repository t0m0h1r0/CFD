import cupy as cp
from equation.base2d import Equation2D

class Equation1Dto2DConverter:
    """1次元方程式を2次元方程式に変換するファクトリクラス"""
    
    @staticmethod
    def to_x(equation_1d, x_only=False):
        """
        1次元方程式をx方向の2次元方程式に変換
        
        Args:
            equation_1d: 1次元方程式クラスのインスタンス
            x_only: True の場合、y方向の成分を無視
            
        Returns:
            x方向用の2次元方程式インスタンス
        """
        return DirectionalEquation2D(equation_1d, 'x', x_only)
    
    @staticmethod
    def to_y(equation_1d, y_only=False):
        """
        1次元方程式をy方向の2次元方程式に変換
        
        Args:
            equation_1d: 1次元方程式クラスのインスタンス
            y_only: True の場合、x方向の成分を無視
            
        Returns:
            y方向用の2次元方程式インスタンス
        """
        return DirectionalEquation2D(equation_1d, 'y', y_only)
    
    @staticmethod
    def to_xy(equation_1d_x, equation_1d_y=None):
        """
        異なる方程式をx方向とy方向に適用
        
        Args:
            equation_1d_x: x方向に適用する1次元方程式
            equation_1d_y: y方向に適用する1次元方程式（指定しない場合はequation_1d_xと同じ）
            
        Returns:
            両方向に対応する2次元方程式インスタンス
        """
        if equation_1d_y is None:
            equation_1d_y = equation_1d_x
            
        x_eq = DirectionalEquation2D(equation_1d_x, 'x')
        y_eq = DirectionalEquation2D(equation_1d_y, 'y')
        
        return CombinedDirectionalEquation2D(x_eq, y_eq)


class DirectionalEquation2D(Equation2D):
    """
    1次元方程式を指定方向の2次元方程式に変換するアダプタクラス
    """
    
    def __init__(self, equation_1d, direction='x', direction_only=False):
        """
        初期化
        
        Args:
            equation_1d: 1次元方程式のインスタンス
            direction: 'x'または'y'
            direction_only: 特定の方向のみ処理する場合True
        """
        self.equation_1d = equation_1d
        self.direction = direction
        self.direction_only = direction_only
        
        # 方向に応じたインデックスマッピング
        # 2次元の未知数順序: [ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy]
        if direction == 'x':
            # 1次元の未知数[ψ, ψ', ψ'', ψ''']を
            # 2次元の[ψ, ψ_x, ψ_xx, ψ_xxx]にマッピング
            self.index_map = {0: 0, 1: 1, 2: 2, 3: 3}
        else:  # direction == 'y'
            # 1次元の未知数[ψ, ψ', ψ'', ψ''']を
            # 2次元の[ψ, ψ_y, ψ_yy, ψ_yyy]にマッピング
            self.index_map = {0: 0, 1: 4, 2: 5, 3: 6}
    
    def get_stencil_coefficients(self, grid, i, j):
        """
        ステンシル係数を取得
        
        Args:
            grid: Grid2D
            i, j: 格子点インデックス
            
        Returns:
            2D方程式用のステンシル係数
        """
        # 1次元方程式から適切な間隔を取得
        if self.direction == 'x':
            h = grid.get_spacing()[0]  # hx
            i_1d = i  # x方向のインデックス
            is_boundary = i == 0 or i == grid.nx_points - 1
        else:  # self.direction == 'y'
            h = grid.get_spacing()[1]  # hy
            i_1d = j  # y方向のインデックス
            is_boundary = j == 0 or j == grid.ny_points - 1
        
        # 1D Grid オブジェクトをエミュレート
        class Grid1DEmulator:
            def __init__(self, points, spacing, n_points):
                self.points = points
                self.h = spacing
                self.n_points = n_points
            
            def get_point(self, idx):
                return self.points[idx]
            
            def get_points(self):
                return self.points
            
            def get_spacing(self):
                return self.h
        
        # 方向に応じた1Dグリッドを作成
        if self.direction == 'x':
            emulated_grid = Grid1DEmulator(grid.x, h, grid.nx_points)
        else:  # self.direction == 'y'
            emulated_grid = Grid1DEmulator(grid.y, h, grid.ny_points)
        
        # 1次元方程式からステンシル係数を取得
        if self.direction_only and is_boundary:
            # 境界点で特定方向のみの場合、有効性チェックをスキップ
            # 実際の有効性は後で is_valid_at で判定
            coeffs_1d = {}
        else:
            coeffs_1d = self.equation_1d.get_stencil_coefficients(emulated_grid, i_1d)
        
        # 1次元の係数を2次元に変換
        coeffs_2d = {}
        for offset_1d, coeff_array_1d in coeffs_1d.items():
            # 方向に応じてオフセットを変換
            if self.direction == 'x':
                offset_2d = (offset_1d, 0)
            else:  # self.direction == 'y'
                offset_2d = (0, offset_1d)
            
            # 係数配列を変換（7要素の配列に拡張）
            coeff_array_2d = cp.zeros(7)
            for idx_1d, idx_2d in self.index_map.items():
                if idx_1d < len(coeff_array_1d):
                    coeff_array_2d[idx_2d] = coeff_array_1d[idx_1d]
            
            coeffs_2d[offset_2d] = coeff_array_2d
        
        return coeffs_2d
    
    def get_rhs(self, grid, i, j):
        """
        右辺値を取得
        
        Args:
            grid: Grid2D
            i, j: 格子点インデックス
            
        Returns:
            右辺値
        """
        # 方向に応じたインデックスと座標値を選択
        if self.direction == 'x':
            # x方向の場合は1次元方程式にiを渡す
            point = grid.get_point(i, j)[0]  # x座標のみ
            i_1d = i
            is_boundary = i == 0 or i == grid.nx_points - 1
        else:  # self.direction == 'y'
            # y方向の場合は1次元方程式にjを渡す
            point = grid.get_point(i, j)[1]  # y座標のみ
            i_1d = j
            is_boundary = j == 0 or j == grid.ny_points - 1
        
        if self.direction_only and is_boundary:
            # 境界点で特定方向のみの場合、右辺値を0とする
            return 0.0
        
        # 1D Grid オブジェクトをエミュレート
        class Grid1DEmulator:
            def __init__(self, point, spacing, n_points):
                self.point_value = point
                self.h = spacing
                self.n_points = n_points
            
            def get_point(self, idx):
                return self.point_value
            
            def get_spacing(self):
                return self.h
        
        # 方向に応じた間隔を選択
        h = grid.get_spacing()[0] if self.direction == 'x' else grid.get_spacing()[1]
        n = grid.nx_points if self.direction == 'x' else grid.ny_points
        
        # エミュレートされた1Dグリッドを作成
        emulated_grid = Grid1DEmulator(point, h, n)
        
        # 1次元方程式から右辺値を取得
        return self.equation_1d.get_rhs(emulated_grid, i_1d)
    
    def is_valid_at(self, grid, i, j):
        """
        方程式が指定点で有効かどうかを判定
        
        Args:
            grid: Grid2D
            i, j: 格子点インデックス
            
        Returns:
            有効性を示すブール値
        """
        # 1次元方程式のis_valid_atを利用する
        # 方向を考慮した1Dグリッドのエミュレータを作成
        class Grid1DEmulator:
            def __init__(self, n_points):
                self.n_points = n_points
        
        if self.direction == 'x':
            # x方向の境界判定
            if self.direction_only:
                # x方向のみの場合、内部点および左右境界のみで有効（上下境界では無効）
                if j == 0 or j == grid.ny_points - 1:
                    return False
            
            # 1次元方程式のvalid判定を使用
            emulated_grid = Grid1DEmulator(grid.nx_points)
            return self.equation_1d.is_valid_at(emulated_grid, i)
            
        else:  # self.direction == 'y'
            # y方向の境界判定
            if self.direction_only:
                # y方向のみの場合、内部点および上下境界のみで有効（左右境界では無効）
                if i == 0 or i == grid.nx_points - 1:
                    return False
            
            # 1次元方程式のvalid判定を使用
            emulated_grid = Grid1DEmulator(grid.ny_points)
            return self.equation_1d.is_valid_at(emulated_grid, j)


class CombinedDirectionalEquation2D(Equation2D):
    """
    x方向とy方向の2次元方程式を組み合わせたクラス
    """
    
    def __init__(self, x_direction_eq, y_direction_eq):
        """
        初期化
        
        Args:
            x_direction_eq: x方向の2次元方程式
            y_direction_eq: y方向の2次元方程式
        """
        self.x_eq = x_direction_eq
        self.y_eq = y_direction_eq
    
    def get_stencil_coefficients(self, grid, i, j):
        """
        ステンシル係数を取得（両方向の係数を結合）
        
        Args:
            grid: Grid2D
            i, j: 格子点インデックス
            
        Returns:
            2D方程式用のステンシル係数
        """
        # 両方の方程式から係数を取得
        x_coeffs = self.x_eq.get_stencil_coefficients(grid, i, j)
        y_coeffs = self.y_eq.get_stencil_coefficients(grid, i, j)
        
        # 係数を結合（オフセットが重複する場合は加算）
        combined_coeffs = {}
        
        # まずx方向の係数を追加
        for offset, coeff in x_coeffs.items():
            combined_coeffs[offset] = coeff.copy()
        
        # y方向の係数を追加・結合
        for offset, coeff in y_coeffs.items():
            if offset in combined_coeffs:
                combined_coeffs[offset] += coeff
            else:
                combined_coeffs[offset] = coeff
        
        return combined_coeffs
    
    def get_rhs(self, grid, i, j):
        """
        右辺値を取得（両方向の右辺値を加算）
        
        Args:
            grid: Grid2D
            i, j: 格子点インデックス
            
        Returns:
            右辺値
        """
        # 両方の方程式から右辺値を取得して加算
        return self.x_eq.get_rhs(grid, i, j) + self.y_eq.get_rhs(grid, i, j)
    
    def is_valid_at(self, grid, i, j):
        """
        方程式が指定点で有効かどうかを判定（両方向とも有効な場合のみ有効）
        
        Args:
            grid: Grid2D
            i, j: 格子点インデックス
            
        Returns:
            有効性を示すブール値
        """
        # 両方の方程式が有効な場合のみ有効
        return self.x_eq.is_valid_at(grid, i, j) and self.y_eq.is_valid_at(grid, i, j)