import cupy as cp
from equation.base2d import Equation2D

class Equation1Dto2DConverter:
    """1次元方程式を2次元方程式に変換するファクトリクラス"""
    
    @staticmethod
    def to_x(equation_1d, x_only=False, grid=None):
        """
        1次元方程式をx方向の2次元方程式に変換
        
        Args:
            equation_1d: 1次元方程式クラスのインスタンス
            x_only: True の場合、y方向の成分を無視
            grid: Grid2Dオブジェクト（オプション）
            
        Returns:
            x方向用の2次元方程式インスタンス
        """
        return DirectionalEquation2D(equation_1d, 'x', x_only, grid)
    
    @staticmethod
    def to_y(equation_1d, y_only=False, grid=None):
        """
        1次元方程式をy方向の2次元方程式に変換
        
        Args:
            equation_1d: 1次元方程式クラスのインスタンス
            y_only: True の場合、x方向の成分を無視
            grid: Grid2Dオブジェクト（オプション）
            
        Returns:
            y方向用の2次元方程式インスタンス
        """
        return DirectionalEquation2D(equation_1d, 'y', y_only, grid)
    
    @staticmethod
    def to_xy(equation_1d_x, equation_1d_y=None, grid=None):
        """
        異なる方程式をx方向とy方向に適用
        
        Args:
            equation_1d_x: x方向に適用する1次元方程式
            equation_1d_y: y方向に適用する1次元方程式（指定しない場合はequation_1d_xと同じ）
            grid: Grid2Dオブジェクト（オプション）
            
        Returns:
            両方向に対応する2次元方程式インスタンス
        """
        if equation_1d_y is None:
            equation_1d_y = equation_1d_x
            
        x_eq = DirectionalEquation2D(equation_1d_x, 'x', False, grid)
        y_eq = DirectionalEquation2D(equation_1d_y, 'y', False, grid)
        
        return CombinedDirectionalEquation2D(x_eq, y_eq, grid)


class DirectionalEquation2D(Equation2D):
    """
    1次元方程式を指定方向の2次元方程式に変換するアダプタクラス
    """
    
    def __init__(self, equation_1d, direction='x', direction_only=False, grid=None):
        """
        初期化
        
        Args:
            equation_1d: 1次元方程式のインスタンス
            direction: 'x'または'y'
            direction_only: 特定の方向のみ処理する場合True
            grid: Grid2Dオブジェクト（オプション）
        """
        super().__init__(grid)
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
            
        # 部分方程式にもgridを設定
        if self.grid is not None and hasattr(equation_1d, 'set_grid'):
            # 適切な1Dグリッドを生成して1D方程式に設定
            self._set_1d_grid_to_equation()
    
    def _set_1d_grid_to_equation(self):
        """1次元方程式に対応するグリッドを設定"""
        if self.grid is None:
            return
            
        # 簡易的な1Dグリッドエミュレータを作成
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
            emulated_grid = Grid1DEmulator(
                self.grid.x, 
                self.grid.get_spacing()[0], 
                self.grid.nx_points
            )
        else:  # self.direction == 'y'
            emulated_grid = Grid1DEmulator(
                self.grid.y, 
                self.grid.get_spacing()[1], 
                self.grid.ny_points
            )
            
        # 1次元方程式にエミュレートされたグリッドを設定
        if hasattr(self.equation_1d, 'set_grid'):
            self.equation_1d.set_grid(emulated_grid)
    
    def set_grid(self, grid):
        """
        グリッドを設定
        
        Args:
            grid: Grid2Dオブジェクト
            
        Returns:
            self: メソッドチェーン用
        """
        super().set_grid(grid)
        
        # 部分方程式にもグリッドを設定
        self._set_1d_grid_to_equation()
        
        return self
    
    def get_stencil_coefficients(self, grid=None, i=None, j=None):
        """
        ステンシル係数を取得
        
        Args:
            grid: Grid2D（Noneの場合はself.gridを使用）
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
            
        Returns:
            2D方程式用のステンシル係数
        """
        # gridパラメータの処理
        using_grid = grid
        if using_grid is None:
            if self.grid is None:
                raise ValueError("gridが設定されていません。set_grid()で設定するか、引数で指定してください。")
            using_grid = self.grid
            
        if i is None or j is None:
            raise ValueError("グリッド点のインデックスiとjを指定する必要があります。")
        
        # 方向に応じた1次元のインデックスと境界判定
        if self.direction == 'x':
            i_1d = i  # x方向のインデックス
            is_boundary = i == 0 or i == using_grid.nx_points - 1
        else:  # self.direction == 'y'
            i_1d = j  # y方向のインデックス
            is_boundary = j == 0 or j == using_grid.ny_points - 1
        
        # 境界点で特定方向のみの場合は処理を飛ばす
        if self.direction_only and is_boundary:
            return {}
        
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
            h = using_grid.get_spacing()[0]  # hx
            emulated_grid = Grid1DEmulator(using_grid.x, h, using_grid.nx_points)
        else:  # self.direction == 'y'
            h = using_grid.get_spacing()[1]  # hy
            emulated_grid = Grid1DEmulator(using_grid.y, h, using_grid.ny_points)
        
        # 1次元方程式からステンシル係数を取得
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
    
    def get_rhs(self, grid=None, i=None, j=None):
        """
        右辺値を取得
        
        Args:
            grid: Grid2D（Noneの場合はself.gridを使用）
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
            
        Returns:
            右辺値
        """
        # gridパラメータの処理
        using_grid = grid
        if using_grid is None:
            if self.grid is None:
                raise ValueError("gridが設定されていません。set_grid()で設定するか、引数で指定してください。")
            using_grid = self.grid
            
        if i is None or j is None:
            raise ValueError("グリッド点のインデックスiとjを指定する必要があります。")
        
        # 方向に応じたインデックスと座標値を選択
        if self.direction == 'x':
            # x方向の場合は1次元方程式にiを渡す
            i_1d = i
            is_boundary = i == 0 or i == using_grid.nx_points - 1
        else:  # self.direction == 'y'
            # y方向の場合は1次元方程式にjを渡す
            i_1d = j
            is_boundary = j == 0 or j == using_grid.ny_points - 1
        
        # 境界点で特定方向のみの場合は右辺値を0とする
        if self.direction_only and is_boundary:
            return 0.0
        
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
            point = using_grid.get_point(i, j)[0]  # x座標のみ
            h = using_grid.get_spacing()[0]
            n = using_grid.nx_points
            emulated_grid = Grid1DEmulator([point], h, n)
        else:  # self.direction == 'y'
            point = using_grid.get_point(i, j)[1]  # y座標のみ
            h = using_grid.get_spacing()[1]
            n = using_grid.ny_points
            emulated_grid = Grid1DEmulator([point], h, n)
        
        # 1次元方程式から右辺値を取得
        return self.equation_1d.get_rhs(emulated_grid, 0)  # インデックスは常に0
    
    def is_valid_at(self, grid=None, i=None, j=None):
        """
        方程式が指定点で有効かどうかを判定
        
        Args:
            grid: Grid2D（Noneの場合はself.gridを使用）
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
            
        Returns:
            有効性を示すブール値
        """
        # gridパラメータの処理
        using_grid = grid
        if using_grid is None:
            if self.grid is None:
                raise ValueError("gridが設定されていません。set_grid()で設定するか、引数で指定してください。")
            using_grid = self.grid
            
        if i is None or j is None:
            raise ValueError("グリッド点のインデックスiとjを指定する必要があります。")
        
        # 1次元方程式のis_valid_atを利用する
        # 方向を考慮した1Dグリッドのエミュレータを作成
        class Grid1DEmulator:
            def __init__(self, n_points):
                self.n_points = n_points
        
        if self.direction == 'x':
            # x方向の境界判定
            if self.direction_only:
                # x方向のみの場合、内部点および左右境界のみで有効（上下境界では無効）
                if j == 0 or j == using_grid.ny_points - 1:
                    return False
            
            # 1次元方程式のvalid判定を使用
            emulated_grid = Grid1DEmulator(using_grid.nx_points)
            return self.equation_1d.is_valid_at(emulated_grid, i)
            
        else:  # self.direction == 'y'
            # y方向の境界判定
            if self.direction_only:
                # y方向のみの場合、内部点および上下境界のみで有効（左右境界では無効）
                if i == 0 or i == using_grid.nx_points - 1:
                    return False
            
            # 1次元方程式のvalid判定を使用
            emulated_grid = Grid1DEmulator(using_grid.ny_points)
            return self.equation_1d.is_valid_at(emulated_grid, j)


class CombinedDirectionalEquation2D(Equation2D):
    """
    x方向とy方向の2次元方程式を組み合わせたクラス
    """
    
    def __init__(self, x_direction_eq, y_direction_eq, grid=None):
        """
        初期化
        
        Args:
            x_direction_eq: x方向の2次元方程式
            y_direction_eq: y方向の2次元方程式
            grid: Grid2Dオブジェクト（オプション）
        """
        # gridが指定されていなければ、部分方程式のものを使用
        if grid is None:
            if hasattr(x_direction_eq, 'grid') and x_direction_eq.grid is not None:
                grid = x_direction_eq.grid
            elif hasattr(y_direction_eq, 'grid') and y_direction_eq.grid is not None:
                grid = y_direction_eq.grid
                
        super().__init__(grid)
        self.x_eq = x_direction_eq
        self.y_eq = y_direction_eq
        
        # 部分方程式にもgridを設定
        if self.grid is not None:
            if hasattr(x_direction_eq, 'set_grid'):
                x_direction_eq.set_grid(self.grid)
            if hasattr(y_direction_eq, 'set_grid'):
                y_direction_eq.set_grid(self.grid)
    
    def set_grid(self, grid):
        """
        グリッドを設定
        
        Args:
            grid: Grid2Dオブジェクト
            
        Returns:
            self: メソッドチェーン用
        """
        super().set_grid(grid)
        
        # 部分方程式にもグリッドを設定
        if hasattr(self.x_eq, 'set_grid'):
            self.x_eq.set_grid(grid)
        if hasattr(self.y_eq, 'set_grid'):
            self.y_eq.set_grid(grid)
            
        return self
    
    def get_stencil_coefficients(self, grid=None, i=None, j=None):
        """
        ステンシル係数を取得（両方向の係数を結合）
        
        Args:
            grid: Grid2D（Noneの場合はself.gridを使用）
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
            
        Returns:
            2D方程式用のステンシル係数
        """
        # gridパラメータの処理
        using_grid = grid
        if using_grid is None:
            if self.grid is None:
                raise ValueError("gridが設定されていません。set_grid()で設定するか、引数で指定してください。")
            using_grid = self.grid
        
        # 両方の方程式から係数を取得
        x_coeffs = self.x_eq.get_stencil_coefficients(using_grid, i, j)
        y_coeffs = self.y_eq.get_stencil_coefficients(using_grid, i, j)
        
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
    
    def get_rhs(self, grid=None, i=None, j=None):
        """
        右辺値を取得（両方向の右辺値を加算）
        
        Args:
            grid: Grid2D（Noneの場合はself.gridを使用）
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
            
        Returns:
            右辺値
        """
        # gridパラメータの処理
        using_grid = grid
        if using_grid is None:
            if self.grid is None:
                raise ValueError("gridが設定されていません。set_grid()で設定するか、引数で指定してください。")
            using_grid = self.grid
        
        # 両方の方程式から右辺値を取得して加算
        return self.x_eq.get_rhs(using_grid, i, j) + self.y_eq.get_rhs(using_grid, i, j)
    
    def is_valid_at(self, grid=None, i=None, j=None):
        """
        方程式が指定点で有効かどうかを判定（両方向とも有効な場合のみ有効）
        
        Args:
            grid: Grid2D（Noneの場合はself.gridを使用）
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
            
        Returns:
            有効性を示すブール値
        """
        # gridパラメータの処理
        using_grid = grid
        if using_grid is None:
            if self.grid is None:
                raise ValueError("gridが設定されていません。set_grid()で設定するか、引数で指定してください。")
            using_grid = self.grid
        
        # 両方の方程式が有効な場合のみ有効
        return self.x_eq.is_valid_at(using_grid, i, j) and self.y_eq.is_valid_at(using_grid, i, j)