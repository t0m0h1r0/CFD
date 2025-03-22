import numpy as cp
from equation.base2d import Equation2D
from equation.base3d import Equation3D

class Equation1Dto2DConverter:
    """1次元方程式を2次元方程式に変換するファクトリクラス"""
    
    @staticmethod
    def to_x(equation_1d, x_only=False, grid=None):
        """
        1次元方程式をx方向の2次元方程式に変換
        
        Args:
            equation_1d: 1次元方程式クラスのインスタンス
            x_only: True の場合、y方向の成分を無視
            grid: Grid2Dオブジェクト
            
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
            grid: Grid2Dオブジェクト
            
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
            grid: Grid2Dオブジェクト
            
        Returns:
            両方向に対応する2次元方程式インスタンス
        """
        if equation_1d_y is None:
            equation_1d_y = equation_1d_x
            
        x_eq = DirectionalEquation2D(equation_1d_x, 'x', False, grid)
        y_eq = DirectionalEquation2D(equation_1d_y, 'y', False, grid)
        
        return CombinedDirectionalEquation2D(x_eq, y_eq, grid)


class Equation2Dto3DConverter:
    """2次元方程式を3次元方程式に変換するファクトリクラス"""
    
    @staticmethod
    def to_xy(equation_2d, xy_only=False, grid=None):
        """
        2次元方程式をxy平面方向の3次元方程式に変換
        
        Args:
            equation_2d: 2次元方程式クラスのインスタンス
            xy_only: True の場合、z方向の成分を無視
            grid: Grid3Dオブジェクト
            
        Returns:
            xy平面用の3次元方程式インスタンス
        """
        return DirectionalEquation3D(equation_2d, 'xy', xy_only, grid)
    
    @staticmethod
    def to_xz(equation_2d, xz_only=False, grid=None):
        """
        2次元方程式をxz平面方向の3次元方程式に変換
        
        Args:
            equation_2d: 2次元方程式クラスのインスタンス
            xz_only: True の場合、y方向の成分を無視
            grid: Grid3Dオブジェクト
            
        Returns:
            xz平面用の3次元方程式インスタンス
        """
        return DirectionalEquation3D(equation_2d, 'xz', xz_only, grid)
    
    @staticmethod
    def to_yz(equation_2d, yz_only=False, grid=None):
        """
        2次元方程式をyz平面方向の3次元方程式に変換
        
        Args:
            equation_2d: 2次元方程式クラスのインスタンス
            yz_only: True の場合、x方向の成分を無視
            grid: Grid3Dオブジェクト
            
        Returns:
            yz平面用の3次元方程式インスタンス
        """
        return DirectionalEquation3D(equation_2d, 'yz', yz_only, grid)
    
    @staticmethod
    def to_xyz(equation_2d_xy, equation_2d_xz=None, equation_2d_yz=None, grid=None):
        """
        異なる方程式を各平面方向に適用
        
        Args:
            equation_2d_xy: xy平面に適用する2次元方程式
            equation_2d_xz: xz平面に適用する2次元方程式（指定しない場合はequation_2d_xyと同じ）
            equation_2d_yz: yz平面に適用する2次元方程式（指定しない場合はequation_2d_xyと同じ）
            grid: Grid3Dオブジェクト
            
        Returns:
            全方向に対応する3次元方程式インスタンス
        """
        if equation_2d_xz is None:
            equation_2d_xz = equation_2d_xy
        if equation_2d_yz is None:
            equation_2d_yz = equation_2d_xy
            
        xy_eq = DirectionalEquation3D(equation_2d_xy, 'xy', False, grid)
        xz_eq = DirectionalEquation3D(equation_2d_xz, 'xz', False, grid)
        yz_eq = DirectionalEquation3D(equation_2d_yz, 'yz', False, grid)
        
        return CombinedDirectionalEquation3D(xy_eq, xz_eq, yz_eq, grid)

    @staticmethod
    def to_x(equation_2d, grid=None):
        """
        2次元方程式のx方向成分のみを取り出して3次元方程式に変換
        
        Args:
            equation_2d: 2次元方程式クラスのインスタンス
            grid: Grid3Dオブジェクト
            
        Returns:
            x方向に特化した3次元方程式インスタンス
        """
        # x方向にのみ適用する特殊な3次元方程式
        return DirectionalEquation3D(equation_2d, 'x', True, grid)
    
    @staticmethod
    def to_y(equation_2d, grid=None):
        """
        2次元方程式のy方向成分のみを取り出して3次元方程式に変換
        
        Args:
            equation_2d: 2次元方程式クラスのインスタンス
            grid: Grid3Dオブジェクト
            
        Returns:
            y方向に特化した3次元方程式インスタンス
        """
        # y方向にのみ適用する特殊な3次元方程式
        return DirectionalEquation3D(equation_2d, 'y', True, grid)
    
    @staticmethod
    def to_z(equation_2d, grid=None):
        """
        2次元方程式のz方向成分のみを取り出して3次元方程式に変換
        
        Args:
            equation_2d: 2次元方程式クラスのインスタンス
            grid: Grid3Dオブジェクト
            
        Returns:
            z方向に特化した3次元方程式インスタンス
        """
        # z方向にのみ適用する特殊な3次元方程式
        return DirectionalEquation3D(equation_2d, 'z', True, grid)


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
            grid: Grid2Dオブジェクト
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
    
    def get_stencil_coefficients(self, i=None, j=None):
        """
        ステンシル係数を取得
        
        Args:
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
            
        Returns:
            2D方程式用のステンシル係数
        """
        if self.grid is None:
            raise ValueError("gridが設定されていません。set_grid()で設定してください。")
            
        if i is None or j is None:
            raise ValueError("グリッド点のインデックスiとjを指定する必要があります。")
        
        # 方向に応じた1次元のインデックスと境界判定
        if self.direction == 'x':
            i_1d = i  # x方向のインデックス
            is_boundary = i == 0 or i == self.grid.nx_points - 1
        else:  # self.direction == 'y'
            i_1d = j  # y方向のインデックス
            is_boundary = j == 0 or j == self.grid.ny_points - 1
        
        # 境界点で特定方向のみの場合は処理を飛ばす
        if self.direction_only and is_boundary:
            return {}
        
        # 1次元方程式からステンシル係数を取得
        coeffs_1d = self.equation_1d.get_stencil_coefficients(i_1d)
        
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
    
    def is_valid_at(self, i=None, j=None):
        """
        方程式が指定点で有効かどうかを判定
        
        Args:
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
            
        Returns:
            有効性を示すブール値
        """
        if self.grid is None:
            raise ValueError("gridが設定されていません。set_grid()で設定してください。")
            
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
                if j == 0 or j == self.grid.ny_points - 1:
                    return False
            
            # 1次元方程式のvalid判定を使用
            emulator = Grid1DEmulator(self.grid.nx_points)
            return self.equation_1d.is_valid_at(i)
            
        else:  # self.direction == 'y'
            # y方向の境界判定
            if self.direction_only:
                # y方向のみの場合、内部点および上下境界のみで有効（左右境界では無効）
                if i == 0 or i == self.grid.nx_points - 1:
                    return False
            
            # 1次元方程式のvalid判定を使用
            emulator = Grid1DEmulator(self.grid.ny_points)
            return self.equation_1d.is_valid_at(j)


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
            grid: Grid2Dオブジェクト
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
    
    def get_stencil_coefficients(self, i=None, j=None):
        """
        ステンシル係数を取得（両方向の係数を結合）
        
        Args:
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
            
        Returns:
            2D方程式用のステンシル係数
        """
        if self.grid is None:
            raise ValueError("gridが設定されていません。set_grid()で設定してください。")
        
        # 両方の方程式から係数を取得
        x_coeffs = self.x_eq.get_stencil_coefficients(i, j)
        y_coeffs = self.y_eq.get_stencil_coefficients(i, j)
        
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
    
    def is_valid_at(self, i=None, j=None):
        """
        方程式が指定点で有効かどうかを判定（両方向とも有効な場合のみ有効）
        
        Args:
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
            
        Returns:
            有効性を示すブール値
        """
        if self.grid is None:
            raise ValueError("gridが設定されていません。set_grid()で設定してください。")
        
        # 両方の方程式が有効な場合のみ有効
        return self.x_eq.is_valid_at(i, j) and self.y_eq.is_valid_at(i, j)


class DirectionalEquation3D(Equation3D):
    """
    2次元方程式を指定平面の3次元方程式に変換するアダプタクラス
    """
    
    def __init__(self, equation_2d, direction='xy', direction_only=False, grid=None):
        """
        初期化
        
        Args:
            equation_2d: 2次元方程式のインスタンス
            direction: 'xy'、'xz'、'yz'、'x'、'y'、'z'
            direction_only: 特定の平面/方向のみ処理する場合True
            grid: Grid3Dオブジェクト
        """
        super().__init__(grid)
        self.equation_2d = equation_2d
        self.direction = direction
        self.direction_only = direction_only
        
        # 方向に応じたインデックスマッピング
        # 3次元の未知数順序: [ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy, ψ_z, ψ_zz, ψ_zzz]
        if direction == 'xy':
            # 2次元の未知数[ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy]を
            # 3次元の[ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy, 0, 0, 0]にマッピング
            self.index_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
        elif direction == 'xz':
            # 2次元の未知数[ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy]を
            # 3次元の[ψ, ψ_x, ψ_xx, ψ_xxx, 0, 0, 0, ψ_z, ψ_zz, ψ_zzz]にマッピング （yをzに変更）
            self.index_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 7, 5: 8, 6: 9}
        elif direction == 'yz':
            # 2次元の未知数[ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy]を
            # 3次元の[ψ, 0, 0, 0, ψ_y, ψ_yy, ψ_yyy, ψ_z, ψ_zz, ψ_zzz]にマッピング （xをzに変更）
            self.index_map = {0: 0, 1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9}
        elif direction == 'x':
            # x方向のみの特殊ケース
            self.index_map = {0: 0, 1: 1, 2: 2, 3: 3}
        elif direction == 'y':
            # y方向のみの特殊ケース
            self.index_map = {0: 0, 1: 4, 2: 5, 3: 6}
        elif direction == 'z':
            # z方向のみの特殊ケース
            self.index_map = {0: 0, 1: 7, 2: 8, 3: 9}
            
        # 部分方程式にもgridを設定
        if self.grid is not None and hasattr(equation_2d, 'set_grid'):
            # 適切な2Dグリッドを生成して2D方程式に設定
            self._set_2d_grid_to_equation()
    
    def _set_2d_grid_to_equation(self):
        """2次元方程式に対応するグリッドを設定"""
        if self.grid is None:
            return
            
        # 簡易的な2Dグリッドエミュレータを作成
        class Grid2DEmulator:
            def __init__(self, x_points, y_points, x_spacing, y_spacing, nx_points, ny_points):
                self.x = x_points
                self.y = y_points
                self.hx = x_spacing
                self.hy = y_spacing
                self.nx_points = nx_points
                self.ny_points = ny_points
                self.is_2d = True
            
            def get_point(self, i, j):
                return self.x[i], self.y[j]
            
            def get_spacing(self):
                return self.hx, self.hy
        
        # 方向に応じた2Dグリッドを作成
        if self.direction in ['xy', 'x', 'y']:
            emulated_grid = Grid2DEmulator(
                self.grid.x, self.grid.y,
                self.grid.get_spacing()[0], self.grid.get_spacing()[1],
                self.grid.nx_points, self.grid.ny_points
            )
        elif self.direction in ['xz', 'x', 'z']:
            emulated_grid = Grid2DEmulator(
                self.grid.x, self.grid.z,
                self.grid.get_spacing()[0], self.grid.get_spacing()[2],
                self.grid.nx_points, self.grid.nz_points
            )
        else:  # direction in ['yz', 'y', 'z']
            emulated_grid = Grid2DEmulator(
                self.grid.y, self.grid.z,
                self.grid.get_spacing()[1], self.grid.get_spacing()[2],
                self.grid.ny_points, self.grid.nz_points
            )
            
        # 2次元方程式にエミュレートされたグリッドを設定
        if hasattr(self.equation_2d, 'set_grid'):
            self.equation_2d.set_grid(emulated_grid)
    
    def set_grid(self, grid):
        """
        グリッドを設定
        
        Args:
            grid: Grid3Dオブジェクト
            
        Returns:
            self: メソッドチェーン用
        """
        super().set_grid(grid)
        
        # 部分方程式にもグリッドを設定
        self._set_2d_grid_to_equation()
        
        return self
    
    def get_stencil_coefficients(self, i=None, j=None, k=None):
        """
        ステンシル係数を取得
        
        Args:
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
            k: z方向のグリッド点インデックス
            
        Returns:
            3D方程式用のステンシル係数
        """
        if self.grid is None:
            raise ValueError("gridが設定されていません。set_grid()で設定してください。")
            
        if i is None or j is None or k is None:
            raise ValueError("グリッド点のインデックスi, j, kを指定する必要があります。")
        
        # 方向に応じた2次元のインデックスと境界判定
        if self.direction == 'xy':
            i_2d, j_2d = i, j
            is_boundary = k == 0 or k == self.grid.nz_points - 1
        elif self.direction == 'xz':
            i_2d, j_2d = i, k
            is_boundary = j == 0 or j == self.grid.ny_points - 1
        elif self.direction == 'yz':
            i_2d, j_2d = j, k
            is_boundary = i == 0 or i == self.grid.nx_points - 1
        elif self.direction == 'x':
            i_2d, j_2d = i, 0  # j座標は使用しない
            is_boundary = True  # 特定方向のみ
        elif self.direction == 'y':
            i_2d, j_2d = j, 0  # i座標は使用しない
            is_boundary = True  # 特定方向のみ
        else:  # self.direction == 'z'
            i_2d, j_2d = k, 0  # k座標は使用しない
            is_boundary = True  # 特定方向のみ
        
        # 境界点で特定平面のみの場合は処理を飛ばす
        if self.direction_only and is_boundary and self.direction not in ['x', 'y', 'z']:
            return {}
        
        # 2次元方程式からステンシル係数を取得
        coeffs_2d = self.equation_2d.get_stencil_coefficients(i_2d, j_2d)
        
        # 2次元の係数を3次元に変換
        coeffs_3d = {}
        for (offset_i, offset_j), coeff_array_2d in coeffs_2d.items():
            # 方向に応じてオフセットを変換
            if self.direction == 'xy':
                offset_3d = (offset_i, offset_j, 0)
            elif self.direction == 'xz':
                offset_3d = (offset_i, 0, offset_j)
            elif self.direction == 'yz':
                offset_3d = (0, offset_i, offset_j)
            elif self.direction == 'x':
                offset_3d = (offset_i, 0, 0)
            elif self.direction == 'y':
                offset_3d = (0, offset_i, 0)
            else:  # self.direction == 'z'
                offset_3d = (0, 0, offset_i)
            
            # 係数配列を変換（10要素の配列に拡張）
            coeff_array_3d = cp.zeros(10)
            for idx_2d, idx_3d in self.index_map.items():
                if idx_2d < len(coeff_array_2d):
                    coeff_array_3d[idx_3d] = coeff_array_2d[idx_2d]
            
            coeffs_3d[offset_3d] = coeff_array_3d
        
        return coeffs_3d
    
    def is_valid_at(self, i=None, j=None, k=None):
        """
        方程式が指定点で有効かどうかを判定
        
        Args:
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
            k: z方向のグリッド点インデックス
            
        Returns:
            有効性を示すブール値
        """
        if self.grid is None:
            raise ValueError("gridが設定されていません。set_grid()で設定してください。")
            
        if i is None or j is None or k is None:
            raise ValueError("グリッド点のインデックスi, j, kを指定する必要があります。")
        
        # 2次元方程式のis_valid_atを利用する
        # 方向を考慮した2Dグリッドのエミュレータを作成
        class Grid2DEmulator:
            def __init__(self, nx_points, ny_points):
                self.nx_points = nx_points
                self.ny_points = ny_points
                self.is_2d = True
        
        if self.direction == 'xy':
            # xy平面の境界判定
            if self.direction_only:
                # xy平面のみの場合、内部点およびxy平面境界のみで有効（z方向境界では無効）
                if k == 0 or k == self.grid.nz_points - 1:
                    return False
            
            # 2次元方程式のvalid判定を使用
            emulator = Grid2DEmulator(self.grid.nx_points, self.grid.ny_points)
            return self.equation_2d.is_valid_at(i, j)
            
        elif self.direction == 'xz':
            # xz平面の境界判定
            if self.direction_only:
                # xz平面のみの場合、内部点およびxz平面境界のみで有効（y方向境界では無効）
                if j == 0 or j == self.grid.ny_points - 1:
                    return False
            
            # 2次元方程式のvalid判定を使用
            emulator = Grid2DEmulator(self.grid.nx_points, self.grid.nz_points)
            return self.equation_2d.is_valid_at(i, k)
            
        elif self.direction == 'yz':
            # yz平面の境界判定
            if self.direction_only:
                # yz平面のみの場合、内部点およびyz平面境界のみで有効（x方向境界では無効）
                if i == 0 or i == self.grid.nx_points - 1:
                    return False
            
            # 2次元方程式のvalid判定を使用
            emulator = Grid2DEmulator(self.grid.ny_points, self.grid.nz_points)
            return self.equation_2d.is_valid_at(j, k)
            
        elif self.direction == 'x':
            # x方向のみ: 常に有効
            return True
            
        elif self.direction == 'y':
            # y方向のみ: 常に有効
            return True
            
        else:  # self.direction == 'z'
            # z方向のみ: 常に有効
            return True


class CombinedDirectionalEquation3D(Equation3D):
    """
    xy, xz, yz平面の3次元方程式を組み合わせたクラス
    """
    
    def __init__(self, xy_direction_eq, xz_direction_eq, yz_direction_eq, grid=None):
        """
        初期化
        
        Args:
            xy_direction_eq: xy平面の3次元方程式
            xz_direction_eq: xz平面の3次元方程式
            yz_direction_eq: yz平面の3次元方程式
            grid: Grid3Dオブジェクト
        """
        # gridが指定されていなければ、部分方程式のものを使用
        if grid is None:
            for eq in [xy_direction_eq, xz_direction_eq, yz_direction_eq]:
                if hasattr(eq, 'grid') and eq.grid is not None:
                    grid = eq.grid
                    break
                
        super().__init__(grid)
        self.xy_eq = xy_direction_eq
        self.xz_eq = xz_direction_eq
        self.yz_eq = yz_direction_eq
        
        # 部分方程式にもgridを設定
        if self.grid is not None:
            for eq in [xy_direction_eq, xz_direction_eq, yz_direction_eq]:
                if hasattr(eq, 'set_grid'):
                    eq.set_grid(self.grid)
    
    def set_grid(self, grid):
        """
        グリッドを設定
        
        Args:
            grid: Grid3Dオブジェクト
            
        Returns:
            self: メソッドチェーン用
        """
        super().set_grid(grid)
        
        # 部分方程式にもグリッドを設定
        for eq in [self.xy_eq, self.xz_eq, self.yz_eq]:
            if hasattr(eq, 'set_grid'):
                eq.set_grid(grid)
            
        return self
    
    def get_stencil_coefficients(self, i=None, j=None, k=None):
        """
        ステンシル係数を取得（全方向の係数を結合）
        
        Args:
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
            k: z方向のグリッド点インデックス
            
        Returns:
            3D方程式用のステンシル係数
        """
        if self.grid is None:
            raise ValueError("gridが設定されていません。set_grid()で設定してください。")
        
        # 全ての方程式から係数を取得
        xy_coeffs = self.xy_eq.get_stencil_coefficients(i, j, k)
        xz_coeffs = self.xz_eq.get_stencil_coefficients(i, j, k)
        yz_coeffs = self.yz_eq.get_stencil_coefficients(i, j, k)
        
        # 係数を結合（オフセットが重複する場合は加算）
        combined_coeffs = {}
        
        # 全ての方向の係数を追加・結合
        for direction_coeffs in [xy_coeffs, xz_coeffs, yz_coeffs]:
            for offset, coeff in direction_coeffs.items():
                if offset in combined_coeffs:
                    combined_coeffs[offset] += coeff
                else:
                    combined_coeffs[offset] = coeff.copy()
        
        return combined_coeffs
    
    def is_valid_at(self, i=None, j=None, k=None):
        """
        方程式が指定点で有効かどうかを判定（全方向で有効な場合のみ有効）
        
        Args:
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
            k: z方向のグリッド点インデックス
            
        Returns:
            有効性を示すブール値
        """
        if self.grid is None:
            raise ValueError("gridが設定されていません。set_grid()で設定してください。")
        
        # 全ての方向で有効な場合のみ有効
        return (self.xy_eq.is_valid_at(i, j, k) and 
                self.xz_eq.is_valid_at(i, j, k) and 
                self.yz_eq.is_valid_at(i, j, k))