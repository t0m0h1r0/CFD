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


class Equation1Dto3DConverter:
    """1次元方程式を3次元方程式に変換するファクトリクラス"""
    
    @staticmethod
    def to_x(equation_1d, x_only=False, grid=None):
        """
        1次元方程式をx方向の3次元方程式に変換
        
        Args:
            equation_1d: 1次元方程式クラスのインスタンス
            x_only: True の場合、y/z方向の成分を無視
            grid: Grid3Dオブジェクト
            
        Returns:
            x方向用の3次元方程式インスタンス
        """
        return DirectionalEquation3D(equation_1d, 'x', x_only, grid)
    
    @staticmethod
    def to_y(equation_1d, y_only=False, grid=None):
        """
        1次元方程式をy方向の3次元方程式に変換
        
        Args:
            equation_1d: 1次元方程式クラスのインスタンス
            y_only: True の場合、x/z方向の成分を無視
            grid: Grid3Dオブジェクト
            
        Returns:
            y方向用の3次元方程式インスタンス
        """
        return DirectionalEquation3D(equation_1d, 'y', y_only, grid)
    
    @staticmethod
    def to_z(equation_1d, z_only=False, grid=None):
        """
        1次元方程式をz方向の3次元方程式に変換
        
        Args:
            equation_1d: 1次元方程式クラスのインスタンス
            z_only: True の場合、x/y方向の成分を無視
            grid: Grid3Dオブジェクト
            
        Returns:
            z方向用の3次元方程式インスタンス
        """
        return DirectionalEquation3D(equation_1d, 'z', z_only, grid)
    
    @staticmethod
    def to_xyz(equation_1d_x, equation_1d_y=None, equation_1d_z=None, grid=None):
        """
        異なる方程式をx, y, z方向に適用
        
        Args:
            equation_1d_x: x方向に適用する1次元方程式
            equation_1d_y: y方向に適用する1次元方程式（指定しない場合はequation_1d_xと同じ）
            equation_1d_z: z方向に適用する1次元方程式（指定しない場合はequation_1d_xと同じ）
            grid: Grid3Dオブジェクト
            
        Returns:
            全方向に対応する3次元方程式インスタンス
        """
        if equation_1d_y is None:
            equation_1d_y = equation_1d_x
        if equation_1d_z is None:
            equation_1d_z = equation_1d_x
            
        x_eq = DirectionalEquation3D(equation_1d_x, 'x', False, grid)
        y_eq = DirectionalEquation3D(equation_1d_y, 'y', False, grid)
        z_eq = DirectionalEquation3D(equation_1d_z, 'z', False, grid)
        
        return CombinedDirectionalEquation3D(x_eq, y_eq, z_eq, grid)


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
            
    # 既存の実装は省略（ファイル内に既に存在する）


class DirectionalEquation3D(Equation3D):
    """
    1次元方程式を指定方向の3次元方程式に変換するアダプタクラス
    """
    
    def __init__(self, equation_1d, direction='x', direction_only=False, grid=None):
        """
        初期化
        
        Args:
            equation_1d: 1次元方程式のインスタンス
            direction: 'x', 'y', または 'z'
            direction_only: 特定の方向のみ処理する場合True
            grid: Grid3Dオブジェクト
        """
        super().__init__(grid)
        self.equation_1d = equation_1d
        self.direction = direction
        self.direction_only = direction_only
        
        # 方向に応じたインデックスマッピング
        # 3次元の未知数順序: [ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy, ψ_z, ψ_zz, ψ_zzz]
        if direction == 'x':
            # 1次元の未知数[ψ, ψ', ψ'', ψ''']を
            # 3次元の[ψ, ψ_x, ψ_xx, ψ_xxx]にマッピング
            self.index_map = {0: 0, 1: 1, 2: 2, 3: 3}
        elif direction == 'y':
            # 1次元の未知数[ψ, ψ', ψ'', ψ''']を
            # 3次元の[ψ, ψ_y, ψ_yy, ψ_yyy]にマッピング
            self.index_map = {0: 0, 1: 4, 2: 5, 3: 6}
        else:  # direction == 'z'
            # 1次元の未知数[ψ, ψ', ψ'', ψ''']を
            # 3次元の[ψ, ψ_z, ψ_zz, ψ_zzz]にマッピング
            self.index_map = {0: 0, 1: 7, 2: 8, 3: 9}
            
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
                self.grid.get_spacing()[0] if self.grid.is_3d else self.grid.hx, 
                self.grid.nx_points
            )
        elif self.direction == 'y':
            emulated_grid = Grid1DEmulator(
                self.grid.y, 
                self.grid.get_spacing()[1] if self.grid.is_3d else self.grid.hy, 
                self.grid.ny_points
            )
        else:  # self.direction == 'z'
            emulated_grid = Grid1DEmulator(
                self.grid.z, 
                self.grid.get_spacing()[2] if self.grid.is_3d else self.grid.hz, 
                self.grid.nz_points
            )
            
        # 1次元方程式にエミュレートされたグリッドを設定
        if hasattr(self.equation_1d, 'set_grid'):
            self.equation_1d.set_grid(emulated_grid)
    
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
        self._set_1d_grid_to_equation()
        
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
        
        # 方向に応じた1次元のインデックスと境界判定
        if self.direction == 'x':
            i_1d = i  # x方向のインデックス
            is_boundary = i == 0 or i == self.grid.nx_points - 1
        elif self.direction == 'y':
            i_1d = j  # y方向のインデックス
            is_boundary = j == 0 or j == self.grid.ny_points - 1
        else:  # self.direction == 'z'
            i_1d = k  # z方向のインデックス
            is_boundary = k == 0 or k == self.grid.nz_points - 1
        
        # 境界点で特定方向のみの場合は処理を飛ばす
        if self.direction_only and is_boundary:
            return {}
        
        # 1次元方程式からステンシル係数を取得
        coeffs_1d = self.equation_1d.get_stencil_coefficients(i_1d)
        
        # 1次元の係数を3次元に変換
        coeffs_3d = {}
        for offset_1d, coeff_array_1d in coeffs_1d.items():
            # 方向に応じてオフセットを変換
            if self.direction == 'x':
                offset_3d = (offset_1d, 0, 0)
            elif self.direction == 'y':
                offset_3d = (0, offset_1d, 0)
            else:  # self.direction == 'z'
                offset_3d = (0, 0, offset_1d)
            
            # 係数配列を変換（10要素の配列に拡張）
            coeff_array_3d = cp.zeros(10)
            for idx_1d, idx_3d in self.index_map.items():
                if idx_1d < len(coeff_array_1d):
                    coeff_array_3d[idx_3d] = coeff_array_1d[idx_1d]
            
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
        
        # 方向に応じた1次元のインデックスを選択
        if self.direction == 'x':
            i_1d = i
            # 特定方向のみの場合、他の方向の境界では無効
            if self.direction_only and (j == 0 or j == self.grid.ny_points - 1 or 
                                         k == 0 or k == self.grid.nz_points - 1):
                return False
        elif self.direction == 'y':
            i_1d = j
            # 特定方向のみの場合、他の方向の境界では無効
            if self.direction_only and (i == 0 or i == self.grid.nx_points - 1 or 
                                         k == 0 or k == self.grid.nz_points - 1):
                return False
        else:  # self.direction == 'z'
            i_1d = k
            # 特定方向のみの場合、他の方向の境界では無効
            if self.direction_only and (i == 0 or i == self.grid.nx_points - 1 or 
                                         j == 0 or j == self.grid.ny_points - 1):
                return False
        
        # 1次元方程式のvalid判定を使用
        class Grid1DEmulator:
            def __init__(self, n_points):
                self.n_points = n_points
                
        if self.direction == 'x':
            return self.equation_1d.is_valid_at(i_1d)
        elif self.direction == 'y':
            return self.equation_1d.is_valid_at(i_1d)
        else:  # self.direction == 'z'
            return self.equation_1d.is_valid_at(i_1d)


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
        # コンストラクタと他のメソッドは既存の実装を使用
        super().__init__(grid)
        self.x_eq = x_direction_eq
        self.y_eq = y_direction_eq
        
        # 部分方程式にもgridを設定
        if self.grid is not None:
            if hasattr(x_direction_eq, 'set_grid'):
                x_direction_eq.set_grid(self.grid)
            if hasattr(y_direction_eq, 'set_grid'):
                y_direction_eq.set_grid(self.grid)
                
    # 残りのメソッドは既存の実装を使用


class CombinedDirectionalEquation3D(Equation3D):
    """
    x方向、y方向、z方向の3次元方程式を組み合わせたクラス
    """
    
    def __init__(self, x_direction_eq, y_direction_eq, z_direction_eq, grid=None):
        """
        初期化
        
        Args:
            x_direction_eq: x方向の3次元方程式
            y_direction_eq: y方向の3次元方程式
            z_direction_eq: z方向の3次元方程式
            grid: Grid3Dオブジェクト
        """
        # gridが指定されていなければ、部分方程式のものを使用
        if grid is None:
            if hasattr(x_direction_eq, 'grid') and x_direction_eq.grid is not None:
                grid = x_direction_eq.grid
            elif hasattr(y_direction_eq, 'grid') and y_direction_eq.grid is not None:
                grid = y_direction_eq.grid
            elif hasattr(z_direction_eq, 'grid') and z_direction_eq.grid is not None:
                grid = z_direction_eq.grid
                
        super().__init__(grid)
        self.x_eq = x_direction_eq
        self.y_eq = y_direction_eq
        self.z_eq = z_direction_eq
        
        # 部分方程式にもgridを設定
        if self.grid is not None:
            if hasattr(x_direction_eq, 'set_grid'):
                x_direction_eq.set_grid(self.grid)
            if hasattr(y_direction_eq, 'set_grid'):
                y_direction_eq.set_grid(self.grid)
            if hasattr(z_direction_eq, 'set_grid'):
                z_direction_eq.set_grid(self.grid)
    
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
        if hasattr(self.x_eq, 'set_grid'):
            self.x_eq.set_grid(grid)
        if hasattr(self.y_eq, 'set_grid'):
            self.y_eq.set_grid(grid)
        if hasattr(self.z_eq, 'set_grid'):
            self.z_eq.set_grid(grid)
            
        return self
    
    def get_stencil_coefficients(self, i=None, j=None, k=None):
        """
        ステンシル係数を取得（3方向の係数を結合）
        
        Args:
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
            k: z方向のグリッド点インデックス
            
        Returns:
            3D方程式用のステンシル係数
        """
        if self.grid is None:
            raise ValueError("gridが設定されていません。set_grid()で設定してください。")
        
        # 各方向の方程式から係数を取得
        x_coeffs = self.x_eq.get_stencil_coefficients(i, j, k)
        y_coeffs = self.y_eq.get_stencil_coefficients(i, j, k)
        z_coeffs = self.z_eq.get_stencil_coefficients(i, j, k)
        
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
        
        # z方向の係数を追加・結合
        for offset, coeff in z_coeffs.items():
            if offset in combined_coeffs:
                combined_coeffs[offset] += coeff
            else:
                combined_coeffs[offset] = coeff
        
        return combined_coeffs
    
    def is_valid_at(self, i=None, j=None, k=None):
        """
        方程式が指定点で有効かどうかを判定（3方向とも有効な場合のみ有効）
        
        Args:
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
            k: z方向のグリッド点インデックス
            
        Returns:
            有効性を示すブール値
        """
        if self.grid is None:
            raise ValueError("gridが設定されていません。set_grid()で設定してください。")
        
        # 全ての方向の方程式が有効な場合のみ有効
        return (self.x_eq.is_valid_at(i, j, k) and 
                self.y_eq.is_valid_at(i, j, k) and 
                self.z_eq.is_valid_at(i, j, k))