import numpy as np
from .dim2.base import Equation2D
from .dim3.base import Equation3D

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
            x_only: True の場合、y方向とz方向の成分を無視
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
            y_only: True の場合、x方向とz方向の成分を無視
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
            z_only: True の場合、x方向とy方向の成分を無視
            grid: Grid3Dオブジェクト
            
        Returns:
            z方向用の3次元方程式インスタンス
        """
        return DirectionalEquation3D(equation_1d, 'z', z_only, grid)
    
    @staticmethod
    def to_xyz(equation_1d_x=None, equation_1d_y=None, equation_1d_z=None, grid=None):
        """
        異なる方程式をx方向、y方向、z方向に適用
        
        Args:
            equation_1d_x: x方向に適用する1次元方程式
            equation_1d_y: y方向に適用する1次元方程式（指定しない場合はequation_1d_xと同じ）
            equation_1d_z: z方向に適用する1次元方程式（指定しない場合はequation_1d_xと同じ）
            grid: Grid3Dオブジェクト
            
        Returns:
            全方向に対応する3次元方程式インスタンス
        """
        if equation_1d_x is None:
            raise ValueError("少なくとも1つの方程式を指定する必要があります")
            
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
            coeff_array_2d = np.zeros(7)
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
            
    def get_equation_type(self):
        """
        方程式の種類を方向を考慮して返す
        
        Returns:
            str: 方程式の種類
        """
        if hasattr(self.equation_1d, 'get_equation_type'):
            eq_type = self.equation_1d.get_equation_type()
            # ノイマン条件の場合は方向を付加
            if eq_type == "neumann":
                if self.direction == 'x':
                    return "neumann_x"
                elif self.direction == 'y':
                    return "neumann_y"
            return eq_type
        return super().get_equation_type()


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
        
    def get_equation_type(self):
        """
        組み合わせた方程式の種類を返す
        
        Returns:
            str: 方程式の種類
        """
        x_type = self.x_eq.get_equation_type() if hasattr(self.x_eq, 'get_equation_type') else "auxiliary"
        y_type = self.y_eq.get_equation_type() if hasattr(self.y_eq, 'get_equation_type') else "auxiliary"
        
        # 優先順位: governing > dirichlet > neumann_x/y > auxiliary
        if "governing" in [x_type, y_type]:
            return "governing"
        if "dirichlet" in [x_type, y_type]:
            return "dirichlet"
        if "neumann_x" in [x_type, y_type]:
            return "neumann_x"
        if "neumann_y" in [x_type, y_type]:
            return "neumann_y"
        return "auxiliary"


class DirectionalEquation3D(Equation3D):
    """
    1次元方程式を指定方向の3次元方程式に変換するアダプタクラス
    """
    
    def __init__(self, equation_1d, direction='x', direction_only=False, grid=None):
        """
        初期化
        
        Args:
            equation_1d: 1次元方程式のインスタンス
            direction: 'x'、'y'または'z'
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
            # 3次元の[ψ, ψ_x, ψ_xx, ψ_xxx, 0, 0, 0, 0, 0, 0]にマッピング
            self.index_map = {0: 0, 1: 1, 2: 2, 3: 3}
        elif direction == 'y':
            # 1次元の未知数[ψ, ψ', ψ'', ψ''']を
            # 3次元の[ψ, 0, 0, 0, ψ_y, ψ_yy, ψ_yyy, 0, 0, 0]にマッピング
            self.index_map = {0: 0, 1: 4, 2: 5, 3: 6}
        else:  # direction == 'z'
            # 1次元の未知数[ψ, ψ', ψ'', ψ''']を
            # 3次元の[ψ, 0, 0, 0, 0, 0, 0, ψ_z, ψ_zz, ψ_zzz]にマッピング
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
                self.grid.get_spacing()[0], 
                self.grid.nx_points
            )
        elif self.direction == 'y':
            emulated_grid = Grid1DEmulator(
                self.grid.y, 
                self.grid.get_spacing()[1], 
                self.grid.ny_points
            )
        else:  # self.direction == 'z'
            emulated_grid = Grid1DEmulator(
                self.grid.z, 
                self.grid.get_spacing()[2], 
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
            is_boundary = (j == 0 or j == self.grid.ny_points - 1 or 
                           k == 0 or k == self.grid.nz_points - 1)
        elif self.direction == 'y':
            i_1d = j  # y方向のインデックス
            is_boundary = (i == 0 or i == self.grid.nx_points - 1 or 
                           k == 0 or k == self.grid.nz_points - 1)
        else:  # self.direction == 'z'
            i_1d = k  # z方向のインデックス
            is_boundary = (i == 0 or i == self.grid.nx_points - 1 or 
                           j == 0 or j == self.grid.ny_points - 1)
        
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
            coeff_array_3d = np.zeros(10)
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
        
        # 1次元方程式のis_valid_atを利用する
        # 方向を考慮した1Dグリッドのエミュレータを作成
        class Grid1DEmulator:
            def __init__(self, n_points):
                self.n_points = n_points
        
        if self.direction == 'x':
            # x方向の境界判定
            if self.direction_only:
                # x方向のみの場合、内部点およびx方向の境界のみで有効（y・z方向境界では無効）
                if (j == 0 or j == self.grid.ny_points - 1 or 
                    k == 0 or k == self.grid.nz_points - 1):
                    return False
            
            # 1次元方程式のvalid判定を使用
            emulator = Grid1DEmulator(self.grid.nx_points)
            return self.equation_1d.is_valid_at(i)
            
        elif self.direction == 'y':
            # y方向の境界判定
            if self.direction_only:
                # y方向のみの場合、内部点およびy方向の境界のみで有効（x・z方向境界では無効）
                if (i == 0 or i == self.grid.nx_points - 1 or 
                    k == 0 or k == self.grid.nz_points - 1):
                    return False
            
            # 1次元方程式のvalid判定を使用
            emulator = Grid1DEmulator(self.grid.ny_points)
            return self.equation_1d.is_valid_at(j)
            
        else:  # self.direction == 'z'
            # z方向の境界判定
            if self.direction_only:
                # z方向のみの場合、内部点およびz方向の境界のみで有効（x・y方向境界では無効）
                if (i == 0 or i == self.grid.nx_points - 1 or
                    j == 0 or j == self.grid.ny_points - 1):
                    return False
            
            # 1次元方程式のvalid判定を使用
            emulator = Grid1DEmulator(self.grid.nz_points)
            return self.equation_1d.is_valid_at(k)
            
    def get_equation_type(self):
        """
        方程式の種類を方向を考慮して返す
        
        Returns:
            str: 方程式の種類
        """
        if hasattr(self.equation_1d, 'get_equation_type'):
            eq_type = self.equation_1d.get_equation_type()
            # ノイマン条件の場合は方向を付加
            if eq_type == "neumann":
                if self.direction == 'x':
                    return "neumann_x"
                elif self.direction == 'y':
                    return "neumann_y"
                elif self.direction == 'z':
                    return "neumann_z"
            return eq_type
        return super().get_equation_type()


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
            for eq in [x_direction_eq, y_direction_eq, z_direction_eq]:
                if hasattr(eq, 'grid') and eq.grid is not None:
                    grid = eq.grid
                    break
                
        super().__init__(grid)
        self.x_eq = x_direction_eq
        self.y_eq = y_direction_eq
        self.z_eq = z_direction_eq
        
        # 部分方程式にもgridを設定
        if self.grid is not None:
            for eq in [x_direction_eq, y_direction_eq, z_direction_eq]:
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
        for eq in [self.x_eq, self.y_eq, self.z_eq]:
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
        x_coeffs = self.x_eq.get_stencil_coefficients(i, j, k)
        y_coeffs = self.y_eq.get_stencil_coefficients(i, j, k)
        z_coeffs = self.z_eq.get_stencil_coefficients(i, j, k)
        
        # 係数を結合（オフセットが重複する場合は加算）
        combined_coeffs = {}
        
        # 全ての方向の係数を追加・結合
        for direction_coeffs in [x_coeffs, y_coeffs, z_coeffs]:
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
        return (self.x_eq.is_valid_at(i, j, k) and 
                self.y_eq.is_valid_at(i, j, k) and 
                self.z_eq.is_valid_at(i, j, k))
                
    def get_equation_type(self):
        """
        組み合わせた方程式の種類を返す
        
        Returns:
            str: 方程式の種類
        """
        x_type = self.x_eq.get_equation_type() if hasattr(self.x_eq, 'get_equation_type') else "auxiliary"
        y_type = self.y_eq.get_equation_type() if hasattr(self.y_eq, 'get_equation_type') else "auxiliary"
        z_type = self.z_eq.get_equation_type() if hasattr(self.z_eq, 'get_equation_type') else "auxiliary"
        
        # 優先順位: governing > dirichlet > neumann_x/y/z > auxiliary
        types = [x_type, y_type, z_type]
        if "governing" in types:
            return "governing"
        if "dirichlet" in types:
            return "dirichlet"
        
        # ノイマン境界条件の優先順位: x > y > z
        if "neumann_x" in types:
            return "neumann_x"
        if "neumann_y" in types:
            return "neumann_y"
        if "neumann_z" in types:
            return "neumann_z"
        
        return "auxiliary"