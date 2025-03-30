"""
方程式変換クラスモジュール

このモジュールは、次元の異なる方程式間の変換機能を提供します。
1D→多次元方向性変換と組み合わせ変換を含みます。
"""

import numpy as np
from .dim2.base import Equation2D
from .dim3.base import Equation3D


class DirectionalEquationBase:
    """方向性方程式の基底クラス"""
    
    def __init__(self, equation_1d, direction, direction_only=False, grid=None):
        """
        初期化
        
        Args:
            equation_1d: 1次元方程式のインスタンス
            direction: 方向 ('x', 'y', 'z' など)
            direction_only: 特定の方向のみ処理する場合True
            grid: グリッドオブジェクト
        """
        self.equation_1d = equation_1d
        self.direction = direction
        self.direction_only = direction_only
        self.grid = grid
        
        # 方向に応じたインデックスマッピングを設定
        self._setup_index_mapping()
        
        # 部分方程式にもgridを設定
        if self.grid is not None and hasattr(equation_1d, 'set_grid'):
            self._set_1d_grid_to_equation()
    
    def _setup_index_mapping(self):
        """方向に応じたインデックスマッピングを設定（サブクラスで実装）"""
        raise NotImplementedError("サブクラスで実装する必要があります")
    
    def _set_1d_grid_to_equation(self):
        """1次元方程式に対応するグリッドを設定（サブクラスで実装）"""
        raise NotImplementedError("サブクラスで実装する必要があります")
    
    def set_grid(self, grid):
        """
        グリッドを設定
        
        Args:
            grid: グリッドオブジェクト
            
        Returns:
            self: メソッドチェーン用
        """
        self.grid = grid
        
        # 部分方程式にもグリッドを設定
        self._set_1d_grid_to_equation()
        
        return self
    
    def _get_direction_idx(self, indices):
        """方向に応じたインデックスを取得"""
        if self.direction == 'x':
            return indices[0]  # x方向のインデックス
        elif self.direction == 'y':
            return indices[1]  # y方向のインデックス
        elif self.direction == 'z':
            return indices[2] if len(indices) > 2 else None  # z方向のインデックス
        return None
    
    def _is_boundary_in_direction(self, indices):
        """方向に応じた境界判定"""
        direction_idx = self._get_direction_idx(indices)
        if direction_idx is None:
            return False
            
        if self.direction == 'x':
            return direction_idx == 0 or direction_idx == self.grid.nx_points - 1
        elif self.direction == 'y':
            return direction_idx == 0 or direction_idx == self.grid.ny_points - 1
        elif self.direction == 'z':
            return direction_idx == 0 or direction_idx == self.grid.nz_points - 1
        return False
    
    def _is_other_direction_boundary(self, indices):
        """他の方向での境界判定（サブクラスで実装）"""
        raise NotImplementedError("サブクラスで実装する必要があります")
    
    def _convert_offset(self, offset_1d):
        """1次元オフセットを多次元に変換（サブクラスで実装）"""
        raise NotImplementedError("サブクラスで実装する必要があります")
    
    def _convert_coefficients(self, coeff_array_1d):
        """1次元係数配列を多次元に変換（サブクラスで実装）"""
        raise NotImplementedError("サブクラスで実装する必要があります")
    
    def get_stencil_coefficients(self, *indices):
        """
        ステンシル係数を取得
        
        Args:
            *indices: 次元に応じたグリッド点インデックス
            
        Returns:
            多次元方程式用のステンシル係数
        """
        if self.grid is None:
            raise ValueError("gridが設定されていません。set_grid()で設定してください。")
            
        # 方向に応じた1次元インデックスと境界判定
        i_1d = self._get_direction_idx(indices)
        is_boundary = self._is_boundary_in_direction(indices)
        
        # 境界点で特定方向のみの場合は処理を飛ばす
        if self.direction_only and (is_boundary or self._is_other_direction_boundary(indices)):
            return {}
        
        # 1次元方程式からステンシル係数を取得
        coeffs_1d = self.equation_1d.get_stencil_coefficients(i_1d)
        
        # 1次元の係数を多次元に変換
        coeffs_nd = {}
        for offset_1d, coeff_array_1d in coeffs_1d.items():
            # 方向に応じてオフセットを変換
            offset_nd = self._convert_offset(offset_1d)
            
            # 係数配列を変換
            coeff_array_nd = self._convert_coefficients(coeff_array_1d)
            
            coeffs_nd[offset_nd] = coeff_array_nd
        
        return coeffs_nd
    
    def is_valid_at(self, *indices):
        """
        方程式が指定点で有効かどうかを判定
        
        Args:
            *indices: 次元に応じたグリッド点インデックス
            
        Returns:
            有効性を示すブール値
        """
        if self.grid is None:
            raise ValueError("gridが設定されていません。set_grid()で設定してください。")
        
        # 方向を考慮した境界判定
        if self.direction_only and self._is_other_direction_boundary(indices):
            return False
        
        # 1次元方程式のis_valid_atを利用する
        i_1d = self._get_direction_idx(indices)
        return self.equation_1d.is_valid_at(i_1d)
    
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
                return f"neumann_{self.direction}"
            return eq_type
        return "auxiliary"  # デフォルト


class DirectionalEquation2D(Equation2D, DirectionalEquationBase):
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
        Equation2D.__init__(self, grid)
        DirectionalEquationBase.__init__(self, equation_1d, direction, direction_only, grid)
    
    def _setup_index_mapping(self):
        """方向に応じたインデックスマッピングを設定"""
        # 2次元の未知数順序: [ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy]
        if self.direction == 'x':
            # 1次元の未知数[ψ, ψ', ψ'', ψ''']を
            # 2次元の[ψ, ψ_x, ψ_xx, ψ_xxx]にマッピング
            self.index_map = {0: 0, 1: 1, 2: 2, 3: 3}
        else:  # direction == 'y'
            # 1次元の未知数[ψ, ψ', ψ'', ψ''']を
            # 2次元の[ψ, ψ_y, ψ_yy, ψ_yyy]にマッピング
            self.index_map = {0: 0, 1: 4, 2: 5, 3: 6}
    
    def _set_1d_grid_to_equation(self):
        """1次元方程式に対応するグリッドを設定"""
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
    
    def _is_other_direction_boundary(self, indices):
        """他の方向での境界判定"""
        i, j = indices
        if self.direction == 'x':
            # x方向のみの場合、y方向の境界では無効
            return j == 0 or j == self.grid.ny_points - 1
        else:  # self.direction == 'y'
            # y方向のみの場合、x方向の境界では無効
            return i == 0 or i == self.grid.nx_points - 1
    
    def _convert_offset(self, offset_1d):
        """1次元オフセットを2次元に変換"""
        if self.direction == 'x':
            return (offset_1d, 0)
        else:  # self.direction == 'y'
            return (0, offset_1d)
    
    def _convert_coefficients(self, coeff_array_1d):
        """1次元係数配列を2次元に変換"""
        # 係数配列を変換（7要素の配列に拡張）
        coeff_array_2d = np.zeros(7)
        for idx_1d, idx_2d in self.index_map.items():
            if idx_1d < len(coeff_array_1d):
                coeff_array_2d[idx_2d] = coeff_array_1d[idx_1d]
        
        return coeff_array_2d
    
    def get_stencil_coefficients(self, i=None, j=None):
        """
        2D用のインターフェースを維持
        
        Args:
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
        """
        if i is None or j is None:
            raise ValueError("グリッド点のインデックスiとjを指定する必要があります。")
        
        return DirectionalEquationBase.get_stencil_coefficients(self, i, j)
    
    def is_valid_at(self, i=None, j=None):
        """
        2D用のインターフェースを維持
        
        Args:
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
        """
        if i is None or j is None:
            raise ValueError("グリッド点のインデックスiとjを指定する必要があります。")
        
        return DirectionalEquationBase.is_valid_at(self, i, j)


class DirectionalEquation3D(Equation3D, DirectionalEquationBase):
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
        Equation3D.__init__(self, grid)
        DirectionalEquationBase.__init__(self, equation_1d, direction, direction_only, grid)
    
    def _setup_index_mapping(self):
        """方向に応じたインデックスマッピングを設定"""
        # 3次元の未知数順序: [ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy, ψ_z, ψ_zz, ψ_zzz]
        if self.direction == 'x':
            # 1次元の未知数[ψ, ψ', ψ'', ψ''']を3次元の[ψ, ψ_x, ψ_xx, ψ_xxx, 0, 0, 0, 0, 0, 0]にマッピング
            self.index_map = {0: 0, 1: 1, 2: 2, 3: 3}
        elif self.direction == 'y':
            # 1次元の未知数[ψ, ψ', ψ'', ψ''']を3次元の[ψ, 0, 0, 0, ψ_y, ψ_yy, ψ_yyy, 0, 0, 0]にマッピング
            self.index_map = {0: 0, 1: 4, 2: 5, 3: 6}
        else:  # direction == 'z'
            # 1次元の未知数[ψ, ψ', ψ'', ψ''']を3次元の[ψ, 0, 0, 0, 0, 0, 0, ψ_z, ψ_zz, ψ_zzz]にマッピング
            self.index_map = {0: 0, 1: 7, 2: 8, 3: 9}
    
    def _set_1d_grid_to_equation(self):
        """1次元方程式に対応するグリッドを設定"""
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
    
    def _is_other_direction_boundary(self, indices):
        """他の方向での境界判定"""
        i, j, k = indices
        if self.direction == 'x':
            # x方向のみの場合、y・z方向の境界では無効
            return (j == 0 or j == self.grid.ny_points - 1 or 
                    k == 0 or k == self.grid.nz_points - 1)
        elif self.direction == 'y':
            # y方向のみの場合、x・z方向の境界では無効
            return (i == 0 or i == self.grid.nx_points - 1 or 
                    k == 0 or k == self.grid.nz_points - 1)
        else:  # self.direction == 'z'
            # z方向のみの場合、x・y方向の境界では無効
            return (i == 0 or i == self.grid.nx_points - 1 or
                    j == 0 or j == self.grid.ny_points - 1)
    
    def _convert_offset(self, offset_1d):
        """1次元オフセットを3次元に変換"""
        if self.direction == 'x':
            return (offset_1d, 0, 0)
        elif self.direction == 'y':
            return (0, offset_1d, 0)
        else:  # self.direction == 'z'
            return (0, 0, offset_1d)
    
    def _convert_coefficients(self, coeff_array_1d):
        """1次元係数配列を3次元に変換"""
        # 係数配列を変換（10要素の配列に拡張）
        coeff_array_3d = np.zeros(10)
        for idx_1d, idx_3d in self.index_map.items():
            if idx_1d < len(coeff_array_1d):
                coeff_array_3d[idx_3d] = coeff_array_1d[idx_1d]
        
        return coeff_array_3d
    
    def get_stencil_coefficients(self, i=None, j=None, k=None):
        """
        3D用のインターフェースを維持
        
        Args:
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
            k: z方向のグリッド点インデックス
        """
        if i is None or j is None or k is None:
            raise ValueError("グリッド点のインデックスi, j, kを指定する必要があります。")
        
        return DirectionalEquationBase.get_stencil_coefficients(self, i, j, k)
    
    def is_valid_at(self, i=None, j=None, k=None):
        """
        3D用のインターフェースを維持
        
        Args:
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
            k: z方向のグリッド点インデックス
        """
        if i is None or j is None or k is None:
            raise ValueError("グリッド点のインデックスi, j, kを指定する必要があります。")
        
        return DirectionalEquationBase.is_valid_at(self, i, j, k)


class CombinedEquationBase:
    """組み合わせ方程式の基底クラス"""
    
    def __init__(self, grid=None):
        """
        初期化
        
        Args:
            grid: グリッドオブジェクト
        """
        self.grid = grid
        self.equations = []
    
    def set_grid(self, grid):
        """
        グリッドを設定
        
        Args:
            grid: グリッドオブジェクト
            
        Returns:
            self: メソッドチェーン用
        """
        self.grid = grid
        
        # 部分方程式にもグリッドを設定
        for eq in self.equations:
            if hasattr(eq, 'set_grid'):
                eq.set_grid(grid)
            
        return self
    
    def get_stencil_coefficients(self, *indices):
        """
        ステンシル係数を取得（全方向の係数を結合）
        
        Args:
            *indices: 次元に応じたグリッド点インデックス
            
        Returns:
            方程式用のステンシル係数
        """
        if self.grid is None:
            raise ValueError("gridが設定されていません。set_grid()で設定してください。")
        
        # 全ての方程式から係数を取得
        combined_coeffs = {}
        
        # 各方程式の係数を追加・結合
        for eq in self.equations:
            eq_coeffs = eq.get_stencil_coefficients(*indices)
            
            for offset, coeff in eq_coeffs.items():
                if offset in combined_coeffs:
                    combined_coeffs[offset] += coeff
                else:
                    combined_coeffs[offset] = coeff.copy()
        
        return combined_coeffs
    
    def is_valid_at(self, *indices):
        """
        方程式が指定点で有効かどうかを判定（全方向で有効な場合のみ有効）
        
        Args:
            *indices: 次元に応じたグリッド点インデックス
            
        Returns:
            有効性を示すブール値
        """
        if self.grid is None:
            raise ValueError("gridが設定されていません。set_grid()で設定してください。")
        
        # 全ての方程式が有効な場合のみ有効
        for eq in self.equations:
            if not eq.is_valid_at(*indices):
                return False
                
        return True
    
    def get_equation_type(self):
        """
        組み合わせた方程式の種類を返す
        
        Returns:
            str: 方程式の種類
        """
        # 全ての方程式のタイプを取得
        eq_types = []
        for eq in self.equations:
            eq_type = eq.get_equation_type() if hasattr(eq, 'get_equation_type') else "auxiliary"
            eq_types.append(eq_type)
        
        # 優先順位: governing > dirichlet > neumann_x/y/z > auxiliary
        if "governing" in eq_types:
            return "governing"
        if "dirichlet" in eq_types:
            return "dirichlet"
        
        # ノイマン境界条件の優先順位: x > y > z
        if "neumann_x" in eq_types:
            return "neumann_x"
        if "neumann_y" in eq_types:
            return "neumann_y"
        if "neumann_z" in eq_types:
            return "neumann_z"
        
        return "auxiliary"


class CombinedDirectionalEquation2D(Equation2D, CombinedEquationBase):
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
                
        Equation2D.__init__(self, grid)
        CombinedEquationBase.__init__(self, grid)
        
        self.equations = [x_direction_eq, y_direction_eq]
        
        # 部分方程式にもgridを設定
        self.set_grid(grid)
    
    def get_stencil_coefficients(self, i=None, j=None):
        """
        2D用のインターフェースを維持
        
        Args:
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
        """
        if i is None or j is None:
            raise ValueError("グリッド点のインデックスiとjを指定する必要があります。")
        
        return CombinedEquationBase.get_stencil_coefficients(self, i, j)
    
    def is_valid_at(self, i=None, j=None):
        """
        2D用のインターフェースを維持
        
        Args:
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
        """
        if i is None or j is None:
            raise ValueError("グリッド点のインデックスiとjを指定する必要があります。")
        
        return CombinedEquationBase.is_valid_at(self, i, j)


class CombinedDirectionalEquation3D(Equation3D, CombinedEquationBase):
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
                
        Equation3D.__init__(self, grid)
        CombinedEquationBase.__init__(self, grid)
        
        self.equations = [x_direction_eq, y_direction_eq, z_direction_eq]
        
        # 部分方程式にもgridを設定
        self.set_grid(grid)
    
    def get_stencil_coefficients(self, i=None, j=None, k=None):
        """
        3D用のインターフェースを維持
        
        Args:
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
            k: z方向のグリッド点インデックス
        """
        if i is None or j is None or k is None:
            raise ValueError("グリッド点のインデックスi, j, kを指定する必要があります。")
        
        return CombinedEquationBase.get_stencil_coefficients(self, i, j, k)
    
    def is_valid_at(self, i=None, j=None, k=None):
        """
        3D用のインターフェースを維持
        
        Args:
            i: x方向のグリッド点インデックス
            j: y方向のグリッド点インデックス
            k: z方向のグリッド点インデックス
        """
        if i is None or j is None or k is None:
            raise ValueError("グリッド点のインデックスi, j, kを指定する必要があります。")
        
        return CombinedEquationBase.is_valid_at(self, i, j, k)


# 以下は既存のファクトリークラスで、互換性のために維持
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