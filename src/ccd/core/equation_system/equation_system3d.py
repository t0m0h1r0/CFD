"""
3次元方程式システムの定義モジュール

このモジュールは、CCD (Combined Compact Difference) 法で使用される
3次元問題の格子点方程式を管理し、線形方程式系の行列を構築するための機能を提供します。
"""

from core.base.base_equation_system import BaseEquationSystem

class EquationSystem3D(BaseEquationSystem):
    """3次元方程式システムを管理するクラス"""
    
    # 3D格子の領域タイプ定義
    REGION_TYPES = {
        # 内部
        'interior': {'desc': '内部点'},
        
        # 面領域
        'face_x_min': {'desc': 'x最小面 (i=0)'},
        'face_x_max': {'desc': 'x最大面 (i=nx-1)'},
        'face_y_min': {'desc': 'y最小面 (j=0)'},
        'face_y_max': {'desc': 'y最大面 (j=ny-1)'},
        'face_z_min': {'desc': 'z最小面 (k=0)'},
        'face_z_max': {'desc': 'z最大面 (k=nz-1)'},
        
        # 辺領域（x方向辺）
        'edge_x_y_min_z_min': {'desc': '(0<i<nx-1, j=0, k=0)'},
        'edge_x_y_min_z_max': {'desc': '(0<i<nx-1, j=0, k=nz-1)'},
        'edge_x_y_max_z_min': {'desc': '(0<i<nx-1, j=ny-1, k=0)'},
        'edge_x_y_max_z_max': {'desc': '(0<i<nx-1, j=ny-1, k=nz-1)'},
        
        # 辺領域（y方向辺）
        'edge_y_x_min_z_min': {'desc': '(i=0, 0<j<ny-1, k=0)'},
        'edge_y_x_min_z_max': {'desc': '(i=0, 0<j<ny-1, k=nz-1)'},
        'edge_y_x_max_z_min': {'desc': '(i=nx-1, 0<j<ny-1, k=0)'},
        'edge_y_x_max_z_max': {'desc': '(i=nx-1, 0<j<ny-1, k=nz-1)'},
        
        # 辺領域（z方向辺）
        'edge_z_x_min_y_min': {'desc': '(i=0, j=0, 0<k<nz-1)'},
        'edge_z_x_min_y_max': {'desc': '(i=0, j=ny-1, 0<k<nz-1)'},
        'edge_z_x_max_y_min': {'desc': '(i=nx-1, j=0, 0<k<nz-1)'},
        'edge_z_x_max_y_max': {'desc': '(i=nx-1, j=ny-1, 0<k<nz-1)'},
        
        # 頂点領域
        'vertex_x_min_y_min_z_min': {'desc': '(i=0, j=0, k=0)'},
        'vertex_x_min_y_min_z_max': {'desc': '(i=0, j=0, k=nz-1)'},
        'vertex_x_min_y_max_z_min': {'desc': '(i=0, j=ny-1, k=0)'},
        'vertex_x_min_y_max_z_max': {'desc': '(i=0, j=ny-1, k=nz-1)'},
        'vertex_x_max_y_min_z_min': {'desc': '(i=nx-1, j=0, k=0)'},
        'vertex_x_max_y_min_z_max': {'desc': '(i=nx-1, j=0, k=nz-1)'},
        'vertex_x_max_y_max_z_min': {'desc': '(i=nx-1, j=ny-1, k=0)'},
        'vertex_x_max_y_max_z_max': {'desc': '(i=nx-1, j=ny-1, k=nz-1)'},
    }

    def _initialize_equations(self):
        """3D用の方程式コレクションを初期化"""
        # 領域ごとの方程式コレクション
        self.equations = {region_type: [] for region_type in self.REGION_TYPES}
        
        # 位置キャッシュ
        self._location_cache = {}
    
    def _get_point_location(self, i, j=None, k=None):
        """
        格子点の位置タイプを判定
        
        Args:
            i: x方向のインデックス
            j: y方向のインデックス
            k: z方向のインデックス
            
        Returns:
            位置を表す文字列
        """
        if j is None or k is None:
            raise ValueError("3D格子では j, k インデックスが必要です")
            
        # キャッシュをチェック
        cache_key = (i, j, k)
        if cache_key in self._location_cache:
            return self._location_cache[cache_key]
        
        # Grid3Dの境界タイプ判定を利用
        result = self.grid.get_boundary_type(i, j, k)
        self._location_cache[cache_key] = result
        return result
    
    def _assign_equations_3d(self, eq_by_type, i, j, k):
        """
        3D格子点における方程式の割り当てを決定
        
        Args:
            eq_by_type: 種類別に分類された方程式
            i: x方向のインデックス
            j: y方向のインデックス
            k: z方向のインデックス
            
        Returns:
            各行に割り当てる方程式のリスト [0行目, 1行目, ..., 9行目]
        """
        governing = eq_by_type["governing"]
        dirichlet = eq_by_type["dirichlet"]
        neumann_x = eq_by_type["neumann_x"]
        neumann_y = eq_by_type["neumann_y"]
        neumann_z = eq_by_type["neumann_z"]
        auxiliary = eq_by_type["auxiliary"].copy()  # コピーして操作
        
        # デフォルト方程式
        fallback = governing
        
        # 割り当て結果
        assignments = [None] * 10
        
        # 方程式を割り当てるヘルパー関数
        def assign_with_fallback(idx, primary=None, use_auxiliary=False, tertiary=None):
            if primary:
                assignments[idx] = primary
            elif use_auxiliary and auxiliary:
                assignments[idx] = auxiliary.pop(0)
            elif tertiary:
                assignments[idx] = tertiary
            else:
                assignments[idx] = fallback
        
        # 0行目: 支配方程式 (ψ)
        assignments[0] = governing
        
        # 1行目: x方向のディリクレ境界または補助方程式 (ψ_x)
        assign_with_fallback(1, dirichlet, True)
        
        # 2行目: x方向のノイマン境界または補助方程式 (ψ_xx)
        assign_with_fallback(2, neumann_x, True, dirichlet)
        
        # 3行目: 補助方程式 (ψ_xxx)
        assign_with_fallback(3, None, True, neumann_x or dirichlet)
        
        # 4行目: y方向のディリクレ境界または補助方程式 (ψ_y)
        assign_with_fallback(4, dirichlet if dirichlet and dirichlet != assignments[1] else None, True)
        
        # 5行目: y方向のノイマン境界または補助方程式 (ψ_yy)
        assign_with_fallback(5, neumann_y, True, dirichlet)
        
        # 6行目: 補助方程式 (ψ_yyy)
        assign_with_fallback(6, None, True, neumann_y or dirichlet)
        
        # 7行目: z方向のディリクレ境界または補助方程式 (ψ_z)
        assign_with_fallback(7, dirichlet if dirichlet and dirichlet != assignments[1] and dirichlet != assignments[4] else None, True)
        
        # 8行目: z方向のノイマン境界または補助方程式 (ψ_zz)
        assign_with_fallback(8, neumann_z, True, dirichlet)
        
        # 9行目: 補助方程式 (ψ_zzz)
        assign_with_fallback(9, None, True, neumann_z or dirichlet)
        
        return assignments
    
    # 基底クラスの抽象メソッドを実装（1D, 2D用）
    def _assign_equations_1d(self, eq_by_type, i, j=None, k=None):
        # 3Dシステムでは呼び出されないはずなので例外を投げる
        raise NotImplementedError("3Dシステムで_assign_equations_1dが呼び出されました")
        
    def _assign_equations_2d(self, eq_by_type, i, j, k=None):
        # 3Dシステムでは呼び出されないはずなので例外を投げる
        raise NotImplementedError("3Dシステムで_assign_equations_2dが呼び出されました")