"""
3次元方程式セットの定義を行うモジュール

このモジュールでは、CCD（Combined Compact Difference）法に使用される
3次元の方程式セットを定義します。
"""

from core.base.base_equation_set import EquationSet

# 共通の方程式をインポート
from equation.poisson import PoissonEquation3D
from equation.original import OriginalEquation3D
from equation.boundary import DirichletBoundaryEquation3D
from equation.equation_converter import Equation2Dto3DConverter

class DerivativeEquationSet3D(EquationSet):
    """3次元高階微分のための方程式セット"""

    def __init__(self):
        """初期化"""
        super().__init__()
        # 微分方程式セットでは境界条件は無効
        self.enable_dirichlet = False
        self.enable_neumann = False
    
    def setup_equations(self, system, grid, test_func=None):
        """
        3次元導関数計算用の方程式システムを設定
        
        Args:
            system: 方程式システム
            grid: Grid オブジェクト (3D)
            test_func: テスト関数（オプション）
            
        Returns:
            Tuple[bool, bool]: ディリクレ境界条件とノイマン境界条件の有効フラグ
        """
        if not hasattr(grid, 'is_3d') or not grid.is_3d:
            raise ValueError("3D方程式セットが非3Dグリッドで使用されました")
        
        # 変換器を作成
        converter = Equation2Dto3DConverter
        
        # 元の関数を全ての領域に追加
        system.add_dominant_equation(OriginalEquation3D(grid=grid))
        
        # 内部点用の方程式
        from equation_set.equation_set2d import DerivativeEquationSet2D
        derivative_2d = DerivativeEquationSet2D()
        
        # xy平面の方程式（z方向に拡張）
        system.add_equation('interior', converter.to_xy(derivative_2d, grid=grid))
        
        # xz平面の方程式（y方向に拡張）
        system.add_equation('interior', converter.to_xz(derivative_2d, grid=grid))
        
        # yz平面の方程式（x方向に拡張）
        system.add_equation('interior', converter.to_yz(derivative_2d, grid=grid))
        
        # 面、辺、頂点の方程式設定（各面に対応する平面方程式を追加）
        # 面
        system.add_equation('face_x_min', converter.to_yz(derivative_2d, grid=grid))
        system.add_equation('face_x_max', converter.to_yz(derivative_2d, grid=grid))
        system.add_equation('face_y_min', converter.to_xz(derivative_2d, grid=grid))
        system.add_equation('face_y_max', converter.to_xz(derivative_2d, grid=grid))
        system.add_equation('face_z_min', converter.to_xy(derivative_2d, grid=grid))
        system.add_equation('face_z_max', converter.to_xy(derivative_2d, grid=grid))
        
        # 辺（x方向）
        for edge in ['edge_x_y_min_z_min', 'edge_x_y_min_z_max', 
                    'edge_x_y_max_z_min', 'edge_x_y_max_z_max']:
            system.add_equation(edge, converter.to_x(derivative_2d, grid=grid))
        
        # 辺（y方向）
        for edge in ['edge_y_x_min_z_min', 'edge_y_x_min_z_max', 
                    'edge_y_x_max_z_min', 'edge_y_x_max_z_max']:
            system.add_equation(edge, converter.to_y(derivative_2d, grid=grid))
        
        # 辺（z方向）
        for edge in ['edge_z_x_min_y_min', 'edge_z_x_min_y_max', 
                    'edge_z_x_max_y_min', 'edge_z_x_max_y_max']:
            system.add_equation(edge, converter.to_z(derivative_2d, grid=grid))
        
        # 頂点
        for vertex in ['vertex_x_min_y_min_z_min', 'vertex_x_min_y_min_z_max',
                     'vertex_x_min_y_max_z_min', 'vertex_x_min_y_max_z_max',
                     'vertex_x_max_y_min_z_min', 'vertex_x_max_y_min_z_max',
                     'vertex_x_max_y_max_z_min', 'vertex_x_max_y_max_z_max']:
            system.add_equation(vertex, converter.to_xyz(derivative_2d, grid=grid))
        
        # 微分方程式セットでは境界条件は常に無効（既に導関数計算に制約式を使用）
        return False, False


class PoissonEquationSet3D(EquationSet):
    """ディリクレ・ノイマン混合境界条件の3Dポアソン方程式セット"""
    
    def __init__(self):
        super().__init__()
        self.enable_dirichlet = True
        self.enable_neumann = True
    
    def setup_equations(self, system, grid, test_func=None):
        """
        3次元ポアソン方程式システムを設定
        
        Args:
            system: 方程式システム
            grid: Grid オブジェクト (3D)
            test_func: テスト関数（オプション）
            
        Returns:
            Tuple[bool, bool]: ディリクレ境界条件とノイマン境界条件の有効フラグ
        """
        if not hasattr(grid, 'is_3d') or not grid.is_3d:
            raise ValueError("3D方程式セットが非3Dグリッドで使用されました")
            
        # 変換器を作成
        converter = Equation2Dto3DConverter

        # ポアソン方程式を全ての領域に追加
        system.add_dominant_equation(PoissonEquation3D(grid=grid))
        
        # 内部点用の方程式
        from equation_set.equation_set2d import DerivativeEquationSet2D
        derivative_2d = DerivativeEquationSet2D()
        
        # xy, xz, yz平面の方程式を内部点に追加
        system.add_equation('interior', converter.to_xyz(derivative_2d, grid=grid))
        
        # 面の方程式設定
        for face in ['face_x_min', 'face_x_max', 'face_y_min', 
                    'face_y_max', 'face_z_min', 'face_z_max']:
            system.add_equation(face, DirichletBoundaryEquation3D(grid=grid))
            
            # 対応する平面方程式を追加
            if face in ['face_x_min', 'face_x_max']:
                system.add_equation(face, converter.to_yz(derivative_2d, grid=grid))
            elif face in ['face_y_min', 'face_y_max']:
                system.add_equation(face, converter.to_xz(derivative_2d, grid=grid))
            else:  # face_z_min, face_z_max
                system.add_equation(face, converter.to_xy(derivative_2d, grid=grid))
        
        # 辺の方程式設定
        for edge in ['edge_x_y_min_z_min', 'edge_x_y_min_z_max', 
                   'edge_x_y_max_z_min', 'edge_x_y_max_z_max',
                   'edge_y_x_min_z_min', 'edge_y_x_min_z_max', 
                   'edge_y_x_max_z_min', 'edge_y_x_max_z_max',
                   'edge_z_x_min_y_min', 'edge_z_x_min_y_max', 
                   'edge_z_x_max_y_min', 'edge_z_x_max_y_max']:
            system.add_equation(edge, DirichletBoundaryEquation3D(grid=grid))
        
        # 頂点の方程式設定
        for vertex in ['vertex_x_min_y_min_z_min', 'vertex_x_min_y_min_z_max',
                     'vertex_x_min_y_max_z_min', 'vertex_x_min_y_max_z_max',
                     'vertex_x_max_y_min_z_min', 'vertex_x_max_y_min_z_max',
                     'vertex_x_max_y_max_z_min', 'vertex_x_max_y_max_z_max']:
            system.add_equation(vertex, DirichletBoundaryEquation3D(grid=grid))
        
        return True, True


class PoissonEquationSet3D2(EquationSet):
    """ディリクレ境界条件のみの3Dポアソン方程式セット"""
    
    def __init__(self):
        super().__init__()
        self.enable_dirichlet = True
        self.enable_neumann = False  # ノイマン境界条件を無効化
    
    def setup_equations(self, system, grid, test_func=None):
        """
        ディリクレ境界条件のみの3次元ポアソン方程式システムを設定
        
        Args:
            system: 方程式システム
            grid: Grid オブジェクト (3D)
            test_func: テスト関数（オプション）
            
        Returns:
            Tuple[bool, bool]: ディリクレ境界条件とノイマン境界条件の有効フラグ
        """
        if not hasattr(grid, 'is_3d') or not grid.is_3d:
            raise ValueError("3D方程式セットが非3Dグリッドで使用されました")
            
        # PoissonEquationSet3Dとほぼ同じだが、ノイマン境界条件を使わない
        # 変換器を作成
        converter = Equation2Dto3DConverter

        # ポアソン方程式を全ての領域に追加
        system.add_dominant_equation(PoissonEquation3D(grid=grid))
        
        # 内部点用の方程式
        from equation_set.equation_set2d import DerivativeEquationSet2D
        derivative_2d = DerivativeEquationSet2D()
        
        # xy, xz, yz平面の方程式を内部点に追加
        system.add_equation('interior', converter.to_xyz(derivative_2d, grid=grid))
        
        # 面の方程式設定（ディリクレのみ）
        for face in ['face_x_min', 'face_x_max', 'face_y_min', 
                    'face_y_max', 'face_z_min', 'face_z_max']:
            system.add_equation(face, DirichletBoundaryEquation3D(grid=grid))
            
            # 対応する平面方程式を追加
            if face in ['face_x_min', 'face_x_max']:
                system.add_equation(face, converter.to_yz(derivative_2d, grid=grid))
            elif face in ['face_y_min', 'face_y_max']:
                system.add_equation(face, converter.to_xz(derivative_2d, grid=grid))
            else:  # face_z_min, face_z_max
                system.add_equation(face, converter.to_xy(derivative_2d, grid=grid))
        
        # 辺と頂点に対しても同様（ディリクレのみ）
        # 辺の方程式設定
        for edge in ['edge_x_y_min_z_min', 'edge_x_y_min_z_max', 
                   'edge_x_y_max_z_min', 'edge_x_y_max_z_max',
                   'edge_y_x_min_z_min', 'edge_y_x_min_z_max', 
                   'edge_y_x_max_z_min', 'edge_y_x_max_z_max',
                   'edge_z_x_min_y_min', 'edge_z_x_min_y_max', 
                   'edge_z_x_max_y_min', 'edge_z_x_max_y_max']:
            system.add_equation(edge, DirichletBoundaryEquation3D(grid=grid))
        
        # 頂点の方程式設定
        for vertex in ['vertex_x_min_y_min_z_min', 'vertex_x_min_y_min_z_max',
                     'vertex_x_min_y_max_z_min', 'vertex_x_min_y_max_z_max',
                     'vertex_x_max_y_min_z_min', 'vertex_x_max_y_min_z_max',
                     'vertex_x_max_y_max_z_min', 'vertex_x_max_y_max_z_max']:
            system.add_equation(vertex, DirichletBoundaryEquation3D(grid=grid))
        
        return True, False  # ディリクレ境界条件のみ有効
