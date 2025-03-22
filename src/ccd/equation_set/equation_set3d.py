"""
3次元方程式セットの定義を行うモジュール

このモジュールでは、CCD（Combined Compact Difference）法に使用される
3次元の方程式セットを定義します。
"""

from core.base.base_equation_set import EquationSet

# 共通の方程式をインポート
from equation.poisson import PoissonEquation3D
from equation.original import OriginalEquation3D
from equation.boundary import DirichletBoundaryEquation3D, NeumannBoundaryEquation
from equation.boundary3d import NeumannBoundaryEquation3D
from equation.compact_internal import (
    Internal1stDerivativeEquation,
    Internal2ndDerivativeEquation,
    Internal3rdDerivativeEquation
)
from equation.compact_left_boundary import (
    LeftBoundary1stDerivativeEquation,
    LeftBoundary2ndDerivativeEquation,
    LeftBoundary3rdDerivativeEquation
)
from equation.compact_right_boundary import (
    RightBoundary1stDerivativeEquation,
    RightBoundary2ndDerivativeEquation,
    RightBoundary3rdDerivativeEquation
)
from equation.equation_converter import Equation1Dto3DConverter


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
        converter = Equation1Dto3DConverter
        
        # 元の関数を全ての領域に追加
        system.add_dominant_equation(OriginalEquation3D(grid=grid))
        
        # 内部点用の方程式
        system.add_equations('interior', [
            converter.to_x(Internal1stDerivativeEquation(), grid=grid),
            converter.to_x(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_x(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_y(Internal1stDerivativeEquation(), grid=grid),
            converter.to_y(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_y(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_z(Internal1stDerivativeEquation(), grid=grid),
            converter.to_z(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_z(Internal3rdDerivativeEquation(), grid=grid)
        ])
        
        # 面の方程式設定
        # x最小面 (i=0)
        system.add_equations('face_x_min', [
            converter.to_x(LeftBoundary1stDerivativeEquation(), grid=grid),
            converter.to_x(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_x(LeftBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_y(Internal1stDerivativeEquation(), grid=grid),
            converter.to_y(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_y(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_z(Internal1stDerivativeEquation(), grid=grid),
            converter.to_z(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_z(Internal3rdDerivativeEquation(), grid=grid)
        ])
        
        # x最大面 (i=nx-1)
        system.add_equations('face_x_max', [
            converter.to_x(RightBoundary1stDerivativeEquation(), grid=grid),
            converter.to_x(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_x(RightBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_y(Internal1stDerivativeEquation(), grid=grid),
            converter.to_y(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_y(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_z(Internal1stDerivativeEquation(), grid=grid),
            converter.to_z(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_z(Internal3rdDerivativeEquation(), grid=grid)
        ])
        
        # y最小面 (j=0)
        system.add_equations('face_y_min', [
            converter.to_x(Internal1stDerivativeEquation(), grid=grid),
            converter.to_x(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_x(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_y(LeftBoundary1stDerivativeEquation(), grid=grid),
            converter.to_y(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_y(LeftBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_z(Internal1stDerivativeEquation(), grid=grid),
            converter.to_z(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_z(Internal3rdDerivativeEquation(), grid=grid)
        ])
        
        # y最大面 (j=ny-1)
        system.add_equations('face_y_max', [
            converter.to_x(Internal1stDerivativeEquation(), grid=grid),
            converter.to_x(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_x(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_y(RightBoundary1stDerivativeEquation(), grid=grid),
            converter.to_y(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_y(RightBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_z(Internal1stDerivativeEquation(), grid=grid),
            converter.to_z(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_z(Internal3rdDerivativeEquation(), grid=grid)
        ])
        
        # z最小面 (k=0)
        system.add_equations('face_z_min', [
            converter.to_x(Internal1stDerivativeEquation(), grid=grid),
            converter.to_x(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_x(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_y(Internal1stDerivativeEquation(), grid=grid),
            converter.to_y(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_y(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_z(LeftBoundary1stDerivativeEquation(), grid=grid),
            converter.to_z(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_z(LeftBoundary3rdDerivativeEquation(), grid=grid)
        ])
        
        # z最大面 (k=nz-1)
        system.add_equations('face_z_max', [
            converter.to_x(Internal1stDerivativeEquation(), grid=grid),
            converter.to_x(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_x(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_y(Internal1stDerivativeEquation(), grid=grid),
            converter.to_y(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_y(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_z(RightBoundary1stDerivativeEquation(), grid=grid),
            converter.to_z(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_z(RightBoundary3rdDerivativeEquation(), grid=grid)
        ])

        # 辺の方程式設定（x方向辺）
        for edge in ['edge_x_y_min_z_min', 'edge_x_y_min_z_max', 
                     'edge_x_y_max_z_min', 'edge_x_y_max_z_max']:
            y_bound_type = "Left" if "y_min" in edge else "Right"
            z_bound_type = "Left" if "z_min" in edge else "Right"
            
            system.add_equations(edge, [
                converter.to_x(Internal1stDerivativeEquation(), grid=grid),
                converter.to_x(Internal2ndDerivativeEquation(), grid=grid),
                converter.to_x(Internal3rdDerivativeEquation(), grid=grid),
                converter.to_y(globals()[f"{y_bound_type}Boundary1stDerivativeEquation"](), grid=grid),
                converter.to_y(globals()[f"{y_bound_type}Boundary2ndDerivativeEquation"](), grid=grid),
                converter.to_y(globals()[f"{y_bound_type}Boundary3rdDerivativeEquation"](), grid=grid),
                converter.to_z(globals()[f"{z_bound_type}Boundary1stDerivativeEquation"](), grid=grid),
                converter.to_z(globals()[f"{z_bound_type}Boundary2ndDerivativeEquation"](), grid=grid),
                converter.to_z(globals()[f"{z_bound_type}Boundary3rdDerivativeEquation"](), grid=grid)
            ])
        
        # 辺の方程式設定（y方向辺）
        for edge in ['edge_y_x_min_z_min', 'edge_y_x_min_z_max', 
                     'edge_y_x_max_z_min', 'edge_y_x_max_z_max']:
            x_bound_type = "Left" if "x_min" in edge else "Right"
            z_bound_type = "Left" if "z_min" in edge else "Right"
            
            system.add_equations(edge, [
                converter.to_x(globals()[f"{x_bound_type}Boundary1stDerivativeEquation"](), grid=grid),
                converter.to_x(globals()[f"{x_bound_type}Boundary2ndDerivativeEquation"](), grid=grid),
                converter.to_x(globals()[f"{x_bound_type}Boundary3rdDerivativeEquation"](), grid=grid),
                converter.to_y(Internal1stDerivativeEquation(), grid=grid),
                converter.to_y(Internal2ndDerivativeEquation(), grid=grid),
                converter.to_y(Internal3rdDerivativeEquation(), grid=grid),
                converter.to_z(globals()[f"{z_bound_type}Boundary1stDerivativeEquation"](), grid=grid),
                converter.to_z(globals()[f"{z_bound_type}Boundary2ndDerivativeEquation"](), grid=grid),
                converter.to_z(globals()[f"{z_bound_type}Boundary3rdDerivativeEquation"](), grid=grid)
            ])
        
        # 辺の方程式設定（z方向辺）
        for edge in ['edge_z_x_min_y_min', 'edge_z_x_min_y_max', 
                     'edge_z_x_max_y_min', 'edge_z_x_max_y_max']:
            x_bound_type = "Left" if "x_min" in edge else "Right"
            y_bound_type = "Left" if "y_min" in edge else "Right"
            
            system.add_equations(edge, [
                converter.to_x(globals()[f"{x_bound_type}Boundary1stDerivativeEquation"](), grid=grid),
                converter.to_x(globals()[f"{x_bound_type}Boundary2ndDerivativeEquation"](), grid=grid),
                converter.to_x(globals()[f"{x_bound_type}Boundary3rdDerivativeEquation"](), grid=grid),
                converter.to_y(globals()[f"{y_bound_type}Boundary1stDerivativeEquation"](), grid=grid),
                converter.to_y(globals()[f"{y_bound_type}Boundary2ndDerivativeEquation"](), grid=grid),
                converter.to_y(globals()[f"{y_bound_type}Boundary3rdDerivativeEquation"](), grid=grid),
                converter.to_z(Internal1stDerivativeEquation(), grid=grid),
                converter.to_z(Internal2ndDerivativeEquation(), grid=grid),
                converter.to_z(Internal3rdDerivativeEquation(), grid=grid)
            ])
        
        # 頂点の方程式設定
        for vertex in ['vertex_x_min_y_min_z_min', 'vertex_x_min_y_min_z_max',
                       'vertex_x_min_y_max_z_min', 'vertex_x_min_y_max_z_max',
                       'vertex_x_max_y_min_z_min', 'vertex_x_max_y_min_z_max',
                       'vertex_x_max_y_max_z_min', 'vertex_x_max_y_max_z_max']:
            x_bound_type = "Left" if "x_min" in vertex else "Right"
            y_bound_type = "Left" if "y_min" in vertex else "Right"
            z_bound_type = "Left" if "z_min" in vertex else "Right"
            
            system.add_equations(vertex, [
                converter.to_x(globals()[f"{x_bound_type}Boundary1stDerivativeEquation"](), grid=grid),
                converter.to_x(globals()[f"{x_bound_type}Boundary2ndDerivativeEquation"](), grid=grid),
                converter.to_x(globals()[f"{x_bound_type}Boundary3rdDerivativeEquation"](), grid=grid),
                converter.to_y(globals()[f"{y_bound_type}Boundary1stDerivativeEquation"](), grid=grid),
                converter.to_y(globals()[f"{y_bound_type}Boundary2ndDerivativeEquation"](), grid=grid),
                converter.to_y(globals()[f"{y_bound_type}Boundary3rdDerivativeEquation"](), grid=grid),
                converter.to_z(globals()[f"{z_bound_type}Boundary1stDerivativeEquation"](), grid=grid),
                converter.to_z(globals()[f"{z_bound_type}Boundary2ndDerivativeEquation"](), grid=grid),
                converter.to_z(globals()[f"{z_bound_type}Boundary3rdDerivativeEquation"](), grid=grid)
            ])
        
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
        converter = Equation1Dto3DConverter

        # ポアソン方程式を全ての領域に追加
        system.add_dominant_equation(PoissonEquation3D(grid=grid))
        
        # 内部点用の基本方程式
        internal_equations = [
            converter.to_x(Internal1stDerivativeEquation(), grid=grid),
            converter.to_x(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_x(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_y(Internal1stDerivativeEquation(), grid=grid),
            converter.to_y(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_y(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_z(Internal1stDerivativeEquation(), grid=grid),
            converter.to_z(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_z(Internal3rdDerivativeEquation(), grid=grid)
        ]
        
        system.add_equations('interior', internal_equations)
        
        # 面の方程式設定
        # x最小面 (i=0)
        system.add_equations('face_x_min', [
            DirichletBoundaryEquation3D(grid=grid),
            NeumannBoundaryEquation3D('x', grid=grid),
            converter.to_x(LeftBoundary1stDerivativeEquation(), grid=grid),
            converter.to_x(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_x(LeftBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_y(Internal1stDerivativeEquation(), grid=grid),
            converter.to_y(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_y(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_z(Internal1stDerivativeEquation(), grid=grid),
            converter.to_z(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_z(Internal3rdDerivativeEquation(), grid=grid)
        ])
        
        # x最大面 (i=nx-1)
        system.add_equations('face_x_max', [
            DirichletBoundaryEquation3D(grid=grid),
            NeumannBoundaryEquation3D('x', grid=grid),
            converter.to_x(RightBoundary1stDerivativeEquation(), grid=grid),
            converter.to_x(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_x(RightBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_y(Internal1stDerivativeEquation(), grid=grid),
            converter.to_y(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_y(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_z(Internal1stDerivativeEquation(), grid=grid),
            converter.to_z(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_z(Internal3rdDerivativeEquation(), grid=grid)
        ])
        
        # y最小面 (j=0)
        system.add_equations('face_y_min', [
            DirichletBoundaryEquation3D(grid=grid),
            NeumannBoundaryEquation3D('y', grid=grid),
            converter.to_x(Internal1stDerivativeEquation(), grid=grid),
            converter.to_x(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_x(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_y(LeftBoundary1stDerivativeEquation(), grid=grid),
            converter.to_y(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_y(LeftBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_z(Internal1stDerivativeEquation(), grid=grid),
            converter.to_z(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_z(Internal3rdDerivativeEquation(), grid=grid)
        ])
        
        # y最大面 (j=ny-1)
        system.add_equations('face_y_max', [
            DirichletBoundaryEquation3D(grid=grid),
            NeumannBoundaryEquation3D('y', grid=grid),
            converter.to_x(Internal1stDerivativeEquation(), grid=grid),
            converter.to_x(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_x(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_y(RightBoundary1stDerivativeEquation(), grid=grid),
            converter.to_y(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_y(RightBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_z(Internal1stDerivativeEquation(), grid=grid),
            converter.to_z(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_z(Internal3rdDerivativeEquation(), grid=grid)
        ])
        
        # z最小面 (k=0)
        system.add_equations('face_z_min', [
            DirichletBoundaryEquation3D(grid=grid),
            NeumannBoundaryEquation3D('z', grid=grid),
            converter.to_x(Internal1stDerivativeEquation(), grid=grid),
            converter.to_x(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_x(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_y(Internal1stDerivativeEquation(), grid=grid),
            converter.to_y(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_y(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_z(LeftBoundary1stDerivativeEquation(), grid=grid),
            converter.to_z(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_z(LeftBoundary3rdDerivativeEquation(), grid=grid)
        ])
        
        # z最大面 (k=nz-1)
        system.add_equations('face_z_max', [
            DirichletBoundaryEquation3D(grid=grid),
            NeumannBoundaryEquation3D('z', grid=grid),
            converter.to_x(Internal1stDerivativeEquation(), grid=grid),
            converter.to_x(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_x(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_y(Internal1stDerivativeEquation(), grid=grid),
            converter.to_y(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_y(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_z(RightBoundary1stDerivativeEquation(), grid=grid),
            converter.to_z(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_z(RightBoundary3rdDerivativeEquation(), grid=grid)
        ])

        # 辺と頂点の方程式設定
        # 辺の方程式に境界条件を追加
        for region in system.equations:
            if region.startswith('edge_') or region.startswith('vertex_'):
                system.add_equation(region, DirichletBoundaryEquation3D(grid=grid))
                
                # 適切な方向のノイマン境界条件を追加
                if 'x_min' in region or 'x_max' in region:
                    system.add_equation(region, NeumannBoundaryEquation3D('x', grid=grid))
                if 'y_min' in region or 'y_max' in region:
                    system.add_equation(region, NeumannBoundaryEquation3D('y', grid=grid))
                if 'z_min' in region or 'z_max' in region:
                    system.add_equation(region, NeumannBoundaryEquation3D('z', grid=grid))
        
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
            
        # 変換器を作成
        converter = Equation1Dto3DConverter

        # ポアソン方程式を全ての領域に追加
        system.add_dominant_equation(PoissonEquation3D(grid=grid))
        
        # 内部点用の基本方程式
        internal_equations = [
            converter.to_x(Internal1stDerivativeEquation(), grid=grid),
            converter.to_x(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_x(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_y(Internal1stDerivativeEquation(), grid=grid),
            converter.to_y(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_y(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_z(Internal1stDerivativeEquation(), grid=grid),
            converter.to_z(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_z(Internal3rdDerivativeEquation(), grid=grid)
        ]
        
        system.add_equations('interior', internal_equations)
        
        # 面の方程式設定
        # x最小面 (i=0)
        system.add_equations('face_x_min', [
            DirichletBoundaryEquation3D(grid=grid),
            converter.to_x(LeftBoundary1stDerivativeEquation(), grid=grid),
            converter.to_x(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_x(LeftBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_y(Internal1stDerivativeEquation(), grid=grid),
            converter.to_y(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_y(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_z(Internal1stDerivativeEquation(), grid=grid),
            converter.to_z(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_z(Internal3rdDerivativeEquation(), grid=grid)
        ])
        
        # x最大面 (i=nx-1)
        system.add_equations('face_x_max', [
            DirichletBoundaryEquation3D(grid=grid),
            converter.to_x(RightBoundary1stDerivativeEquation(), grid=grid),
            converter.to_x(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_x(RightBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_y(Internal1stDerivativeEquation(), grid=grid),
            converter.to_y(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_y(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_z(Internal1stDerivativeEquation(), grid=grid),
            converter.to_z(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_z(Internal3rdDerivativeEquation(), grid=grid)
        ])
        
        # y最小面 (j=0)
        system.add_equations('face_y_min', [
            DirichletBoundaryEquation3D(grid=grid),
            converter.to_x(Internal1stDerivativeEquation(), grid=grid),
            converter.to_x(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_x(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_y(LeftBoundary1stDerivativeEquation(), grid=grid),
            converter.to_y(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_y(LeftBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_z(Internal1stDerivativeEquation(), grid=grid),
            converter.to_z(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_z(Internal3rdDerivativeEquation(), grid=grid)
        ])
        
        # y最大面 (j=ny-1)
        system.add_equations('face_y_max', [
            DirichletBoundaryEquation3D(grid=grid),
            converter.to_x(Internal1stDerivativeEquation(), grid=grid),
            converter.to_x(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_x(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_y(RightBoundary1stDerivativeEquation(), grid=grid),
            converter.to_y(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_y(RightBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_z(Internal1stDerivativeEquation(), grid=grid),
            converter.to_z(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_z(Internal3rdDerivativeEquation(), grid=grid)
        ])

        # z最小面 (k=0)
        system.add_equations('face_z_min', [
            DirichletBoundaryEquation3D(grid=grid),
            converter.to_x(Internal1stDerivativeEquation(), grid=grid),
            converter.to_x(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_x(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_y(Internal1stDerivativeEquation(), grid=grid),
            converter.to_y(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_y(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_z(LeftBoundary1stDerivativeEquation(), grid=grid),
            converter.to_z(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_z(LeftBoundary3rdDerivativeEquation(), grid=grid)
        ])
        
        # z最大面 (k=nz-1)
        system.add_equations('face_z_max', [
            DirichletBoundaryEquation3D(grid=grid),
            converter.to_x(Internal1stDerivativeEquation(), grid=grid),
            converter.to_x(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_x(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_y(Internal1stDerivativeEquation(), grid=grid),
            converter.to_y(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_y(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_z(RightBoundary1stDerivativeEquation(), grid=grid),
            converter.to_z(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_z(RightBoundary3rdDerivativeEquation(), grid=grid)
        ])

        # 辺と頂点の方程式設定
        # 全ての辺と頂点にディリクレ境界条件を追加
        for region in system.equations:
            if region.startswith('edge_') or region.startswith('vertex_'):
                system.add_equation(region, DirichletBoundaryEquation3D(grid=grid))
                
        return True, False  # ディリクレト境界条件のみ有効