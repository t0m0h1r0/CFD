"""
3次元ポアソン方程式セットの定義を行うモジュール
"""

from core.base.base_equation_set import EquationSet
from equation.dim3.poisson import PoissonEquation3D
from equation.dim3.boundary import DirichletBoundaryEquation3D
from equation.dim1.boundary import NeumannBoundaryEquation
from equation.dim1.compact_internal import (
    Internal1stDerivativeEquation,
    Internal2ndDerivativeEquation,
    Internal3rdDerivativeEquation
)
from equation.dim1.compact_left_boundary import (
    LeftBoundary1stDerivativeEquation,
    LeftBoundary2ndDerivativeEquation,
    LeftBoundary3rdDerivativeEquation
)
from equation.dim1.compact_right_boundary import (
    RightBoundary1stDerivativeEquation,
    RightBoundary2ndDerivativeEquation,
    RightBoundary3rdDerivativeEquation
)
from equation.converter import Equation1Dto3DConverter


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
        
        # 各面用の方程式 (x, y, z方向)
        
        # x = 0 面 (左面)
        x_min_face_eqs = [
            DirichletBoundaryEquation3D(grid=grid),
            converter.to_x(NeumannBoundaryEquation(), grid=grid),  # x方向のノイマン条件
            converter.to_x(
                LeftBoundary1stDerivativeEquation(),
                #LeftBoundary2ndDerivativeEquation()+ 
                #LeftBoundary3rdDerivativeEquation(), 
                grid=grid),
            converter.to_y(Internal1stDerivativeEquation(), grid=grid),
            converter.to_y(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_y(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_z(Internal1stDerivativeEquation(), grid=grid),
            converter.to_z(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_z(Internal3rdDerivativeEquation(), grid=grid)
        ]
        
        system.add_equations('face_x_min', x_min_face_eqs)
        
        # x = nx-1 面 (右面)
        x_max_face_eqs = [
            DirichletBoundaryEquation3D(grid=grid),
            converter.to_x(NeumannBoundaryEquation(), grid=grid),  # x方向のノイマン条件
            converter.to_x(
                RightBoundary1stDerivativeEquation(),
                #RightBoundary2ndDerivativeEquation()+ 
                #RightBoundary3rdDerivativeEquation(), 
                grid=grid),
            converter.to_y(Internal1stDerivativeEquation(), grid=grid),
            converter.to_y(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_y(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_z(Internal1stDerivativeEquation(), grid=grid),
            converter.to_z(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_z(Internal3rdDerivativeEquation(), grid=grid)
        ]
        
        system.add_equations('face_x_max', x_max_face_eqs)
        
        # y = 0 面 (下面)
        y_min_face_eqs = [
            converter.to_x(Internal1stDerivativeEquation(), grid=grid),
            converter.to_x(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_x(Internal3rdDerivativeEquation(), grid=grid),
            DirichletBoundaryEquation3D(grid=grid),
            converter.to_y(NeumannBoundaryEquation(), grid=grid),  # y方向のノイマン条件
            converter.to_y(
                LeftBoundary1stDerivativeEquation(),
                #LeftBoundary2ndDerivativeEquation()+ 
                #LeftBoundary3rdDerivativeEquation(), 
                grid=grid),
            converter.to_z(Internal1stDerivativeEquation(), grid=grid),
            converter.to_z(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_z(Internal3rdDerivativeEquation(), grid=grid)
        ]
        
        system.add_equations('face_y_min', y_min_face_eqs)
        
        # y = ny-1 面 (上面)
        y_max_face_eqs = [
            converter.to_x(Internal1stDerivativeEquation(), grid=grid),
            converter.to_x(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_x(Internal3rdDerivativeEquation(), grid=grid),
            DirichletBoundaryEquation3D(grid=grid),
            converter.to_y(NeumannBoundaryEquation(), grid=grid),  # y方向のノイマン条件
            converter.to_y(
                RightBoundary1stDerivativeEquation(),
                #RightBoundary2ndDerivativeEquation()+ 
                #RightBoundary3rdDerivativeEquation(), 
                grid=grid),
            converter.to_z(Internal1stDerivativeEquation(), grid=grid),
            converter.to_z(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_z(Internal3rdDerivativeEquation(), grid=grid)
        ]
        
        system.add_equations('face_y_max', y_max_face_eqs)
        
        # z = 0 面 (前面)
        z_min_face_eqs = [
            converter.to_x(Internal1stDerivativeEquation(), grid=grid),
            converter.to_x(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_x(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_y(Internal1stDerivativeEquation(), grid=grid),
            converter.to_y(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_y(Internal3rdDerivativeEquation(), grid=grid),
            DirichletBoundaryEquation3D(grid=grid),
            converter.to_z(NeumannBoundaryEquation(), grid=grid),  # z方向のノイマン条件
            converter.to_z(
                LeftBoundary1stDerivativeEquation(),
                #LeftBoundary2ndDerivativeEquation()+ 
                #LeftBoundary3rdDerivativeEquation(), 
                grid=grid)
        ]
        
        system.add_equations('face_z_min', z_min_face_eqs)
        
        # z = nz-1 面 (後面)
        z_max_face_eqs = [
            converter.to_x(Internal1stDerivativeEquation(), grid=grid),
            converter.to_x(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_x(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_y(Internal1stDerivativeEquation(), grid=grid),
            converter.to_y(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_y(Internal3rdDerivativeEquation(), grid=grid),
            DirichletBoundaryEquation3D(grid=grid),
            converter.to_z(NeumannBoundaryEquation(), grid=grid),  # z方向のノイマン条件
            converter.to_z(
                RightBoundary1stDerivativeEquation(),
                #RightBoundary2ndDerivativeEquation()+ 
                #RightBoundary3rdDerivativeEquation(), 
                grid=grid)
        ]
        
        system.add_equations('face_z_max', z_max_face_eqs)
        
        # ===== エッジの方程式設定 =====
        # x方向エッジ (y = 0, z = 0)
        edge_x_y_min_z_min_eqs = [
            converter.to_x(Internal1stDerivativeEquation(), grid=grid),
            converter.to_x(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_x(Internal3rdDerivativeEquation(), grid=grid),
            DirichletBoundaryEquation3D(grid=grid),
            converter.to_y(NeumannBoundaryEquation(), grid=grid),
            converter.to_y(
                LeftBoundary1stDerivativeEquation(),
                #LeftBoundary2ndDerivativeEquation()+ 
                #LeftBoundary3rdDerivativeEquation(), 
                grid=grid),
            converter.to_z(NeumannBoundaryEquation(), grid=grid),
            converter.to_z(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_z(
                LeftBoundary1stDerivativeEquation(),
                #LeftBoundary3rdDerivativeEquation(), 
                grid=grid)
        ]
        
        system.add_equations('edge_x_y_min_z_min', edge_x_y_min_z_min_eqs)
        
        # x方向エッジ (y = 0, z = nz-1)
        edge_x_y_min_z_max_eqs = [
            converter.to_x(Internal1stDerivativeEquation(), grid=grid),
            converter.to_x(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_x(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_y(NeumannBoundaryEquation(), grid=grid),
            converter.to_y(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_y(
                LeftBoundary1stDerivativeEquation(),
                #LeftBoundary3rdDerivativeEquation(), 
                grid=grid),
            DirichletBoundaryEquation3D(grid=grid),
            converter.to_z(NeumannBoundaryEquation(), grid=grid),
            converter.to_z(
                RightBoundary1stDerivativeEquation(),
                #RightBoundary2ndDerivativeEquation()+ 
                #RightBoundary3rdDerivativeEquation(), 
                grid=grid)
        ]
        
        system.add_equations('edge_x_y_min_z_max', edge_x_y_min_z_max_eqs)
        
        # x方向エッジ (y = ny-1, z = 0)
        edge_x_y_max_z_min_eqs = [
            converter.to_x(Internal1stDerivativeEquation(), grid=grid),
            converter.to_x(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_x(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_y(NeumannBoundaryEquation(), grid=grid),
            converter.to_y(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_y(
                RightBoundary1stDerivativeEquation(),
                #RightBoundary3rdDerivativeEquation(), 
                grid=grid),
            DirichletBoundaryEquation3D(grid=grid),
            converter.to_z(NeumannBoundaryEquation(), grid=grid),
            converter.to_z(
                LeftBoundary1stDerivativeEquation(),
                #LeftBoundary2ndDerivativeEquation()+ 
                #LeftBoundary3rdDerivativeEquation(), 
                grid=grid)
        ]
        
        system.add_equations('edge_x_y_max_z_min', edge_x_y_max_z_min_eqs)
        
        # x方向エッジ (y = ny-1, z = nz-1)
        edge_x_y_max_z_max_eqs = [
            converter.to_x(Internal1stDerivativeEquation(), grid=grid),
            converter.to_x(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_x(Internal3rdDerivativeEquation(), grid=grid),
            DirichletBoundaryEquation3D(grid=grid),
            converter.to_y(NeumannBoundaryEquation(), grid=grid),
            converter.to_y(
                RightBoundary1stDerivativeEquation(),
                #RightBoundary2ndDerivativeEquation()+ 
                #RightBoundary3rdDerivativeEquation(), 
                grid=grid),
            converter.to_z(NeumannBoundaryEquation(), grid=grid),
            converter.to_z(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_z(
                RightBoundary1stDerivativeEquation(),
                #RightBoundary3rdDerivativeEquation(), 
                grid=grid)
        ]
        
        system.add_equations('edge_x_y_max_z_max', edge_x_y_max_z_max_eqs)
        
        # y方向エッジ (x = 0, z = 0)
        edge_y_x_min_z_min_eqs = [
            DirichletBoundaryEquation3D(grid=grid),
            converter.to_x(NeumannBoundaryEquation(), grid=grid),
            converter.to_x(
                LeftBoundary1stDerivativeEquation(),
                #LeftBoundary2ndDerivativeEquation()+ 
                #LeftBoundary3rdDerivativeEquation(), 
                grid=grid),
            converter.to_y(Internal1stDerivativeEquation(), grid=grid),
            converter.to_y(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_y(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_z(NeumannBoundaryEquation(), grid=grid),
            converter.to_z(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_z(
                LeftBoundary1stDerivativeEquation(),
                #LeftBoundary3rdDerivativeEquation(), 
                grid=grid)
        ]
        
        system.add_equations('edge_y_x_min_z_min', edge_y_x_min_z_min_eqs)
        
        # y方向エッジ (x = 0, z = nz-1)
        edge_y_x_min_z_max_eqs = [
            converter.to_x(NeumannBoundaryEquation(), grid=grid),
            converter.to_x(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_x(
                LeftBoundary1stDerivativeEquation(),
                #LeftBoundary3rdDerivativeEquation(), 
                grid=grid),
            converter.to_y(Internal1stDerivativeEquation(), grid=grid),
            converter.to_y(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_y(Internal3rdDerivativeEquation(), grid=grid),
            DirichletBoundaryEquation3D(grid=grid),
            converter.to_z(NeumannBoundaryEquation(), grid=grid),
            converter.to_z(
                RightBoundary1stDerivativeEquation(),
                #RightBoundary2ndDerivativeEquation()+ 
                #RightBoundary3rdDerivativeEquation(), 
                grid=grid)
        ]
        
        system.add_equations('edge_y_x_min_z_max', edge_y_x_min_z_max_eqs)
        
        # y方向エッジ (x = nx-1, z = 0)
        edge_y_x_max_z_min_eqs = [
            converter.to_x(NeumannBoundaryEquation(), grid=grid),
            converter.to_x(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_x(
                RightBoundary1stDerivativeEquation(),
                #RightBoundary3rdDerivativeEquation(), 
                grid=grid),
            converter.to_y(Internal1stDerivativeEquation(), grid=grid),
            converter.to_y(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_y(Internal3rdDerivativeEquation(), grid=grid),
            DirichletBoundaryEquation3D(grid=grid),
            converter.to_z(NeumannBoundaryEquation(), grid=grid),
            converter.to_z(
                LeftBoundary1stDerivativeEquation(),
                #LeftBoundary2ndDerivativeEquation()+ 
                #LeftBoundary3rdDerivativeEquation(), 
                grid=grid)
        ]
        
        system.add_equations('edge_y_x_max_z_min', edge_y_x_max_z_min_eqs)
        
        # y方向エッジ (x = nx-1, z = nz-1)
        edge_y_x_max_z_max_eqs = [
            DirichletBoundaryEquation3D(grid=grid),
            converter.to_x(NeumannBoundaryEquation(), grid=grid),
            converter.to_x(
                RightBoundary1stDerivativeEquation(),
                #RightBoundary2ndDerivativeEquation()+ 
                #RightBoundary3rdDerivativeEquation(), 
                grid=grid),
            converter.to_y(Internal1stDerivativeEquation(), grid=grid),
            converter.to_y(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_y(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_z(NeumannBoundaryEquation(), grid=grid),
            converter.to_z(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_z(
                RightBoundary1stDerivativeEquation(),
                #RightBoundary3rdDerivativeEquation(), 
                grid=grid)
        ]
        
        system.add_equations('edge_y_x_max_z_max', edge_y_x_max_z_max_eqs)
        
        # z方向エッジ (x = 0, y = 0)
        edge_z_x_min_y_min_eqs = [
            DirichletBoundaryEquation3D(grid=grid),
            converter.to_x(NeumannBoundaryEquation(), grid=grid),
            converter.to_x(
                LeftBoundary1stDerivativeEquation(),
                #LeftBoundary2ndDerivativeEquation()+ 
                #LeftBoundary3rdDerivativeEquation(), 
                grid=grid),
            converter.to_y(NeumannBoundaryEquation(), grid=grid),
            converter.to_y(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_y(
                LeftBoundary1stDerivativeEquation(),
                #LeftBoundary3rdDerivativeEquation(), 
                grid=grid),
            converter.to_z(Internal1stDerivativeEquation(), grid=grid),
            converter.to_z(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_z(Internal3rdDerivativeEquation(), grid=grid)
        ]
        
        system.add_equations('edge_z_x_min_y_min', edge_z_x_min_y_min_eqs)
        
        # z方向エッジ (x = 0, y = ny-1)
        edge_z_x_min_y_max_eqs = [
            converter.to_x(NeumannBoundaryEquation(), grid=grid),
            converter.to_x(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_x(
                LeftBoundary1stDerivativeEquation(),
                #LeftBoundary3rdDerivativeEquation(), 
                grid=grid),
            DirichletBoundaryEquation3D(grid=grid),
            converter.to_y(NeumannBoundaryEquation(), grid=grid),
            converter.to_y(
                RightBoundary1stDerivativeEquation(),
                #RightBoundary2ndDerivativeEquation()+ 
                #RightBoundary3rdDerivativeEquation(), 
                grid=grid),
            converter.to_z(Internal1stDerivativeEquation(), grid=grid),
            converter.to_z(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_z(Internal3rdDerivativeEquation(), grid=grid)
        ]
        
        system.add_equations('edge_z_x_min_y_max', edge_z_x_min_y_max_eqs)
        
        # z方向エッジ (x = nx-1, y = 0)
        edge_z_x_max_y_min_eqs = [
            converter.to_x(NeumannBoundaryEquation(), grid=grid),
            converter.to_x(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_x(
                RightBoundary1stDerivativeEquation(),
                #RightBoundary3rdDerivativeEquation(), 
                grid=grid),
            DirichletBoundaryEquation3D(grid=grid),
            converter.to_y(NeumannBoundaryEquation(), grid=grid),
            converter.to_y(
                LeftBoundary1stDerivativeEquation(),
                #LeftBoundary2ndDerivativeEquation()+ 
                #LeftBoundary3rdDerivativeEquation(), 
                grid=grid),
            converter.to_z(Internal1stDerivativeEquation(), grid=grid),
            converter.to_z(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_z(Internal3rdDerivativeEquation(), grid=grid)
        ]
        
        system.add_equations('edge_z_x_max_y_min', edge_z_x_max_y_min_eqs)
        
        # z方向エッジ (x = nx-1, y = ny-1)
        edge_z_x_max_y_max_eqs = [
            DirichletBoundaryEquation3D(grid=grid),
            converter.to_x(NeumannBoundaryEquation(), grid=grid),
            converter.to_x(
                RightBoundary1stDerivativeEquation(),
                #RightBoundary2ndDerivativeEquation()+ 
                #RightBoundary3rdDerivativeEquation(), 
                grid=grid),
            converter.to_y(NeumannBoundaryEquation(), grid=grid),
            converter.to_y(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_y(
                RightBoundary1stDerivativeEquation(),
                #RightBoundary3rdDerivativeEquation(), 
                grid=grid),
            converter.to_z(Internal1stDerivativeEquation(), grid=grid),
            converter.to_z(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_z(Internal3rdDerivativeEquation(), grid=grid)
        ]
        
        system.add_equations('edge_z_x_max_y_max', edge_z_x_max_y_max_eqs)
        
        # ===== 頂点の方程式設定 =====
        # 頂点 (x = 0, y = 0, z = 0)
        vertex_x_min_y_min_z_min_eqs = [
            DirichletBoundaryEquation3D(grid=grid),
            converter.to_x(NeumannBoundaryEquation(), grid=grid),
            converter.to_x(
                LeftBoundary1stDerivativeEquation(),
                #LeftBoundary2ndDerivativeEquation()+ 
                #LeftBoundary3rdDerivativeEquation(), 
                grid=grid),
            converter.to_y(NeumannBoundaryEquation(), grid=grid),
            converter.to_y(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_y(
                LeftBoundary1stDerivativeEquation(),
                #LeftBoundary3rdDerivativeEquation(), 
                grid=grid),
            converter.to_z(NeumannBoundaryEquation(), grid=grid),
            converter.to_z(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_z(
                LeftBoundary1stDerivativeEquation(),
                #LeftBoundary3rdDerivativeEquation(), 
                grid=grid)
        ]
        
        system.add_equations('vertex_x_min_y_min_z_min', vertex_x_min_y_min_z_min_eqs)
        
        # 頂点 (x = 0, y = 0, z = nz-1)
        vertex_x_min_y_min_z_max_eqs = [
            converter.to_x(NeumannBoundaryEquation(), grid=grid),
            converter.to_x(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_x(
                LeftBoundary1stDerivativeEquation(),
                #LeftBoundary3rdDerivativeEquation(), 
                grid=grid),
            DirichletBoundaryEquation3D(grid=grid),
            converter.to_y(NeumannBoundaryEquation(), grid=grid),
            converter.to_y(
                LeftBoundary1stDerivativeEquation(),
                #LeftBoundary2ndDerivativeEquation()+ 
                #LeftBoundary3rdDerivativeEquation(), 
                grid=grid),
            converter.to_z(NeumannBoundaryEquation(), grid=grid),
            converter.to_z(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_z(
                RightBoundary1stDerivativeEquation(),
                #RightBoundary3rdDerivativeEquation(), 
                grid=grid)
        ]
        
        system.add_equations('vertex_x_min_y_min_z_max', vertex_x_min_y_min_z_max_eqs)
        
        # 頂点 (x = 0, y = ny-1, z = 0)
        vertex_x_min_y_max_z_min_eqs = [
            converter.to_x(NeumannBoundaryEquation(), grid=grid),
            converter.to_x(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_x(
                LeftBoundary1stDerivativeEquation(),
                #LeftBoundary3rdDerivativeEquation(), 
                grid=grid),
            converter.to_y(NeumannBoundaryEquation(), grid=grid),
            converter.to_y(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_y(
                RightBoundary1stDerivativeEquation(),
                #RightBoundary3rdDerivativeEquation(), 
                grid=grid),
            DirichletBoundaryEquation3D(grid=grid),
            converter.to_z(NeumannBoundaryEquation(), grid=grid),
            converter.to_z(
                LeftBoundary1stDerivativeEquation(),
                #LeftBoundary2ndDerivativeEquation()+ 
                #LeftBoundary3rdDerivativeEquation(), 
                grid=grid)
        ]
        
        system.add_equations('vertex_x_min_y_max_z_min', vertex_x_min_y_max_z_min_eqs)
        
        # 頂点 (x = 0, y = ny-1, z = nz-1)
        vertex_x_min_y_max_z_max_eqs = [
            DirichletBoundaryEquation3D(grid=grid),
            converter.to_x(NeumannBoundaryEquation(), grid=grid),
            converter.to_x(
                LeftBoundary1stDerivativeEquation(),
                #LeftBoundary2ndDerivativeEquation()+ 
                #LeftBoundary3rdDerivativeEquation(), 
                grid=grid),
            converter.to_y(NeumannBoundaryEquation(), grid=grid),
            converter.to_y(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_y(
                RightBoundary1stDerivativeEquation(),
                #RightBoundary3rdDerivativeEquation(), 
                grid=grid),
            converter.to_z(NeumannBoundaryEquation(), grid=grid),
            converter.to_z(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_z(
                RightBoundary1stDerivativeEquation(),
                #RightBoundary3rdDerivativeEquation(), 
                grid=grid)
        ]
        
        system.add_equations('vertex_x_min_y_max_z_max', vertex_x_min_y_max_z_max_eqs)
        
        # 頂点 (x = nx-1, y = 0, z = 0)
        vertex_x_max_y_min_z_min_eqs = [
            converter.to_x(NeumannBoundaryEquation(), grid=grid),
            converter.to_x(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_x(
                RightBoundary1stDerivativeEquation(),
                #RightBoundary3rdDerivativeEquation(), 
                grid=grid),
            DirichletBoundaryEquation3D(grid=grid),
            converter.to_y(NeumannBoundaryEquation(), grid=grid),
            converter.to_y(
                LeftBoundary1stDerivativeEquation(),
                #LeftBoundary2ndDerivativeEquation()+ 
                #LeftBoundary3rdDerivativeEquation(), 
                grid=grid),
            converter.to_z(NeumannBoundaryEquation(), grid=grid),
            converter.to_z(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_z(
                LeftBoundary1stDerivativeEquation(),
                #LeftBoundary3rdDerivativeEquation(), 
                grid=grid)
        ]
        
        system.add_equations('vertex_x_max_y_min_z_min', vertex_x_max_y_min_z_min_eqs)
        
        # 頂点 (x = nx-1, y = 0, z = nz-1)
        vertex_x_max_y_min_z_max_eqs = [
            converter.to_x(NeumannBoundaryEquation(), grid=grid),
            converter.to_x(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_x(
                RightBoundary1stDerivativeEquation(),
                #RightBoundary3rdDerivativeEquation(), 
                grid=grid),
            converter.to_y(NeumannBoundaryEquation(), grid=grid),
            converter.to_y(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_y(
                LeftBoundary1stDerivativeEquation(),
                #LeftBoundary3rdDerivativeEquation(), 
                grid=grid),
            DirichletBoundaryEquation3D(grid=grid),
            converter.to_z(NeumannBoundaryEquation(), grid=grid),
            converter.to_z(
                RightBoundary1stDerivativeEquation(),
                #RightBoundary2ndDerivativeEquation()+ 
                #RightBoundary3rdDerivativeEquation(), 
                grid=grid)
        ]
        
        system.add_equations('vertex_x_max_y_min_z_max', vertex_x_max_y_min_z_max_eqs)
        
        # 頂点 (x = nx-1, y = ny-1, z = 0)
        vertex_x_max_y_max_z_min_eqs = [
            DirichletBoundaryEquation3D(grid=grid),
            converter.to_x(NeumannBoundaryEquation(), grid=grid),
            converter.to_x(
                RightBoundary1stDerivativeEquation(),
                #RightBoundary2ndDerivativeEquation()+ 
                #RightBoundary3rdDerivativeEquation(), 
                grid=grid),
            converter.to_y(NeumannBoundaryEquation(), grid=grid),
            converter.to_y(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_y(
                RightBoundary1stDerivativeEquation(),
                #RightBoundary3rdDerivativeEquation(), 
                grid=grid),
            converter.to_z(NeumannBoundaryEquation(), grid=grid),
            converter.to_z(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_z(
                LeftBoundary1stDerivativeEquation(),
                #LeftBoundary3rdDerivativeEquation(), 
                grid=grid)
        ]
        
        system.add_equations('vertex_x_max_y_max_z_min', vertex_x_max_y_max_z_min_eqs)
        
        # 頂点 (x = nx-1, y = ny-1, z = nz-1)
        vertex_x_max_y_max_z_max_eqs = [
            converter.to_x(NeumannBoundaryEquation(), grid=grid),
            converter.to_x(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_x(
                RightBoundary1stDerivativeEquation(),
                #RightBoundary3rdDerivativeEquation(), 
                grid=grid),
            DirichletBoundaryEquation3D(grid=grid),
            converter.to_y(NeumannBoundaryEquation(), grid=grid),
            converter.to_y(
                RightBoundary1stDerivativeEquation(),
                #RightBoundary2ndDerivativeEquation()+ 
                #RightBoundary3rdDerivativeEquation(), 
                grid=grid),
            converter.to_z(NeumannBoundaryEquation(), grid=grid),
            converter.to_z(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_z(
                RightBoundary1stDerivativeEquation(),
                #RightBoundary3rdDerivativeEquation(), 
                grid=grid)
        ]
        
        system.add_equations('vertex_x_max_y_max_z_max', vertex_x_max_y_max_z_max_eqs)
        
        return True, True