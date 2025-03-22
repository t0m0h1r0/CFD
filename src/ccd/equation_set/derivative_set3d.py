"""
3次元高階微分のための方程式セット定義モジュール
"""

from core.base.base_equation_set import EquationSet
from equation.dim3.original import OriginalEquation3D
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
        
        # ========== 内部点 ==========
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
        
        # ========== 面の方程式 ==========
        # x = 0 (左面)
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
        
        # x = nx-1 (右面)
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
        
        # y = 0 (下面)
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
        
        # y = ny-1 (上面)
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
        
        # z = 0 (前面)
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
        
        # z = nz-1 (後面)
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
        
        # ========== 辺の方程式 ==========
        # x方向の辺 (y面とz面の交線)
        
        # y=0, z=0面交差辺（下前）
        system.add_equations('edge_x_y_min_z_min', [
            converter.to_x(Internal1stDerivativeEquation(), grid=grid),
            converter.to_x(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_x(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_y(LeftBoundary1stDerivativeEquation(), grid=grid),
            converter.to_y(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_y(LeftBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_z(LeftBoundary1stDerivativeEquation(), grid=grid),
            converter.to_z(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_z(LeftBoundary3rdDerivativeEquation(), grid=grid)
        ])
        
        # y=0, z=nz-1面交差辺（下後）
        system.add_equations('edge_x_y_min_z_max', [
            converter.to_x(Internal1stDerivativeEquation(), grid=grid),
            converter.to_x(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_x(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_y(LeftBoundary1stDerivativeEquation(), grid=grid),
            converter.to_y(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_y(LeftBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_z(RightBoundary1stDerivativeEquation(), grid=grid),
            converter.to_z(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_z(RightBoundary3rdDerivativeEquation(), grid=grid)
        ])
        
        # y=ny-1, z=0面交差辺（上前）
        system.add_equations('edge_x_y_max_z_min', [
            converter.to_x(Internal1stDerivativeEquation(), grid=grid),
            converter.to_x(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_x(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_y(RightBoundary1stDerivativeEquation(), grid=grid),
            converter.to_y(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_y(RightBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_z(LeftBoundary1stDerivativeEquation(), grid=grid),
            converter.to_z(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_z(LeftBoundary3rdDerivativeEquation(), grid=grid)
        ])
        
        # y=ny-1, z=nz-1面交差辺（上後）
        system.add_equations('edge_x_y_max_z_max', [
            converter.to_x(Internal1stDerivativeEquation(), grid=grid),
            converter.to_x(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_x(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_y(RightBoundary1stDerivativeEquation(), grid=grid),
            converter.to_y(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_y(RightBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_z(RightBoundary1stDerivativeEquation(), grid=grid),
            converter.to_z(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_z(RightBoundary3rdDerivativeEquation(), grid=grid)
        ])
        
        # y方向の辺 (x面とz面の交線)
        
        # x=0, z=0面交差辺（左前）
        system.add_equations('edge_y_x_min_z_min', [
            converter.to_x(LeftBoundary1stDerivativeEquation(), grid=grid),
            converter.to_x(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_x(LeftBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_y(Internal1stDerivativeEquation(), grid=grid),
            converter.to_y(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_y(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_z(LeftBoundary1stDerivativeEquation(), grid=grid),
            converter.to_z(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_z(LeftBoundary3rdDerivativeEquation(), grid=grid)
        ])
        
        # x=0, z=nz-1面交差辺（左後）
        system.add_equations('edge_y_x_min_z_max', [
            converter.to_x(LeftBoundary1stDerivativeEquation(), grid=grid),
            converter.to_x(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_x(LeftBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_y(Internal1stDerivativeEquation(), grid=grid),
            converter.to_y(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_y(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_z(RightBoundary1stDerivativeEquation(), grid=grid),
            converter.to_z(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_z(RightBoundary3rdDerivativeEquation(), grid=grid)
        ])
        
        # x=nx-1, z=0面交差辺（右前）
        system.add_equations('edge_y_x_max_z_min', [
            converter.to_x(RightBoundary1stDerivativeEquation(), grid=grid),
            converter.to_x(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_x(RightBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_y(Internal1stDerivativeEquation(), grid=grid),
            converter.to_y(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_y(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_z(LeftBoundary1stDerivativeEquation(), grid=grid),
            converter.to_z(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_z(LeftBoundary3rdDerivativeEquation(), grid=grid)
        ])
        
        # x=nx-1, z=nz-1面交差辺（右後）
        system.add_equations('edge_y_x_max_z_max', [
            converter.to_x(RightBoundary1stDerivativeEquation(), grid=grid),
            converter.to_x(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_x(RightBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_y(Internal1stDerivativeEquation(), grid=grid),
            converter.to_y(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_y(Internal3rdDerivativeEquation(), grid=grid),
            converter.to_z(RightBoundary1stDerivativeEquation(), grid=grid),
            converter.to_z(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_z(RightBoundary3rdDerivativeEquation(), grid=grid)
        ])
        
        # z方向の辺 (x面とy面の交線)
        
        # x=0, y=0面交差辺（左下）
        system.add_equations('edge_z_x_min_y_min', [
            converter.to_x(LeftBoundary1stDerivativeEquation(), grid=grid),
            converter.to_x(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_x(LeftBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_y(LeftBoundary1stDerivativeEquation(), grid=grid),
            converter.to_y(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_y(LeftBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_z(Internal1stDerivativeEquation(), grid=grid),
            converter.to_z(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_z(Internal3rdDerivativeEquation(), grid=grid)
        ])
        
        # x=0, y=ny-1面交差辺（左上）
        system.add_equations('edge_z_x_min_y_max', [
            converter.to_x(LeftBoundary1stDerivativeEquation(), grid=grid),
            converter.to_x(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_x(LeftBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_y(RightBoundary1stDerivativeEquation(), grid=grid),
            converter.to_y(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_y(RightBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_z(Internal1stDerivativeEquation(), grid=grid),
            converter.to_z(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_z(Internal3rdDerivativeEquation(), grid=grid)
        ])
        
        # x=nx-1, y=0面交差辺（右下）
        system.add_equations('edge_z_x_max_y_min', [
            converter.to_x(RightBoundary1stDerivativeEquation(), grid=grid),
            converter.to_x(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_x(RightBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_y(LeftBoundary1stDerivativeEquation(), grid=grid),
            converter.to_y(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_y(LeftBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_z(Internal1stDerivativeEquation(), grid=grid),
            converter.to_z(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_z(Internal3rdDerivativeEquation(), grid=grid)
        ])
        
        # x=nx-1, y=ny-1面交差辺（右上）
        system.add_equations('edge_z_x_max_y_max', [
            converter.to_x(RightBoundary1stDerivativeEquation(), grid=grid),
            converter.to_x(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_x(RightBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_y(RightBoundary1stDerivativeEquation(), grid=grid),
            converter.to_y(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_y(RightBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_z(Internal1stDerivativeEquation(), grid=grid),
            converter.to_z(Internal2ndDerivativeEquation(), grid=grid),
            converter.to_z(Internal3rdDerivativeEquation(), grid=grid)
        ])
        
        # ========== 頂点の方程式 ==========
        
        # 左下前 (i=0, j=0, k=0)
        system.add_equations('vertex_x_min_y_min_z_min', [
            converter.to_x(LeftBoundary1stDerivativeEquation(), grid=grid),
            converter.to_x(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_x(LeftBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_y(LeftBoundary1stDerivativeEquation(), grid=grid),
            converter.to_y(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_y(LeftBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_z(LeftBoundary1stDerivativeEquation(), grid=grid),
            converter.to_z(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_z(LeftBoundary3rdDerivativeEquation(), grid=grid)
        ])
        
        # 右下前 (i=nx-1, j=0, k=0)
        system.add_equations('vertex_x_max_y_min_z_min', [
            converter.to_x(RightBoundary1stDerivativeEquation(), grid=grid),
            converter.to_x(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_x(RightBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_y(LeftBoundary1stDerivativeEquation(), grid=grid),
            converter.to_y(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_y(LeftBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_z(LeftBoundary1stDerivativeEquation(), grid=grid),
            converter.to_z(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_z(LeftBoundary3rdDerivativeEquation(), grid=grid)
        ])
        
        # 左上前 (i=0, j=ny-1, k=0)
        system.add_equations('vertex_x_min_y_max_z_min', [
            converter.to_x(LeftBoundary1stDerivativeEquation(), grid=grid),
            converter.to_x(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_x(LeftBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_y(RightBoundary1stDerivativeEquation(), grid=grid),
            converter.to_y(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_y(RightBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_z(LeftBoundary1stDerivativeEquation(), grid=grid),
            converter.to_z(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_z(LeftBoundary3rdDerivativeEquation(), grid=grid)
        ])
        
        # 右上前 (i=nx-1, j=ny-1, k=0)
        system.add_equations('vertex_x_max_y_max_z_min', [
            converter.to_x(RightBoundary1stDerivativeEquation(), grid=grid),
            converter.to_x(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_x(RightBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_y(RightBoundary1stDerivativeEquation(), grid=grid),
            converter.to_y(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_y(RightBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_z(LeftBoundary1stDerivativeEquation(), grid=grid),
            converter.to_z(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_z(LeftBoundary3rdDerivativeEquation(), grid=grid)
        ])
        
        # 左下後 (i=0, j=0, k=nz-1)
        system.add_equations('vertex_x_min_y_min_z_max', [
            converter.to_x(LeftBoundary1stDerivativeEquation(), grid=grid),
            converter.to_x(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_x(LeftBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_y(LeftBoundary1stDerivativeEquation(), grid=grid),
            converter.to_y(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_y(LeftBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_z(RightBoundary1stDerivativeEquation(), grid=grid),
            converter.to_z(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_z(RightBoundary3rdDerivativeEquation(), grid=grid)
        ])
        
        # 右下後 (i=nx-1, j=0, k=nz-1)
        system.add_equations('vertex_x_max_y_min_z_max', [
            converter.to_x(RightBoundary1stDerivativeEquation(), grid=grid),
            converter.to_x(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_x(RightBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_y(LeftBoundary1stDerivativeEquation(), grid=grid),
            converter.to_y(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_y(LeftBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_z(RightBoundary1stDerivativeEquation(), grid=grid),
            converter.to_z(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_z(RightBoundary3rdDerivativeEquation(), grid=grid)
        ])
        
        # 左上後 (i=0, j=ny-1, k=nz-1)
        system.add_equations('vertex_x_min_y_max_z_max', [
            converter.to_x(LeftBoundary1stDerivativeEquation(), grid=grid),
            converter.to_x(LeftBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_x(LeftBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_y(RightBoundary1stDerivativeEquation(), grid=grid),
            converter.to_y(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_y(RightBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_z(RightBoundary1stDerivativeEquation(), grid=grid),
            converter.to_z(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_z(RightBoundary3rdDerivativeEquation(), grid=grid)
        ])
        
        # 右上後 (i=nx-1, j=ny-1, k=nz-1)
        system.add_equations('vertex_x_max_y_max_z_max', [
            converter.to_x(RightBoundary1stDerivativeEquation(), grid=grid),
            converter.to_x(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_x(RightBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_y(RightBoundary1stDerivativeEquation(), grid=grid),
            converter.to_y(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_y(RightBoundary3rdDerivativeEquation(), grid=grid),
            converter.to_z(RightBoundary1stDerivativeEquation(), grid=grid),
            converter.to_z(RightBoundary2ndDerivativeEquation(), grid=grid),
            converter.to_z(RightBoundary3rdDerivativeEquation(), grid=grid)
        ])
        
        # 微分方程式セットでは境界条件は常に無効（既に導関数計算に制約式を使用）
        return False, False