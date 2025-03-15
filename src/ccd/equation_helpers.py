from equation.equation_converter import Equation1Dto2DConverter
from equation.poisson import PoissonEquation, PoissonEquation2D
from equation.original import OriginalEquation, OriginalEquation2D
from equation.boundary import (
    DirichletBoundaryEquation, NeumannBoundaryEquation,
    DirichletBoundaryEquation2D, NeumannXBoundaryEquation2D, NeumannYBoundaryEquation2D
)
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

def create_derivative_set(direction, boundary_type, merge=False, grid=None):
    """
    指定された方向と境界タイプに基づいて微分方程式セットを作成
    
    Args:
        direction: 方向 ('x', 'y') または None (1Dの場合)
        boundary_type: 境界タイプ ('left'/'min', 'right'/'max', 'interior')
        merge: 2階と3階の微分を結合するか
        grid: 計算グリッド (オプション)
    
    Returns:
        指定条件に合った微分方程式のリスト
    """
    converter = Equation1Dto2DConverter
    
    if boundary_type == 'interior':
        # 内部点の導関数近似
        eqs = [
            Internal1stDerivativeEquation(),
            Internal2ndDerivativeEquation(),
            Internal3rdDerivativeEquation()
        ]
    elif boundary_type in ['left', 'min']:
        # 左/下境界の導関数近似
        if merge:
            eqs = [
                LeftBoundary1stDerivativeEquation(),
                LeftBoundary2ndDerivativeEquation() + LeftBoundary3rdDerivativeEquation()
            ]
        else:
            eqs = [
                LeftBoundary1stDerivativeEquation(),
                LeftBoundary2ndDerivativeEquation(),
                LeftBoundary3rdDerivativeEquation()
            ]
    else:  # 'right', 'max'
        # 右/上境界の導関数近似
        if merge:
            eqs = [
                RightBoundary1stDerivativeEquation(),
                RightBoundary2ndDerivativeEquation() + RightBoundary3rdDerivativeEquation()
            ]
        else:
            eqs = [
                RightBoundary1stDerivativeEquation(),
                RightBoundary2ndDerivativeEquation(),
                RightBoundary3rdDerivativeEquation()
            ]
    
    # 方向に基づいて変換 (2Dの場合)
    if direction:
        if direction == 'x':
            return [converter.to_x(eq, grid=grid) for eq in eqs]
        else:  # 'y'
            return [converter.to_y(eq, grid=grid) for eq in eqs]
    else:
        # 1Dの場合はそのまま返す
        if grid:
            return [eq.set_grid(grid) for eq in eqs]
        return eqs

def setup_region_equations(system, region_type, merge_derivatives=False, grid=None):
    """
    特定領域の方程式を設定
    
    Args:
        system: 方程式システム
        region_type: 領域タイプ ('interior', 'left', 'right', 'bottom', 'top', 'left_bottom', ...)
        merge_derivatives: 高階微分を結合するかどうか
        grid: 計算グリッド
    """
    if system.is_2d:
        # 2Dグリッド用の設定
        if region_type == 'interior':
            # 内部点: 両方向とも内部点の微分
            system.add_equations('interior', create_derivative_set('x', 'interior', False, grid))
            system.add_equations('interior', create_derivative_set('y', 'interior', False, grid))
        
        elif region_type == 'left':
            # 左境界: x方向は左境界、y方向は内部点
            system.add_equations('left', create_derivative_set('x', 'left', merge_derivatives, grid))
            system.add_equations('left', create_derivative_set('y', 'interior', False, grid))
        
        elif region_type == 'right':
            # 右境界: x方向は右境界、y方向は内部点
            system.add_equations('right', create_derivative_set('x', 'right', merge_derivatives, grid))
            system.add_equations('right', create_derivative_set('y', 'interior', False, grid))
        
        elif region_type == 'bottom':
            # 下境界: x方向は内部点、y方向は下境界
            system.add_equations('bottom', create_derivative_set('x', 'interior', False, grid))
            system.add_equations('bottom', create_derivative_set('y', 'left', merge_derivatives, grid))
        
        elif region_type == 'top':
            # 上境界: x方向は内部点、y方向は上境界
            system.add_equations('top', create_derivative_set('x', 'interior', False, grid))
            system.add_equations('top', create_derivative_set('y', 'right', merge_derivatives, grid))
        
        elif region_type == 'left_bottom':
            # 左下角: x方向は左境界、y方向は下境界
            system.add_equations('left_bottom', create_derivative_set('x', 'left', merge_derivatives, grid))
            system.add_equations('left_bottom', create_derivative_set('y', 'left', merge_derivatives, grid))
        
        elif region_type == 'right_bottom':
            # 右下角: x方向は右境界、y方向は下境界
            system.add_equations('right_bottom', create_derivative_set('x', 'right', merge_derivatives, grid))
            system.add_equations('right_bottom', create_derivative_set('y', 'left', merge_derivatives, grid))
        
        elif region_type == 'left_top':
            # 左上角: x方向は左境界、y方向は上境界
            system.add_equations('left_top', create_derivative_set('x', 'left', merge_derivatives, grid))
            system.add_equations('left_top', create_derivative_set('y', 'right', merge_derivatives, grid))
        
        elif region_type == 'right_top':
            # 右上角: x方向は右境界、y方向は上境界
            system.add_equations('right_top', create_derivative_set('x', 'right', merge_derivatives, grid))
            system.add_equations('right_top', create_derivative_set('y', 'right', merge_derivatives, grid))
    else:
        # 1Dグリッド用の設定
        if region_type == 'interior':
            # 内部点
            system.add_equations('interior', create_derivative_set(None, 'interior', False, grid))
        elif region_type == 'left':
            # 左境界
            system.add_equations('left', create_derivative_set(None, 'left', merge_derivatives, grid))
        elif region_type == 'right':
            # 右境界
            system.add_equations('right', create_derivative_set(None, 'right', merge_derivatives, grid))

def setup_derivative_equations_1d(system, grid, merge_derivatives=False):
    """
    1Dグリッドの導関数方程式を設定
    
    Args:
        system: 方程式システム
        grid: 計算グリッド
        merge_derivatives: 高階微分を結合するかどうか
    """
    # 元の関数を全ての領域に追加
    system.add_dominant_equation(OriginalEquation(grid=grid))
    
    # 各領域に微分方程式を設定
    setup_region_equations(system, 'interior', False, grid)
    setup_region_equations(system, 'left', merge_derivatives, grid)
    setup_region_equations(system, 'right', merge_derivatives, grid)

def setup_derivative_equations_2d(system, grid, merge_derivatives=False):
    """
    2Dグリッドの導関数方程式を設定
    
    Args:
        system: 方程式システム
        grid: 計算グリッド
        merge_derivatives: 高階微分を結合するかどうか
    """
    # 元の関数を全ての領域に追加
    system.add_dominant_equation(OriginalEquation2D(grid=grid))
    
    # 内部点
    setup_region_equations(system, 'interior', False, grid)
    
    # 境界
    for region in ['left', 'right', 'bottom', 'top']:
        setup_region_equations(system, region, merge_derivatives, grid)
    
    # 角
    for corner in ['left_bottom', 'right_bottom', 'left_top', 'right_top']:
        setup_region_equations(system, corner, merge_derivatives, grid)

def setup_poisson_equations_1d(system, grid, use_dirichlet=True, use_neumann=True, merge_derivatives=False):
    """
    1Dポアソン方程式システムを設定
    
    Args:
        system: 方程式システム
        grid: 計算グリッド
        use_dirichlet: ディリクレ境界条件を使用するか
        use_neumann: ノイマン境界条件を使用するか
        merge_derivatives: 高階微分を結合するかどうか
    """
    # ポアソン方程式を全ての領域に追加
    system.add_dominant_equation(PoissonEquation(grid=grid))
    
    # 境界条件を設定
    system.add_boundary_equations({
        'dirichlet': use_dirichlet,
        'neumann': use_neumann
    })
    
    # 各領域に微分方程式を設定
    setup_region_equations(system, 'interior', False, grid)
    setup_region_equations(system, 'left', merge_derivatives, grid)
    setup_region_equations(system, 'right', merge_derivatives, grid)

def setup_poisson_equations_2d(system, grid, use_dirichlet=True, use_neumann=True, merge_derivatives=False):
    """
    2Dポアソン方程式システムを設定
    
    Args:
        system: 方程式システム
        grid: 計算グリッド
        use_dirichlet: ディリクレ境界条件を使用するか
        use_neumann: ノイマン境界条件を使用するか
        merge_derivatives: 高階微分を結合するかどうか
    """
    # ポアソン方程式を全ての領域に追加
    system.add_dominant_equation(PoissonEquation2D(grid=grid))
    
    # 境界条件を設定
    system.add_boundary_equations({
        'dirichlet': use_dirichlet,
        'neumann': use_neumann
    })
    
    # 内部点
    setup_region_equations(system, 'interior', False, grid)
    
    # 境界
    for region in ['left', 'right', 'bottom', 'top']:
        setup_region_equations(system, region, merge_derivatives, grid)
    
    # 角
    for corner in ['left_bottom', 'right_bottom', 'left_top', 'right_top']:
        setup_region_equations(system, corner, merge_derivatives, grid)
