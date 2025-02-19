import os
from typing import Tuple, Optional, Dict, Union
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.core.spatial_discretization.base import SpatialDiscretizationBase
from src.core.spatial_discretization.operators.ccd import CombinedCompactDifference
from src.core.linear_solvers.base import LinearSolverBase
from src.core.linear_solvers.iterative.cg import ConjugateGradient
from src.core.linear_solvers.iterative.sor import SORSolver
from src.core.common.grid import GridManager, GridConfig
from src.core.common.types import GridType, BoundaryCondition, BCType

@dataclass
class PoissonTest:
    """テストケース1: 標準的なポアソン方程式: -∇²u = f"""
    
    def exact_solution(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """厳密解: u(x,y) = sin(πx)sin(πy)"""
        return jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)
    
    def source_term(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """ソース項の計算 f = -∇²u"""
        return 2 * jnp.pi**2 * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)
    
    def boundary_conditions(self) -> Dict[str, BoundaryCondition]:
        """境界条件の定義"""
        return {
            'left': BoundaryCondition(
                type=BCType.DIRICHLET,
                value=0.0,
                location='left'
            ),
            'right': BoundaryCondition(
                type=BCType.DIRICHLET,
                value=0.0,
                location='right'
            ),
            'bottom': BoundaryCondition(
                type=BCType.DIRICHLET,
                value=0.0,
                location='bottom'
            ),
            'top': BoundaryCondition(
                type=BCType.DIRICHLET,
                value=0.0,
                location='top'
            )
        }

@dataclass
class VariableCoefficientPoissonTest:
    """テストケース2: 変数係数ポアソン方程式: -∇⋅(a∇u) = f"""
    
    def coefficient(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """係数 a(x,y) = 1 + x² + y²"""
        return 1.0 + x**2 + y**2
    
    def exact_solution(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """厳密解: u(x,y) = exp(-((x-0.5)²+(y-0.5)²)/0.1)"""
        return jnp.exp(-((x-0.5)**2 + (y-0.5)**2)/0.1)
    
    def source_term(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """ソース項 f = -∇⋅(a∇u)"""
        a = self.coefficient(x, y)
        u = self.exact_solution(x, y)
        
        # 導関数の計算
        dux = -2*(x-0.5)/0.1 * u
        duy = -2*(y-0.5)/0.1 * u
        duxx = (-2/0.1 + 4*(x-0.5)**2/0.1**2) * u
        duyy = (-2/0.1 + 4*(y-0.5)**2/0.1**2) * u
        
        # 係数の導関数
        dax = 2*x
        day = 2*y
        
        return -(a * (duxx + duyy) + dax * dux + day * duy)
    
    def boundary_conditions(self) -> Dict[str, BoundaryCondition]:
        """境界条件の定義"""
        def boundary_value(x: float, y: float) -> float:
            return self.exact_solution(
                jnp.array(x), jnp.array(y)
            ).item()
            
        return {
            'left': BoundaryCondition(
                type=BCType.DIRICHLET,
                value=boundary_value,
                location='left'
            ),
            'right': BoundaryCondition(
                type=BCType.DIRICHLET,
                value=boundary_value,
                location='right'
            ),
            'bottom': BoundaryCondition(
                type=BCType.DIRICHLET,
                value=boundary_value,
                location='bottom'
            ),
            'top': BoundaryCondition(
                type=BCType.DIRICHLET,
                value=boundary_value,
                location='top'
            )
        }

def run_solver_test(
    test_case: Union[PoissonTest, VariableCoefficientPoissonTest],
    nx: int,
    ny: int,
    discretization: SpatialDiscretizationBase,
    solver: LinearSolverBase,
    tolerance: float = 1e-6
) -> Tuple[jnp.ndarray, float, dict]:
    """ソルバのテストを実行"""
    
    # 格子の生成
    x = jnp.linspace(0, 1, nx)
    y = jnp.linspace(0, 1, ny)
    X, Y = jnp.meshgrid(x, y)
    
    # ソース項と厳密解の計算
    f = test_case.source_term(X, Y)
    u_exact = test_case.exact_solution(X, Y)
    
    # 係数の設定（変数係数の場合）
    if isinstance(test_case, VariableCoefficientPoissonTest):
        a = test_case.coefficient(X, Y)
    else:
        a = jnp.ones_like(X)
    
    # 作用素の構築
    def operator(u: jnp.ndarray) -> jnp.ndarray:
        # 導関数の計算
        du_dx = discretization.discretize(u, 'x')[0]
        du_dy = discretization.discretize(u, 'y')[0]
        
        # 係数の適用
        adu_dx = a * du_dx
        adu_dy = a * du_dy
        
        # 発散の計算
        div_x = discretization.discretize(adu_dx, 'x')[0]
        div_y = discretization.discretize(adu_dy, 'y')[0]
        
        return -(div_x + div_y)
    
    # システムを解く
    u0 = jnp.zeros_like(f)
    u, history = solver.solve(operator, f, u0)
    
    # 誤差の計算
    error = jnp.linalg.norm(u - u_exact) / jnp.linalg.norm(u_exact)
    
    return u, error, history

def plot_solution_comparison(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    u_numerical: jnp.ndarray,
    u_exact: jnp.ndarray,
    title: str,
    filename: str
) -> Figure:
    """数値解と厳密解の比較プロット"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # 数値解のプロット
    im1 = ax1.pcolormesh(X, Y, u_numerical, shading='auto')
    ax1.set_title('数値解')
    fig.colorbar(im1, ax=ax1)
    
    # 厳密解のプロット
    im2 = ax2.pcolormesh(X, Y, u_exact, shading='auto')
    ax2.set_title('厳密解')
    fig.colorbar(im2, ax=ax2)
    
    # 誤差のプロット
    error = jnp.abs(u_numerical - u_exact)
    im3 = ax3.pcolormesh(X, Y, error, shading='auto')
    ax3.set_title('絶対誤差')
    fig.colorbar(im3, ax=ax3)
    
    plt.suptitle(title)
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    return fig

def plot_convergence_history(
    history: dict,
    title: str,
    filename: str
) -> Figure:
    """収束履歴のプロット"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    residuals = history['residual_norms']
    iterations = range(len(residuals))
    
    ax.semilogy(iterations, residuals, 'b-')
    ax.set_xlabel('反復回数')
    ax.set_ylabel('残差ノルム')
    ax.set_title(title)
    ax.grid(True)
    
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    return fig

def main():
    """全テストの実行と結果の出力"""
    # 出力ディレクトリの作成
    os.makedirs('test_results/poisson', exist_ok=True)
    
    # 格子の設定
    nx = ny = 64
    grid_config = GridConfig(
        dimensions=(1.0, 1.0, 1.0),
        points=(nx, ny, 1),
        grid_type=GridType.UNIFORM
    )
    grid_manager = GridManager(grid_config)
    
    # 格子の生成
    x = jnp.linspace(0, 1, nx)
    y = jnp.linspace(0, 1, ny)
    X, Y = jnp.meshgrid(x, y)
    
    # テストケースの作成
    poisson_test = PoissonTest()
    variable_poisson_test = VariableCoefficientPoissonTest()
    
    # 離散化とソルバの作成
    discretization = CombinedCompactDifference(
        grid_manager=grid_manager,
        boundary_conditions=poisson_test.boundary_conditions()
    )
    
    solvers = {
        'CG': ConjugateGradient(max_iterations=1000, tolerance=1e-6),
        'SOR': SORSolver(omega=1.5, max_iterations=1000, tolerance=1e-6)
    }
    
    # 標準ポアソン方程式のテスト
    print("標準ポアソン方程式のテスト...")
    for solver_name, solver in solvers.items():
        print(f"{solver_name}を使用...")
        u, error, history = run_solver_test(
            poisson_test, nx, ny, discretization, solver
        )
        
        plot_solution_comparison(
            X, Y, u, poisson_test.exact_solution(X, Y),
            f"ポアソン方程式 - {solver_name}",
            f"test_results/poisson/standard_{solver_name.lower()}"
        )
        
        plot_convergence_history(
            history,
            f"収束履歴 - {solver_name}",
            f"test_results/poisson/standard_{solver_name.lower()}_convergence"
        )
        
        print(f"相対誤差: {error}")
    
    # 変数係数ポアソン方程式のテスト
    print("\n変数係数ポアソン方程式のテスト...")
    for solver_name, solver in solvers.items():
        print(f"{solver_name}を使用...")
        u, error, history = run_solver_test(
            variable_poisson_test, nx, ny, discretization, solver
        )
        
        plot_solution_comparison(
            X, Y, u, variable_poisson_test.exact_solution(X, Y),
            f"変数係数ポアソン方程式 - {solver_name}",
            f"test_results/poisson/variable_{solver_name.lower()}"
        )
        
        plot_convergence_history(
            history,
            f"収束履歴 - {solver_name}",
            f"test_results/poisson/variable_{solver_name.lower()}_convergence"
        )
        
        print(f"相対誤差: {error}")

if __name__ == '__main__':
    main()