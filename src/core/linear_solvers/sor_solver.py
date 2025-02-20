from typing import Optional, Dict, Tuple
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from functools import partial

from ..common.grid import GridManager
from ..common.types import BoundaryCondition, BCType
from ..spatial_discretization.operators.ccd_laplacian import CCDLaplacianSolver

class PoissonSORSolver:
    """
    高精度コンパクト差分法を用いたポアソン方程式のSORソルバ
    
    理論背景：
    - ラプラシアン計算にCCDLaplacianSolverを使用
    - SOR法による反復解法
    - JAX最適化による高速化
    """
    
    def __init__(
        self,
        grid_manager: GridManager,
        boundary_conditions: Optional[Dict[str, BoundaryCondition]] = None,
        omega: float = 1.5,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        verbose: bool = False
    ):
        """
        ポアソンSORソルバの初期化
        
        Args:
            grid_manager: グリッド管理オブジェクト
            boundary_conditions: 境界条件の辞書
            omega: 緩和パラメータ (1 < omega < 2)
            max_iterations: 最大反復回数
            tolerance: 収束判定の許容誤差
            verbose: デバッグ出力フラグ
        """
        self.grid_manager = grid_manager
        self.laplacian_solver = CCDLaplacianSolver(
            grid_manager=grid_manager, 
            boundary_conditions=boundary_conditions
        )
        self.omega = omega
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose
        
        # グリッドの情報取得
        self.shape = (
            grid_manager.get_grid_points('x'),
            grid_manager.get_grid_points('y'),
            grid_manager.get_grid_points('z')
        )

    @partial(jax.jit, static_argnums=(0,))
    def _compute_laplacian(self, p: ArrayLike, f: ArrayLike) -> ArrayLike:
        """JAX最適化されたラプラシアン計算"""
        return self.laplacian_solver.compute_laplacian(p) - f

    @partial(jax.jit, static_argnums=(0,))
    def _single_sor_iteration(
        self, 
        inputs: Tuple[ArrayLike, ArrayLike, int]
    ) -> Tuple[ArrayLike, ArrayLike, int]:
        """
        単一SOR反復のJAX最適化バージョン
        
        Args:
            inputs: (現在の解p, 右辺f, 現在の反復回数)
        
        Returns:
            (更新された解p, 右辺f, 更新された反復回数)
        """
        p, f, iteration = inputs
        omega = self.omega
        
        # Red-Black SOR法の高速化
        def update_color(p, color):
            # JAXでのインデックスマスキング
            mask = jnp.indices(p.shape).sum(0) % 2 == color
            
            # SOR更新のための関数
            def update_point(p, idx):
                i, j, k = idx
                
                # 6近傍点の取得（周期的/ゼロ境界条件を想定）
                neighbors = jnp.array([
                    p[max(0, i-1), j, k] if i > 0 else 0,
                    p[min(self.shape[0]-1, i+1), j, k] if i < self.shape[0]-1 else 0,
                    p[i, max(0, j-1), k] if j > 0 else 0,
                    p[i, min(self.shape[1]-1, j+1), k] if j < self.shape[1]-1 else 0,
                    p[i, j, max(0, k-1)] if k > 0 else 0,
                    p[i, j, min(self.shape[2]-1, k+1)] if k < self.shape[2]-1 else 0,
                ])
                
                # 隣接点数の計算
                neighbor_count = jnp.sum(neighbors != 0)
                
                # 平均値の計算
                avg_neighbor = jnp.sum(neighbors) / neighbor_count
                
                # SOR更新式
                p_new = (1 - omega) * p[i, j, k] + omega * (avg_neighbor - f[i, j, k] / 6)
                
                return p_new
            
            # カラーに応じた点の更新
            indices = jnp.argwhere(mask)
            updated_points = jax.vmap(update_point)(indices)
            
            # インデックスに応じて更新
            return p.at[mask].set(updated_points)
        
        # 両方の色で更新
        p = update_color(p, 0)
        p = update_color(p, 1)
        
        return p, f, iteration + 1

    @partial(jax.jit, static_argnums=(0,))
    def solve(
        self, 
        f: ArrayLike, 
        initial_guess: Optional[ArrayLike] = None
    ) -> Tuple[ArrayLike, Dict]:
        """
        JAX最適化されたポアソン方程式の解法
        
        Args:
            f: 右辺項のソース関数
            initial_guess: 初期推定解（デフォルトはゼロ）
        
        Returns:
            解フィールドと収束情報
        """
        # 初期推定解の準備
        p = initial_guess if initial_guess is not None else jnp.zeros_like(f)
        
        # 反復終了条件の定義
        def convergence_condition(state):
            p, f, iteration = state
            
            # ラプラシアンの計算と残差評価
            laplacian = self._compute_laplacian(p, f)
            residual = jnp.linalg.norm(laplacian)
            
            # 収束判定
            return jnp.logical_and(
                residual >= self.tolerance,
                iteration < self.max_iterations
            )
        
        # while_loopによる反復
        final_state = jax.lax.while_loop(
            convergence_condition, 
            self._single_sor_iteration, 
            (p, f, 0)
        )
        
        # 最終状態の展開
        p_solved, _, iterations = final_state
        
        # ラプラシアンと残差の計算
        laplacian = self._compute_laplacian(p_solved, f)
        final_residual = float(jnp.linalg.norm(laplacian))
        
        # 収束情報の構築
        history = {
            'iterations': int(iterations),
            'final_residual': final_residual,
            'converged': final_residual < self.tolerance
        }
        
        return p_solved, history

    def optimize_omega(
        self, 
        f: ArrayLike, 
        omega_range: Tuple[float, float] = (1.0, 2.0),
        steps: int = 20
    ) -> float:
        """
        最適な緩和パラメータの探索
        
        Args:
            f: 右辺項
            omega_range: 探索する緩和パラメータの範囲
            steps: 探索ステップ数
        
        Returns:
            最適な緩和パラメータ
        """
        def test_omega(omega):
            self.omega = omega
            _, history = self.solve(f)
            return history['iterations']
        
        # JAXでのベクトル化
        test_omega_vec = jax.vmap(test_omega)
        
        # オメガ値の探索
        omegas = jnp.linspace(omega_range[0], omega_range[1], steps)
        iterations = test_omega_vec(omegas)
        
        # 最適なオメガの選択
        return float(omegas[jnp.argmin(iterations)])