from typing import Optional, Dict, Tuple
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..common.grid import GridManager
from ..common.types import BoundaryCondition, BCType
from ..spatial_discretization.operators.ccd_laplacian import CCDLaplacianSolver

class PoissonSORSolver:
    """
    高精度コンパクト差分法を用いたポアソン方程式のSORソルバ
    
    理論背景：
    - ラプラシアン計算にCCDLaplacianSolverを使用
    - SOR法による反復解法
    - 緩和パラメータによる収束加速
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
        
    def solve(self, f: ArrayLike, initial_guess: Optional[ArrayLike] = None) -> Tuple[ArrayLike, Dict]:
        """
        ポアソン方程式の解法
        
        Args:
            f: 右辺項のソース関数
            initial_guess: 初期推定解（デフォルトはゼロ）
        
        Returns:
            解フィールドと収束情報
        """
        # 初期推定解の準備
        p = initial_guess if initial_guess is not None else jnp.zeros_like(f)
        
        # 収束情報の初期化
        history = {
            'iterations': 0,
            'residual_history': [],
            'converged': False
        }
        
        # 反復計算
        for iter_count in range(self.max_iterations):
            p_old = p.copy()
            
            # Red-Black SOR法の実装
            for color in [0, 1]:
                for k in range(self.shape[2]):
                    for j in range(self.shape[1]):
                        for i in range(self.shape[0]):
                            if (i + j + k) % 2 == color:
                                # ラプラシアンの計算
                                p = self._update_point(p, f, i, j, k)
            
            # 残差の計算
            laplacian = self.laplacian_solver.compute_laplacian(p)
            residual = jnp.linalg.norm(laplacian - f)
            
            history['residual_history'].append(residual)
            
            if self.verbose:
                print(f"反復 {iter_count+1}: 残差 = {residual}")
            
            # 収束判定
            if residual < self.tolerance:
                history['converged'] = True
                history['iterations'] = iter_count + 1
                break
        
        return p, history
    
    def _update_point(
        self, 
        p: ArrayLike, 
        f: ArrayLike, 
        i: int, 
        j: int, 
        k: int
    ) -> ArrayLike:
        """
        単一点の更新（SOR法）
        
        Args:
            p: 現在の解フィールド
            f: 右辺項
            i, j, k: グリッド座標
        
        Returns:
            更新された解フィールド
        """
        # 6近傍点の取得（周期的/ゼロ境界条件を想定）
        neighbors = [
            p[max(0, i-1), j, k] if i > 0 else 0,
            p[min(self.shape[0]-1, i+1), j, k] if i < self.shape[0]-1 else 0,
            p[i, max(0, j-1), k] if j > 0 else 0,
            p[i, min(self.shape[1]-1, j+1), k] if j < self.shape[1]-1 else 0,
            p[i, j, max(0, k-1)] if k > 0 else 0,
            p[i, j, min(self.shape[2]-1, k+1)] if k < self.shape[2]-1 else 0,
        ]
        
        # 隣接点数の計算
        neighbor_count = sum(1 for n in neighbors if n is not None)
        
        # 平均値の計算
        avg_neighbor = sum(n for n in neighbors if n is not None) / neighbor_count
        
        # SOR更新式
        p_new = p.at[i, j, k].set(
            (1 - self.omega) * p[i, j, k] + 
            self.omega * (avg_neighbor - f[i, j, k] / 6)
        )
        
        return p_new
    
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
        
        omegas = jnp.linspace(omega_range[0], omega_range[1], steps)
        iterations = jnp.array([test_omega(float(omega)) for omega in omegas])
        
        return float(omegas[jnp.argmin(iterations)])