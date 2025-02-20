import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple
import matplotlib.pyplot as plt

class FastPoissonSolver:
    """
    2次精度中心差分を用いたポアソンソルバー
    Red-Black SORによる高速化実装
    -∇²u = f with u = 0 on boundary
    """
    def __init__(self, nx: int, ny: int, dx: float = 1.0):
        self.nx = nx  
        self.ny = ny
        self.dx = dx
        self.dx2 = dx * dx
        
        # Red-Black maskの準備
        x = jnp.arange(nx)
        y = jnp.arange(ny)
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        self.red_mask = ((X + Y) % 2 == 0)
        self.black_mask = ((X + Y) % 2 == 1)
        
        # 境界マスクの準備
        self.interior_mask = jnp.ones((nx, ny), dtype=bool)
        self.interior_mask = self.interior_mask.at[0,:].set(False)
        self.interior_mask = self.interior_mask.at[-1,:].set(False)
        self.interior_mask = self.interior_mask.at[:,0].set(False)
        self.interior_mask = self.interior_mask.at[:,-1].set(False)
        
        # 実際に更新する点のマスク
        self.red_interior = self.red_mask & self.interior_mask
        self.black_interior = self.black_mask & self.interior_mask
    
    @partial(jax.jit, static_argnums=(0,))
    def rb_sor_step(self, u: jnp.ndarray, f: jnp.ndarray, omega: float) -> jnp.ndarray:
        """
        Red-Black SORの1ステップ (ベクトル化実装)
        """
        # Red pointsの更新
        u_neighbors = (jnp.roll(u, 1, axis=0) + jnp.roll(u, -1, axis=0) +
                      jnp.roll(u, 1, axis=1) + jnp.roll(u, -1, axis=1))
        u_new = jnp.where(
            self.red_interior,
            (1-omega)*u + (omega/4)*(u_neighbors + self.dx2 * f),
            u
        )
        
        # Black pointsの更新
        u_neighbors = (jnp.roll(u_new, 1, axis=0) + jnp.roll(u_new, -1, axis=0) +
                      jnp.roll(u_new, 1, axis=1) + jnp.roll(u_new, -1, axis=1))
        u_new = jnp.where(
            self.black_interior,
            (1-omega)*u_new + (omega/4)*(u_neighbors + self.dx2 * f),
            u_new
        )
        
        return u_new
    
    @partial(jax.jit, static_argnums=(0,))
    def compute_residual(self, u: jnp.ndarray, f: jnp.ndarray) -> jnp.ndarray:
        """残差 r = f + ∇²u の計算 (ベクトル化実装)"""
        laplacian = (jnp.roll(u, 1, axis=0) + jnp.roll(u, -1, axis=0) +
                    jnp.roll(u, 1, axis=1) + jnp.roll(u, -1, axis=1) - 4*u) / self.dx2
        return jnp.where(self.interior_mask, f + laplacian, 0.0)
    
    def solve(self, 
             f: jnp.ndarray, 
             tol: float = 1e-6,
             max_iter: int = 1000,
             omega: float = 1.5) -> Tuple[jnp.ndarray, dict]:
        """
        Red-Black SOR法によるポアソン方程式の求解
        """
        # 初期解をゼロに設定
        u = jnp.zeros_like(f)
        
        # 収束履歴
        residual_history = []
        
        # 反復計算
        for it in range(max_iter):
            # SORステップ
            u_new = self.rb_sor_step(u, f, omega)
            
            # 残差計算 (100イテレーションごと)
            if (it + 1) % 1000 == 0:
                residual = self.compute_residual(u_new, f)
                res_norm = jnp.max(jnp.abs(residual))
                residual_history.append(float(res_norm))
                
                print(f"Iteration {it+1}, residual: {res_norm}")
                
                # 収束判定
                if res_norm < tol:
                    return u_new, {
                        'converged': True,
                        'iterations': it + 1,
                        'residual': res_norm,
                        'residual_history': residual_history
                    }
            
            u = u_new
        
        # 最終残差計算
        residual = self.compute_residual(u, f)
        res_norm = jnp.max(jnp.abs(residual))
        residual_history.append(float(res_norm))
        
        return u, {
            'converged': False,
            'iterations': max_iter,
            'residual': res_norm,
            'residual_history': residual_history
        }

def test_solver():
    """ソルバーのテスト"""
    # グリッド設定
    N = 512  # グリッド数を増やしても高速に計算可能
    L = 1.0  
    dx = L/(N-1)
    
    # グリッド点生成
    x = jnp.linspace(0, L, N)
    y = jnp.linspace(0, L, N)
    X, Y = jnp.meshgrid(x, y, indexing='ij')
    
    # テスト問題: -∇²u = f
    # 解析解 u = sin(πx)sin(πy)  
    u_exact = jnp.sin(jnp.pi*X) * jnp.sin(jnp.pi*Y)
    
    # 右辺 f = -∇²u = 2π²sin(πx)sin(πy)
    f = 2 * (jnp.pi**2) * u_exact
    
    # ソルバーの作成と実行
    solver = FastPoissonSolver(N, N, dx)
    print("Solving Poisson equation...")
    u_num, info = solver.solve(f, tol=1e-6, max_iter=500000, omega=1.5)
    
    # 相対誤差の計算
    error = jnp.linalg.norm(u_num - u_exact) / jnp.linalg.norm(u_exact)
    print(f"Converged: {info['converged']}")
    print(f"Iterations: {info['iterations']}")
    print(f"Final residual: {info['residual']}")
    print(f"Relative error: {error}")
    
    # 結果の可視化
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # 数値解
    im1 = ax1.pcolormesh(X, Y, u_num, shading='auto')
    ax1.set_title('Numerical Solution')
    plt.colorbar(im1, ax=ax1)
    
    # 厳密解
    im2 = ax2.pcolormesh(X, Y, u_exact, shading='auto')
    ax2.set_title('Exact Solution')
    plt.colorbar(im2, ax=ax2)
    
    # 誤差
    error_field = jnp.abs(u_num - u_exact)
    im3 = ax3.pcolormesh(X, Y, error_field, shading='auto')
    ax3.set_title('Absolute Error')
    plt.colorbar(im3, ax=ax3)
    
    plt.suptitle(f'Poisson Equation Solution (N={N}, ω=1.5)')
    plt.tight_layout()
    plt.savefig('poisson_solution.png', dpi=300, bbox_inches='tight')
    
    # 収束履歴のプロット
    plt.figure(figsize=(8, 6))
    plt.semilogy(info['residual_history'])
    plt.grid(True)
    plt.xlabel('Iteration')
    plt.ylabel('Residual')
    plt.title('Convergence History')
    plt.tight_layout()
    plt.savefig('convergence_history.png', dpi=300, bbox_inches='tight')
    plt.close('all')
    
    return u_num, u_exact, info

if __name__ == "__main__":
    u_num, u_exact, info = test_solver()