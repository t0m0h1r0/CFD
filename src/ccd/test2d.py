"""
シンプル化された2次元CCDソルバー（デバッグ用）

2次元問題を直接解きます。
"""

import cupy as cp
import cupyx.scipy.sparse as cpx_sparse
from typing import List, Optional, Tuple

from grid_config_2d import GridConfig2D
from test_functions_2d import TestFunction2DFactory

class SimpleDebugSolver2D:
    """デバッグ用の2次元偏微分計算クラス"""
    
    def __init__(self, grid_config: GridConfig2D):
        """
        デバッグソルバーの初期化
        
        Args:
            grid_config: 2次元グリッド設定
        """
        self.grid_config = grid_config
        self.nx = grid_config.nx_points
        self.ny = grid_config.ny_points
        self.hx = grid_config.hx
        self.hy = grid_config.hy
        
        print(f"デバッグソルバー初期化: nx={self.nx}, ny={self.ny}, hx={self.hx}, hy={self.hy}")
    
    def solve(self, f: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
        """
        有限差分法で偏導関数を計算（デバッグ用のシンプルなバージョン）
        
        Args:
            f: グリッド点での関数値（nx × ny配列）
            
        Returns:
            (f, f_x, f_y, f_xx, f_xy, f_yy)のタプル
        """
        print(f"入力関数の形状: {f.shape}")
        
        # 入力が正しい形状か確認
        if f.shape != (self.nx, self.ny):
            f = f.reshape(self.nx, self.ny)
            
        # 結果を格納する配列
        f_x = cp.zeros_like(f)
        f_y = cp.zeros_like(f)
        f_xx = cp.zeros_like(f)
        f_xy = cp.zeros_like(f)
        f_yy = cp.zeros_like(f)
        
        # 中心差分で偏導関数を計算（内部点）
        # x方向の偏導関数
        for i in range(1, self.nx - 1):
            for j in range(self.ny):
                f_x[i, j] = (f[i+1, j] - f[i-1, j]) / (2 * self.hx)
        
        # y方向の偏導関数
        for i in range(self.nx):
            for j in range(1, self.ny - 1):
                f_y[i, j] = (f[i, j+1] - f[i, j-1]) / (2 * self.hy)
        
        # x方向の2階偏導関数
        for i in range(1, self.nx - 1):
            for j in range(self.ny):
                f_xx[i, j] = (f[i+1, j] - 2*f[i, j] + f[i-1, j]) / (self.hx**2)
        
        # y方向の2階偏導関数
        for i in range(self.nx):
            for j in range(1, self.ny - 1):
                f_yy[i, j] = (f[i, j+1] - 2*f[i, j] + f[i, j-1]) / (self.hy**2)
        
        # 混合偏導関数
        for i in range(1, self.nx - 1):
            for j in range(1, self.ny - 1):
                f_xy[i, j] = (f[i+1, j+1] - f[i+1, j-1] - f[i-1, j+1] + f[i-1, j-1]) / (4 * self.hx * self.hy)
        
        # 境界値は前方・後方差分で計算（簡略化）
        # x方向境界
        for j in range(self.ny):
            # 左境界
            f_x[0, j] = (-3*f[0, j] + 4*f[1, j] - f[2, j]) / (2 * self.hx)
            # 右境界
            f_x[self.nx-1, j] = (3*f[self.nx-1, j] - 4*f[self.nx-2, j] + f[self.nx-3, j]) / (2 * self.hx)
        
        # y方向境界
        for i in range(self.nx):
            # 下境界
            f_y[i, 0] = (-3*f[i, 0] + 4*f[i, 1] - f[i, 2]) / (2 * self.hy)
            # 上境界
            f_y[i, self.ny-1] = (3*f[i, self.ny-1] - 4*f[i, self.ny-2] + f[i, self.ny-3]) / (2 * self.hy)
        
        print("デバッグソルバーで計算完了")
        return f, f_x, f_y, f_xx, f_xy, f_yy
    
# デバッグ版テストコード
def main():
    # グリッド設定
    nx, ny = 32, 32
    x_range = [-1.0, 1.0]
    y_range = [-1.0, 1.0]
    hx = (x_range[1] - x_range[0]) / (nx - 1)
    hy = (y_range[1] - y_range[0]) / (ny - 1)
    
    # 境界値を設定
    boundary_values = {
        "left": [0.0] * ny,
        "right": [0.0] * ny,
        "bottom": [0.0] * nx,
        "top": [0.0] * nx,
    }
    
    grid_config = GridConfig2D(
        nx_points=nx,
        ny_points=ny,
        hx=hx,
        hy=hy,
        boundary_values=boundary_values,
    )
    
    print(f"グリッド設定: nx={nx}, ny={ny}, hx={hx}, hy={hy}")
    
    # デバッグソルバーの作成
    solver = SimpleDebugSolver2D(grid_config)
    
    # テスト関数の準備
    test_funcs = TestFunction2DFactory.create_standard_functions()
    gaussian = test_funcs[0]  # ガウス関数を選択
    
    # グリッド点
    x = cp.linspace(x_range[0], x_range[1], nx)
    y = cp.linspace(y_range[0], y_range[1], ny)
    
    # テスト関数をグリッド上で評価
    f = cp.zeros((nx, ny))
    f_x_exact = cp.zeros((nx, ny))
    f_y_exact = cp.zeros((nx, ny))
    
    for i in range(nx):
        for j in range(ny):
            f[i, j] = gaussian.f(x[i], y[j])
            f_x_exact[i, j] = gaussian.df_dx(x[i], y[j])
            f_y_exact[i, j] = gaussian.df_dy(x[i], y[j])
    
    print(f"テスト関数: {gaussian.name}")
    print(f"入力関数値の形状: {f.shape}")
    
    # デバッグソルバーで計算
    f_computed, f_x, f_y, f_xx, f_xy, f_yy = solver.solve(f)
    
    # 誤差計算
    error_x = cp.sqrt(cp.mean((f_x - f_x_exact) ** 2))
    error_y = cp.sqrt(cp.mean((f_y - f_y_exact) ** 2))
    
    print(f"x方向微分の誤差 (RMSE): {error_x}")
    print(f"y方向微分の誤差 (RMSE): {error_y}")
    
    # 結果の可視化
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 10))
        
        # 関数値
        plt.subplot(2, 3, 1)
        plt.imshow(f.get(), cmap='viridis')
        plt.colorbar()
        plt.title('Function f(x,y)')
        
        # x方向偏導関数
        plt.subplot(2, 3, 2)
        plt.imshow(f_x.get(), cmap='coolwarm')
        plt.colorbar()
        plt.title('df/dx')
        
        # y方向偏導関数
        plt.subplot(2, 3, 3)
        plt.imshow(f_y.get(), cmap='coolwarm')
        plt.colorbar()
        plt.title('df/dy')
        
        # x方向2階偏導関数
        plt.subplot(2, 3, 4)
        plt.imshow(f_xx.get(), cmap='coolwarm')
        plt.colorbar()
        plt.title('d²f/dx²')
        
        # 混合偏導関数
        plt.subplot(2, 3, 5)
        plt.imshow(f_xy.get(), cmap='coolwarm')
        plt.colorbar()
        plt.title('d²f/dxdy')
        
        # y方向2階偏導関数
        plt.subplot(2, 3, 6)
        plt.imshow(f_yy.get(), cmap='coolwarm')
        plt.colorbar()
        plt.title('d²f/dy²')
        
        plt.tight_layout()
        plt.savefig('2d_derivatives.png')
        print("可視化結果を '2d_derivatives.png' に保存しました")
        
    except ImportError:
        print("matplotlib がインストールされていないため、可視化をスキップします")

main()