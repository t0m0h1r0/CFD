"""
2次元CCDソルバーの使用例
"""

import cupy as cp
import matplotlib.pyplot as plt

from grid_config_2d import GridConfig2D
from ccd_solver_2d import CCDSolver2D
from composite_solver_2d import CCDCompositeSolver2D
from test_functions_2d import TestFunction2DFactory


def main():
    # グリッド設定
    nx, ny = 32, 32
    x_range = [-1.0, 1.0]
    y_range = [-1.0, 1.0]
    hx = (x_range[1] - x_range[0]) / (nx - 1)
    hy = (y_range[1] - y_range[0]) / (ny - 1)
    
    grid_config = GridConfig2D(
        nx_points=nx,
        ny_points=ny,
        hx=hx,
        hy=hy,
    )
    
    # ソルバーの作成
    solver = CCDCompositeSolver2D(
        grid_config,
        scaling="equalization",
        regularization="tikhonov",
    )
    
    # テスト関数の準備
    test_funcs = TestFunction2DFactory.create_standard_functions()
    gaussian = test_funcs[0]  # ガウス関数を選択
    
    # グリッド点
    x = cp.linspace(x_range[0], x_range[1], nx)
    y = cp.linspace(y_range[0], y_range[1], ny)
    
    # テスト関数をグリッド上で評価
    f, f_x_exact, f_y_exact, f_xx_exact, f_xy_exact, f_yy_exact = (
        TestFunction2DFactory.evaluate_on_grid(gaussian, x, y)
    )
    
    # CCDソルバーで偏導関数を計算
    f_computed, f_x, f_y, f_xx, f_xy, f_yy = solver.solve(f)
    
    # 結果の可視化（例：x方向の1階偏導関数）
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(f_x_exact.get(), cmap='jet', extent=[*x_range, *y_range])
    plt.colorbar()
    plt.title('Exact df/dx')
    
    plt.subplot(1, 2, 2)
    plt.imshow(f_x.get(), cmap='jet', extent=[*x_range, *y_range])
    plt.colorbar()
    plt.title('Computed df/dx')
    
    plt.tight_layout()
    plt.savefig('ccd_2d_result.png')
    plt.show()
    
    # 誤差の計算と表示
    error_x = cp.sqrt(cp.mean((f_x - f_x_exact) ** 2))
    error_y = cp.sqrt(cp.mean((f_y - f_y_exact) ** 2))
    error_xx = cp.sqrt(cp.mean((f_xx - f_xx_exact) ** 2))
    error_yy = cp.sqrt(cp.mean((f_yy - f_yy_exact) ** 2))
    
    print(f"RMSE for df/dx: {error_x:.2e}")
    print(f"RMSE for df/dy: {error_y:.2e}")
    print(f"RMSE for d²f/dx²: {error_xx:.2e}")
    print(f"RMSE for d²f/dy²: {error_yy:.2e}")


if __name__ == "__main__":
    main()
