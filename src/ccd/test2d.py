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
    
    print(f"行列サイズ: {nx*ny*4}x{nx*ny*4}")
    
    # ソルバーの作成（デバッグのため正則化を使わない）
    solver = CCDCompositeSolver2D(
        grid_config,
        scaling="none",  # まずはシンプルな設定でテスト
        regularization="none",
    )
    
    # テスト関数の準備
    test_funcs = TestFunction2DFactory.create_standard_functions()
    test_func = test_funcs[0]  # ガウス関数を選択
    
    # グリッド点
    x = cp.linspace(x_range[0], x_range[1], nx)
    y = cp.linspace(y_range[0], y_range[1], ny)
    
    # テスト関数をグリッド上で評価（まず関数値のみ）
    f = cp.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            f[i, j] = test_func.f(x[i], y[j])
    
    print(f"入力関数値の形状: {f.shape}")
    
    # CCDソルバーで偏導関数を計算
    try:
        f_computed, f_x, f_y, f_xx, f_xy, f_yy = solver.solve(f)
        print("計算成功！")
    except Exception as e:
        print(f"エラー発生: {e}")
        # より詳細なデバッグ情報
        import traceback
        traceback.print_exc()
        return
    
    # 結果の可視化と検証は成功したら実装


if __name__ == "__main__":
    main()
