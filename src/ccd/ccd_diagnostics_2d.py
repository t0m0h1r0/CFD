"""
シンプル化された2次元CCD法ソルバーの診断モジュール

2次元行列の基本的な特性を診断する機能を提供します。
"""

import cupy as cp
import numpy as np
from typing import Type, Dict, Any, Optional, Tuple, List

from grid_config_2d import GridConfig2D
from matrix_builder_2d import CCDLeftHandBuilder2D
from vector_builder_2d import CCDRightHandBuilder2D
from composite_solver_2d import CCDCompositeSolver2D
from test_functions_2d import TestFunction2DFactory
from visualization_2d import visualize_2d_field, visualize_vector_field_2d


class CCDSolverDiagnostics2D:
    """2次元CCD法ソルバーの診断を行うクラス"""

    def __init__(
        self,
        solver_class: Type[CCDCompositeSolver2D],
        grid_config: GridConfig2D,
        solver_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            solver_class: 診断対象の2次元CCDソルバークラス
            grid_config: 2次元グリッド設定
            solver_kwargs: ソルバーの初期化パラメータ
        """
        self.grid_config = grid_config
        solver_kwargs = solver_kwargs or {}
        self.solver_class = solver_class
        self.solver_name = solver_class.__name__

        # coeffsをgrid_configに設定
        if "coeffs" in solver_kwargs:
            grid_config.coeffs = solver_kwargs["coeffs"]
            # solver_kwargsからcoeffsを削除（二重設定を避けるため）
            self.coeffs = solver_kwargs.pop("coeffs")
        else:
            # grid_configからcoeffsを参照
            self.coeffs = grid_config.coeffs

        # ソルバーを初期化
        self.solver = solver_class(grid_config, **solver_kwargs)
        self.left_builder = CCDLeftHandBuilder2D()
        self.right_builder = CCDRightHandBuilder2D()

        # 左辺行列を計算
        self.L = self.left_builder.build_matrix(grid_config)

        # テスト関数の読み込み
        self.test_functions = TestFunction2DFactory.create_standard_functions()

    def analyze_boundary_conditions(self) -> Dict[str, Any]:
        """
        境界条件の設定を分析

        Returns:
            境界条件の分析結果
        """
        print("=== 2次元境界条件の分析 ===")

        # 境界条件の状態
        has_dirichlet = self.solver.grid_config.is_dirichlet()
        has_neumann = self.solver.grid_config.is_neumann()

        print(f"ディリクレ境界条件: {'有効' if has_dirichlet else '無効'}")
        print(f"ノイマン境界条件: {'有効' if has_neumann else '無効'}")

        # 境界値の情報を表示
        boundary_values = self.grid_config.boundary_values
        if boundary_values:
            print("\n境界値の統計情報:")
            for edge, values in boundary_values.items():
                if values and len(values) > 0:
                    print(f"  {edge}: min={min(values):.4f}, max={max(values):.4f}, mean={sum(values)/len(values):.4f}")

        # 係数の情報
        print(f"\n係数 [a, b, c, d]: {self.solver.grid_config.coeffs}")

        # 右辺ベクトルを作成（ゼロ関数の場合）
        nx, ny = self.grid_config.nx_points, self.grid_config.ny_points
        zero_f = cp.zeros((nx, ny))
        right_hand = self.right_builder.build_vector(
            self.solver.grid_config,
            zero_f,
        )

        # ベクトルサイズを出力
        print(f"\n右辺ベクトルのサイズ: {right_hand.shape}")
        print(f"左辺行列のサイズ: {self.L.shape}")

        # 行列の特性を出力
        if hasattr(self.L, "nnz"):
            nnz = self.L.nnz
            size = self.L.shape[0] * self.L.shape[1]
            density = nnz / size
            print(f"行列の非ゼロ要素数: {nnz}, 密度: {density:.6f}")

        return {
            "has_dirichlet": has_dirichlet,
            "has_neumann": has_neumann,
            "boundary_values": boundary_values,
            "coeffs": self.solver.grid_config.coeffs,
            "matrix_shape": self.L.shape,
            "vector_shape": right_hand.shape,
        }

    def test_with_functions(self) -> Dict[str, Any]:
        """
        テスト関数を使用したテストを実行

        Returns:
            テスト結果
        """
        print("\n=== 2次元テスト関数によるテスト ===")

        # グリッド設定
        nx, ny = self.grid_config.nx_points, self.grid_config.ny_points
        hx, hy = self.grid_config.hx, self.grid_config.hy
        x_range = [-1.0, 1.0]  # 計算範囲
        y_range = [-1.0, 1.0]  # 計算範囲

        # x座標とy座標のグリッド点
        x = cp.linspace(x_range[0], x_range[1], nx)
        y = cp.linspace(y_range[0], y_range[1], ny)

        results = {}

        # 各テスト関数についてテスト
        for test_func in self.test_functions:
            print(f"\n--- 2次元テスト関数: {test_func.name} ---")

            # 関数値と偏導関数を計算
            f_values = cp.zeros((nx, ny))
            f_x_values = cp.zeros((nx, ny))
            f_y_values = cp.zeros((nx, ny))
            f_xx_values = cp.zeros((nx, ny))
            f_xy_values = cp.zeros((nx, ny))
            f_yy_values = cp.zeros((nx, ny))

            for i in range(nx):
                for j in range(ny):
                    f_values[i, j] = test_func.f(x[i], y[j])
                    f_x_values[i, j] = test_func.df_dx(x[i], y[j])
                    f_y_values[i, j] = test_func.df_dy(x[i], y[j])
                    f_xx_values[i, j] = test_func.d2f_dx2(x[i], y[j])
                    f_xy_values[i, j] = test_func.d2f_dxdy(x[i], y[j])
                    f_yy_values[i, j] = test_func.d2f_dy2(x[i], y[j])

            # 境界値の抽出（単純化のため角部分は使わない）
            left_values = f_values[0, 1:-1]
            right_values = f_values[-1, 1:-1]
            bottom_values = f_values[1:-1, 0]
            top_values = f_values[1:-1, -1]

            # 境界勾配値の抽出
            left_x_grad = f_x_values[0, 1:-1]
            right_x_grad = f_x_values[-1, 1:-1]
            bottom_y_grad = f_y_values[1:-1, 0]
            top_y_grad = f_y_values[1:-1, -1]

            # 境界値の統計を出力
            print("\n境界値の統計:")
            print(f"  左辺平均: {cp.mean(left_values):.4f}, 右辺平均: {cp.mean(right_values):.4f}")
            print(f"  下辺平均: {cp.mean(bottom_values):.4f}, 上辺平均: {cp.mean(top_values):.4f}")
            
            # 勾配の統計を出力
            print("\n境界勾配の統計:")
            print(f"  左辺x勾配平均: {cp.mean(left_x_grad):.4f}, 右辺x勾配平均: {cp.mean(right_x_grad):.4f}")
            print(f"  下辺y勾配平均: {cp.mean(bottom_y_grad):.4f}, 上辺y勾配平均: {cp.mean(top_y_grad):.4f}")

            # 入力関数値の設定
            a, b, c, d = self.coeffs
            
            # 係数に基づいて右辺関数値を計算
            rhs_values = a * f_values + b * f_x_values + c * f_xx_values + d * f_yy_values
            
            # 入力関数値の統計を出力
            print(f"\n入力関数値の統計:")
            print(f"  最小値: {cp.min(rhs_values):.4f}, 最大値: {cp.max(rhs_values):.4f}, 平均: {cp.mean(rhs_values):.4f}")

            # 境界条件の設定
            boundary_values = {
                "left": left_values.get().tolist(),
                "right": right_values.get().tolist(),
                "bottom": bottom_values.get().tolist(),
                "top": top_values.get().tolist(),
                "left_neumann": left_x_grad.get().tolist(),
                "right_neumann": right_x_grad.get().tolist(),
                "bottom_neumann": bottom_y_grad.get().tolist(),
                "top_neumann": top_y_grad.get().tolist(),
            }

            # 更新されたグリッド設定を作成
            test_grid_config = GridConfig2D(
                nx_points=nx,
                ny_points=ny,
                hx=hx,
                hy=hy,
                boundary_values=boundary_values,
                coeffs=self.coeffs,
            )

            # 右辺ベクトルを構築
            rhs = self.right_builder.build_vector(
                test_grid_config,
                rhs_values,
            )

            # 右辺ベクトルのサイズを出力
            print(f"\n右辺ベクトルのサイズ: {rhs.shape}")

            # 結果を保存
            results[test_func.name] = {
                "f_values": np.array(f_values.get()),
                "f_x_values": np.array(f_x_values.get()),
                "f_y_values": np.array(f_y_values.get()),
                "f_xx_values": np.array(f_xx_values.get()),
                "f_xy_values": np.array(f_xy_values.get()),
                "f_yy_values": np.array(f_yy_values.get()),
                "boundary_values": boundary_values,
                "rhs_values": np.array(rhs_values.get()),
            }

        return results

    def check_solution_values(self, func_name: str = "Gaussian") -> Dict[str, Any]:
        """
        ソルバーの解の値をチェック

        Args:
            func_name: 使用するテスト関数の名前

        Returns:
            チェック結果
        """
        print(f"\n=== 2次元ソルバーの解の値チェック ({func_name}) ===")

        # テスト用の関数を選択
        test_func = next(
            (f for f in self.test_functions if f.name == func_name),
            self.test_functions[0],
        )

        # グリッド設定
        nx, ny = self.grid_config.nx_points, self.grid_config.ny_points
        hx, hy = self.grid_config.hx, self.grid_config.hy
        x_range = [-1.0, 1.0]  # 計算範囲
        y_range = [-1.0, 1.0]  # 計算範囲

        # x座標とy座標のグリッド点
        x = cp.linspace(x_range[0], x_range[1], nx)
        y = cp.linspace(y_range[0], y_range[1], ny)

        # 関数値と偏導関数を計算
        f_values = cp.zeros((nx, ny))
        f_x_values = cp.zeros((nx, ny))
        f_y_values = cp.zeros((nx, ny))
        f_xx_values = cp.zeros((nx, ny))
        f_xy_values = cp.zeros((nx, ny))
        f_yy_values = cp.zeros((nx, ny))

        for i in range(nx):
            for j in range(ny):
                f_values[i, j] = test_func.f(x[i], y[j])
                f_x_values[i, j] = test_func.df_dx(x[i], y[j])
                f_y_values[i, j] = test_func.df_dy(x[i], y[j])
                f_xx_values[i, j] = test_func.d2f_dx2(x[i], y[j])
                f_xy_values[i, j] = test_func.d2f_dxdy(x[i], y[j])
                f_yy_values[i, j] = test_func.d2f_dy2(x[i], y[j])

        # 境界値を設定
        left_values = f_values[0, :].tolist()
        right_values = f_values[-1, :].tolist()
        bottom_values = f_values[:, 0].tolist()
        top_values = f_values[:, -1].tolist()

        boundary_values = {
            "left": left_values,
            "right": right_values,
            "bottom": bottom_values,
            "top": top_values,
        }

        # 更新されたグリッド設定を作成
        test_grid_config = GridConfig2D(
            nx_points=nx,
            ny_points=ny,
            hx=hx,
            hy=hy,
            boundary_values=boundary_values,
            coeffs=self.coeffs,
        )

        print(f"テスト関数: {test_func.name}")
        print(f"グリッドサイズ: {nx}x{ny}, 間隔: hx={hx}, hy={hy}")

        # 入力関数値の設定
        a, b, c, d = self.coeffs
        rhs_values = a * f_values + b * f_x_values + c * f_xx_values + d * f_yy_values

        # ソルバーを初期化
        solver_params = {}  # coeffsはgrid_configに含まれているため不要
        solver = self.solver_class(test_grid_config, **solver_params)

        # 解を計算
        f_computed, f_x, f_y, f_xx, f_xy, f_yy = solver.solve(rhs_values)

        # 解の統計情報を出力
        print("\n解の統計情報:")
        print(f"  f - 最小値: {cp.min(f_computed):.4f}, 最大値: {cp.max(f_computed):.4f}, 平均: {cp.mean(f_computed):.4f}")
        print(f"  f_x - 最小値: {cp.min(f_x):.4f}, 最大値: {cp.max(f_x):.4f}, 平均: {cp.mean(f_x):.4f}")
        print(f"  f_y - 最小値: {cp.min(f_y):.4f}, 最大値: {cp.max(f_y):.4f}, 平均: {cp.mean(f_y):.4f}")

        # 解析解との比較
        print("\n解析解との比較 (RMSE):")
        rmse_f = cp.sqrt(cp.mean((f_computed - f_values) ** 2))
        rmse_fx = cp.sqrt(cp.mean((f_x - f_x_values) ** 2))
        rmse_fy = cp.sqrt(cp.mean((f_y - f_y_values) ** 2))
        rmse_fxx = cp.sqrt(cp.mean((f_xx - f_xx_values) ** 2))
        rmse_fxy = cp.sqrt(cp.mean((f_xy - f_xy_values) ** 2))
        rmse_fyy = cp.sqrt(cp.mean((f_yy - f_yy_values) ** 2))

        print(f"  f RMSE: {rmse_f:.4e}")
        print(f"  f_x RMSE: {rmse_fx:.4e}")
        print(f"  f_y RMSE: {rmse_fy:.4e}")
        print(f"  f_xx RMSE: {rmse_fxx:.4e}")
        print(f"  f_xy RMSE: {rmse_fxy:.4e}")
        print(f"  f_yy RMSE: {rmse_fyy:.4e}")

        # 境界条件との一致を確認
        # 左境界
        left_diff = cp.max(cp.abs(f_computed[0, :] - cp.array(left_values)))
        # 右境界
        right_diff = cp.max(cp.abs(f_computed[-1, :] - cp.array(right_values)))
        # 下境界
        bottom_diff = cp.max(cp.abs(f_computed[:, 0] - cp.array(bottom_values)))
        # 上境界
        top_diff = cp.max(cp.abs(f_computed[:, -1] - cp.array(top_values)))

        print("\n境界条件との最大差異:")
        print(f"  左境界: {left_diff:.4e}")
        print(f"  右境界: {right_diff:.4e}")
        print(f"  下境界: {bottom_diff:.4e}")
        print(f"  上境界: {top_diff:.4e}")

        # 結果を辞書に格納
        return {
            "function_name": test_func.name,
            "f_values": np.array(f_computed.get()),
            "f_x_values": np.array(f_x.get()),
            "f_y_values": np.array(f_y.get()),
            "f_xx_values": np.array(f_xx.get()),
            "f_xy_values": np.array(f_xy.get()),
            "f_yy_values": np.array(f_yy.get()),
            "boundary_diff": {
                "left": float(left_diff),
                "right": float(right_diff),
                "bottom": float(bottom_diff),
                "top": float(top_diff),
            },
            "rmse": {
                "f": float(rmse_f),
                "f_x": float(rmse_fx),
                "f_y": float(rmse_fy),
                "f_xx": float(rmse_fxx),
                "f_xy": float(rmse_fxy),
                "f_yy": float(rmse_fyy),
            }
        }

    def perform_diagnosis(
        self, visualize: bool = False, test_func_name: str = "Gaussian"
    ) -> Dict[str, Any]:
        """
        基本的な診断を実行

        Args:
            visualize: 可視化を行うかどうか
            test_func_name: 個別テストに使用するテスト関数の名前

        Returns:
            診断結果の辞書
        """
        solver_name = self.solver_class.__name__
        print(f"\n========== 2次元{solver_name}の診断 ==========")

        # 境界条件の設定情報を表示
        has_dirichlet = self.solver.grid_config.is_dirichlet()
        has_neumann = self.solver.grid_config.is_neumann()

        print(
            f"境界条件: {'ディリクレ' if has_dirichlet else ''}{'と' if has_dirichlet and has_neumann else ''}{'ノイマン' if has_neumann else ''}"
        )
        print(f"係数: {self.solver.grid_config.coeffs}")

        # 境界条件の分析
        print("\n境界条件の分析")
        print("-" * 40)
        boundary_props = self.analyze_boundary_conditions()

        # テスト関数でのテスト
        print("\nテスト関数でのテスト")
        print("-" * 40)
        function_tests = self.test_with_functions()

        # ソルバーの解のチェック
        print("\nソルバーの解のチェック")
        print("-" * 40)
        solution_check = self.check_solution_values(func_name=test_func_name)

        # 可視化
        if visualize:
            # テスト関数を取得
            test_func = next(
                (f for f in self.test_functions if f.name == test_func_name),
                self.test_functions[0],
            )
            
            # グリッド設定
            nx, ny = self.grid_config.nx_points, self.grid_config.ny_points
            x_range = [-1.0, 1.0]
            y_range = [-1.0, 1.0]
            
            # グリッド点を計算
            x = cp.linspace(x_range[0], x_range[1], nx)
            y = cp.linspace(y_range[0], y_range[1], ny)
            
            # 関数値と偏導関数を計算
            f_values = cp.zeros((nx, ny))
            f_x_values = cp.zeros((nx, ny))
            f_y_values = cp.zeros((nx, ny))
            
            for i in range(nx):
                for j in range(ny):
                    f_values[i, j] = test_func.f(x[i], y[j])
                    f_x_values[i, j] = test_func.df_dx(x[i], y[j])
                    f_y_values[i, j] = test_func.df_dy(x[i], y[j])
            
            # 入力関数値を可視化
            visualize_2d_field(
                f_values, 
                self.grid_config,
                (x_range, y_range),
                f"Input Function ({test_func_name})",
                save_path=f"results/2d_diag_{test_func_name.lower()}_input.png"
            )
            
            # 勾配場を可視化
            visualize_vector_field_2d(
                f_x_values,
                f_y_values,
                self.grid_config,
                (x_range, y_range),
                f"Gradient Field ({test_func_name})",
                save_path=f"results/2d_diag_{test_func_name.lower()}_gradient.png"
            )
            
            # ソルバーの結果を可視化
            f_computed = cp.array(solution_check["f_values"])
            f_x_computed = cp.array(solution_check["f_x_values"])
            f_y_computed = cp.array(solution_check["f_y_values"])
            
            visualize_2d_field(
                f_computed, 
                self.grid_config,
                (x_range, y_range),
                f"Computed Function ({test_func_name})",
                save_path=f"results/2d_diag_{test_func_name.lower()}_computed.png"
            )
            
            visualize_vector_field_2d(
                f_x_computed,
                f_y_computed,
                self.grid_config,
                (x_range, y_range),
                f"Computed Gradient Field ({test_func_name})",
                save_path=f"results/2d_diag_{test_func_name.lower()}_computed_gradient.png"
            )

        print("\n========== 診断完了 ==========")

        # 結果を統合
        results = {
            "solver_name": solver_name,
            "grid_config": {
                "nx_points": self.grid_config.nx_points,
                "ny_points": self.grid_config.ny_points,
                "hx": self.grid_config.hx,
                "hy": self.grid_config.hy,
                "coeffs": self.grid_config.coeffs,
            },
            "boundary_properties": boundary_props,
            "solution_check": solution_check,
        }

        return results