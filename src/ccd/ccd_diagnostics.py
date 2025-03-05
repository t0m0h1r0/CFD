"""
シンプル化されたCCD法ソルバーの診断モジュール

行列の基本的な特性を診断する機能を提供します。
"""

import cupy as cp
import numpy as np
from typing import Type, Dict, Any, Optional

from grid_config import GridConfig
from matrix_builder import CCDLeftHandBuilder
from vector_builder import CCDRightHandBuilder
from ccd_solver import CCDSolver
from test_functions import TestFunctionFactory


class CCDSolverDiagnostics:
    """CCD法ソルバーの診断を行うクラス"""

    def __init__(
        self,
        solver_class: Type[CCDSolver],
        grid_config: GridConfig,
        solver_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            solver_class: 診断対象のCCDソルバークラス
            grid_config: グリッド設定
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
        self.left_builder = CCDLeftHandBuilder()
        self.right_builder = CCDRightHandBuilder()

        # 左辺行列を計算
        self.L = self.left_builder.build_matrix(grid_config)

        # テスト関数の読み込み
        self.test_functions = TestFunctionFactory.create_standard_functions()

    def analyze_boundary_conditions(self) -> Dict[str, Any]:
        """
        境界条件の設定を分析

        Returns:
            境界条件の分析結果
        """
        print("=== 境界条件の分析 ===")

        # 境界条件の状態
        has_dirichlet = self.solver.grid_config.is_dirichlet
        has_neumann = self.solver.grid_config.is_neumann

        print(f"ディリクレ境界条件: {'有効' if has_dirichlet else '無効'}")
        print(f"ノイマン境界条件: {'有効' if has_neumann else '無効'}")

        if has_dirichlet:
            print(
                f"ディリクレ境界値: 左端={self.solver.grid_config.dirichlet_values[0]}, 右端={self.solver.grid_config.dirichlet_values[1]}"
            )

        if has_neumann:
            print(
                f"ノイマン境界値: 左端={self.solver.grid_config.neumann_values[0]}, 右端={self.solver.grid_config.neumann_values[1]}"
            )

        # 係数の情報
        print(f"係数 [a, b, c, d]: {self.solver.grid_config.coeffs}")

        # 右辺ベクトルを作成（ゼロ関数の場合）
        zero_f = cp.zeros(self.grid_config.n_points)
        right_hand = self.right_builder.build_vector(
            self.solver.grid_config,
            zero_f,
        )

        # 各要素のインデックス
        n = self.grid_config.n_points
        depth = 4

        # 境界条件に関連するインデックス
        left_idx = [0, 1, 2, 3]  # 左端の4要素
        right_idx = [(n - 1) * depth + i for i in range(4)]  # 右端の4要素

        # 境界値の表示
        print("\n左端の境界値 (インデックス, 値):")
        for i in left_idx:
            print(f"  [{i}]: {right_hand[i]}")

        print("\n右端の境界値 (インデックス, 値):")
        for i in right_idx:
            print(f"  [{i}]: {right_hand[i]}")

        # 左辺行列の行を分析
        L_test = self.left_builder.build_matrix(
            self.solver.grid_config,
        )

        # 対応する左辺行列の行の表示
        print("\n左端の境界条件に対応する左辺行列の行:")
        for i in left_idx:
            print(
                f"  行 {i}: {cp.around(L_test[i, :8], decimals=3)}..."
            )  # 最初の8要素のみ表示

        print("\n右端の境界条件に対応する左辺行列の行:")
        for i in right_idx:
            start_idx = max(0, i - 8)
            print(f"  行 {i}: ...{cp.around(L_test[i, start_idx : i + 1], decimals=3)}")

        return {
            "has_dirichlet": has_dirichlet,
            "has_neumann": has_neumann,
            "dirichlet_values": self.solver.grid_config.dirichlet_values
            if has_dirichlet
            else None,
            "neumann_values": self.solver.grid_config.neumann_values
            if has_neumann
            else None,
            "left_boundary_values": [float(right_hand[i]) for i in left_idx],
            "right_boundary_values": [float(right_hand[i]) for i in right_idx],
        }

    def test_with_functions(self) -> Dict[str, Any]:
        """
        テスト関数を使用したテストを実行

        Returns:
            テスト結果
        """
        print("\n=== テスト関数によるテスト ===")

        # グリッド設定
        n = self.grid_config.n_points
        h = self.grid_config.h
        xrange = [-1.0, 1.0]  # 計算範囲

        # x座標のグリッド点
        x_points = cp.array([xrange[0] + i * h for i in range(n)])

        results = {}

        # 各テスト関数についてテスト
        for test_func in self.test_functions:
            print(f"\n--- テスト関数: {test_func.name} ---")

            # 関数値と導関数を計算
            f_values = cp.array([test_func.f(x) for x in x_points])
            df_values = cp.array([test_func.df(x) for x in x_points])
            d2f_values = cp.array([test_func.d2f(x) for x in x_points])
            d3f_values = cp.array([test_func.d3f(x) for x in x_points])

            # 境界条件の値を計算
            dirichlet_values = [f_values[0], f_values[-1]]
            neumann_values = [df_values[0], df_values[-1]]

            # グリッド設定を更新
            test_grid_config = GridConfig(
                n_points=n,
                h=h,
                dirichlet_values=dirichlet_values,
                neumann_values=neumann_values,
                coeffs=self.coeffs,  # 係数も設定
            )

            # 右辺ベクトルを構築
            a, b, c, d = self.coeffs

            # 設定された係数に基づいて右辺関数値を計算
            rhs_values = a * f_values + b * df_values + c * d2f_values + d * d3f_values

            # 右辺ベクトルを構築
            rhs = self.right_builder.build_vector(
                test_grid_config,
                rhs_values,
            )

            # 左辺行列を構築
            L_test = self.left_builder.build_matrix(
                test_grid_config,
            )

            # 境界条件に関連するインデックス
            depth = 4
            left_idx = [0, 1, 2, 3]  # 左端の4要素
            right_idx = [(n - 1) * depth + i for i in range(4)]  # 右端の4要素

            # ディリクレ条件の位置を特定
            dirichlet_row_left = -1
            dirichlet_row_right = -1

            # 左右の境界でディリクレ条件を探す
            for i in left_idx:
                if cp.all(L_test[i, :4] == cp.array([1, 0, 0, 0])):
                    dirichlet_row_left = i
                    break

            for i in right_idx:
                if cp.all(L_test[i, i - 3 : i + 1] == cp.array([0, 0, 0, 1])):
                    dirichlet_row_right = i
                    break

            # 境界値の表示
            print("\n左端の境界値 (インデックス, 値):")
            for i in left_idx:
                marker = " *" if i == dirichlet_row_left else ""
                print(f"  [{i}]: {rhs[i]}{marker}")

            print("\n右端の境界値 (インデックス, 値):")
            for i in right_idx:
                marker = " *" if i == dirichlet_row_right else ""
                print(f"  [{i}]: {rhs[i]}{marker}")

            # 詳細分析: 関数値とディリクレ境界条件の関係
            print("\n境界条件と関数値の関係:")
            print(f"  関数 f の左端での値: {f_values[0]}")
            print(f"  関数 f の右端での値: {f_values[-1]}")
            print(f"  ディリクレ境界値（左端）: {dirichlet_values[0]}")
            print(f"  ディリクレ境界値（右端）: {dirichlet_values[1]}")

            if dirichlet_row_left >= 0:
                print(
                    f"  右辺ベクトルでの値（左端、インデックス{dirichlet_row_left}）: {rhs[dirichlet_row_left]}"
                )
            if dirichlet_row_right >= 0:
                print(
                    f"  右辺ベクトルでの値（右端、インデックス{dirichlet_row_right}）: {rhs[dirichlet_row_right]}"
                )

            # ディリクレ行の表示
            if dirichlet_row_left >= 0:
                print(f"\n左端ディリクレ条件（行 {dirichlet_row_left}）:")
                print(f"  {cp.around(L_test[dirichlet_row_left, :8], decimals=3)}...")

            if dirichlet_row_right >= 0:
                print(f"\n右端ディリクレ条件（行 {dirichlet_row_right}）:")
                start_idx = max(0, dirichlet_row_right - 7)
                print(
                    f"  ...{cp.around(L_test[dirichlet_row_right, start_idx : dirichlet_row_right + 1], decimals=3)}"
                )

            # 係数の影響分析
            print(f"\n係数 [a={a}, b={b}, c={c}, d={d}] での右辺関数値:")
            print("  f = a*ψ + b*ψ' + c*ψ'' + d*ψ'''")
            print(
                f"  左端での右辺関数値: {a * f_values[0] + b * df_values[0] + c * d2f_values[0] + d * d3f_values[0]}"
            )
            print(
                f"  右端での右辺関数値: {a * f_values[-1] + b * df_values[-1] + c * d2f_values[-1] + d * d3f_values[-1]}"
            )

            # 結果を保存
            results[test_func.name] = {
                "f_values": np.array(f_values),
                "df_values": np.array(df_values),
                "d2f_values": np.array(d2f_values),
                "d3f_values": np.array(d3f_values),
                "rhs_values": np.array(rhs_values),
                "right_hand_vector": np.array(rhs),
                "left_boundary_values": [float(rhs[i]) for i in left_idx],
                "right_boundary_values": [float(rhs[i]) for i in right_idx],
                "dirichlet_values": dirichlet_values,
                "neumann_values": neumann_values,
                "dirichlet_row_left": int(dirichlet_row_left),
                "dirichlet_row_right": int(dirichlet_row_right),
            }

        return results

    def check_solution_values(self, func_name: str = "Sine") -> Dict[str, Any]:
        """
        ソルバーの解の値をチェック

        Args:
            func_name: 使用するテスト関数の名前

        Returns:
            チェック結果
        """
        print(f"\n=== ソルバーの解の値チェック ({func_name}) ===")

        # テスト用の関数を選択
        test_func = next(
            (f for f in self.test_functions if f.name == func_name),
            self.test_functions[0],
        )

        # グリッド設定
        n = self.grid_config.n_points
        h = self.grid_config.h
        xrange = [-1.0, 1.0]  # 計算範囲

        # x座標のグリッド点
        x_points = cp.array([xrange[0] + i * h for i in range(n)])

        # 関数値と導関数を計算
        f_values = cp.array([test_func.f(x) for x in x_points])
        df_values = cp.array([test_func.df(x) for x in x_points])
        d2f_values = cp.array([test_func.d2f(x) for x in x_points])
        d3f_values = cp.array([test_func.d3f(x) for x in x_points])

        # 境界条件の値を計算
        dirichlet_values = [f_values[0], f_values[-1]]
        neumann_values = [df_values[0], df_values[-1]]

        # グリッド設定を更新
        test_grid_config = GridConfig(
            n_points=n,
            h=h,
            dirichlet_values=dirichlet_values,
            neumann_values=neumann_values,
            coeffs=self.coeffs,  # 係数も設定
        )

        print(f"テスト関数: {test_func.name}")
        print(
            f"ディリクレ境界値: 左端={dirichlet_values[0]}, 右端={dirichlet_values[1]}"
        )

        # 係数に基づいて右辺関数値を計算
        a, b, c, d = self.coeffs
        rhs_values = a * f_values + b * df_values + c * d2f_values + d * d3f_values

        # ソルバーを初期化
        solver_params = {}  # coeffsはgrid_configに含まれているため不要
        solver = self.solver_class(test_grid_config, **solver_params)

        # 解を計算
        psi, psi_prime, psi_second, psi_third = solver.solve(rhs_values)

        # 解の境界値を確認
        print("\n解の境界値:")
        print(f"  ψの左端での値: {psi[0]}")
        print(f"  ψの右端での値: {psi[-1]}")
        print(f"  ψ'の左端での値: {psi_prime[0]}")
        print(f"  ψ'の右端での値: {psi_prime[-1]}")

        # 解析解との比較
        print("\n解析解との比較:")
        print(f"  解析解 ψの左端での値: {f_values[0]}")
        print(f"  解析解 ψの右端での値: {f_values[-1]}")
        print(f"  解析解 ψ'の左端での値: {df_values[0]}")
        print(f"  解析解 ψ'の右端での値: {df_values[-1]}")

        # 係数に応じて真の解が異なる場合の処理
        if a == 0 and c == 1:  # f = ψ''
            print("\n特殊係数の場合の解析解:")
            if func_name == "Sine":
                true_solution = lambda x: -cp.sin(cp.pi * x) / (cp.pi**2)
            elif func_name == "Cosine":
                true_solution = lambda x: -cp.cos(2 * cp.pi * x) / (4 * cp.pi**2)
            elif func_name == "QuadPoly":
                # f = (1 - x^2) -> ψ = x^4/12 - x^2/2 + C
                # 境界条件から定数C = 11/12 を決定
                true_solution = lambda x: x**4 / 12 - x**2 / 2 + 11 / 12
            else:
                true_solution = None

            if true_solution:
                true_values = cp.array([true_solution(x) for x in x_points])
                print(f"  特殊解 ψの左端での値: {true_values[0]}")
                print(f"  特殊解 ψの右端での値: {true_values[-1]}")
                # 誤差の計算
                error = cp.sqrt(cp.mean((psi - true_values) ** 2))
                print(f"  特殊解との誤差 (RMSE): {error}")

        # 境界条件との一致を確認
        left_match = abs(psi[0] - dirichlet_values[0]) < 1e-10
        right_match = abs(psi[-1] - dirichlet_values[1]) < 1e-10

        print("\nディリクレ境界条件との一致:")
        print(
            f"  左端: {'一致' if left_match else '不一致'} (差={psi[0] - dirichlet_values[0]})"
        )
        print(
            f"  右端: {'一致' if right_match else '不一致'} (差={psi[-1] - dirichlet_values[1]})"
        )

        # 結果を辞書に格納
        return {
            "function_name": test_func.name,
            "psi_values": np.array(psi),
            "psi_prime_values": np.array(psi_prime),
            "dirichlet_values": dirichlet_values,
            "left_boundary_match": left_match,
            "right_boundary_match": right_match,
            "left_boundary_difference": float(psi[0] - dirichlet_values[0]),
            "right_boundary_difference": float(psi[-1] - dirichlet_values[1]),
        }

    def perform_diagnosis(
        self, visualize: bool = False, test_func_name: str = "Sine"
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
        print(f"\n========== {solver_name}の診断 ==========")

        # 境界条件の設定情報を表示
        has_dirichlet = self.solver.grid_config.is_dirichlet
        has_neumann = self.solver.grid_config.is_neumann

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

        print("\n========== 診断完了 ==========")

        # 結果を統合
        results = {
            "solver_name": solver_name,
            "grid_points": self.grid_config.n_points,
            "grid_spacing": self.grid_config.h,
            "boundary_properties": boundary_props,
            "function_tests": function_tests,
            "solution_check": solution_check,
        }

        return results
