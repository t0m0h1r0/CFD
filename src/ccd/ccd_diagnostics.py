"""
シンプル化されたCCD法ソルバーの診断モジュール

行列の基本的な特性と診断情報を提供します。
"""

import jax.numpy as jnp
import numpy as np
from typing import Type, Dict, Any, Optional, List

from ccd_core import GridConfig, CCDLeftHandBuilder, CCDRightHandBuilder
from ccd_solver import CCDSolver
from test_functions import TestFunctionFactory, TestFunction, TestFunctionFactory


class CCDSolverDiagnostics:
    """CCD法ソルバーの診断を行うクラス"""

    def __init__(
        self,
        solver_class: Type[CCDSolver],
        grid_config: GridConfig,
        solver_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        初期化

        Args:
            solver_class: 診断対象のCCDソルバークラス
            grid_config: グリッド設定
            solver_kwargs: ソルバーの初期化パラメータ
        """
        self.grid_config = grid_config
        solver_kwargs = solver_kwargs or {}
        self.solver_class = solver_class
        self.solver_name = solver_class.__name__

        # 標準的な境界条件を持つグリッド設定を作成
        boundary_grid_config = GridConfig(
            n_points=grid_config.n_points,
            h=grid_config.h,
            dirichlet_values=[0.0, 0.0],
            neumann_values=[0.0, 0.0],
        )

        # ソルバーを初期化
        self.solver = solver_class(boundary_grid_config, **solver_kwargs)
        self.left_builder = CCDLeftHandBuilder()
        self.right_builder = CCDRightHandBuilder()

        # 係数を設定
        self.coeffs = solver_kwargs.get("coeffs", None)

        # 左辺行列を計算
        self.L = self.left_builder.build_matrix(boundary_grid_config, self.coeffs)

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

        boundary_info = {
            "has_dirichlet": has_dirichlet,
            "has_neumann": has_neumann,
            "dirichlet_values": self.solver.grid_config.dirichlet_values
            if has_dirichlet
            else None,
            "neumann_values": self.solver.grid_config.neumann_values
            if has_neumann
            else None,
        }

        if has_dirichlet:
            print(
                f"ディリクレ境界値: 左端={boundary_info['dirichlet_values'][0]}, 右端={boundary_info['dirichlet_values'][1]}"
            )

        if has_neumann:
            print(
                f"ノイマン境界値: 左端={boundary_info['neumann_values'][0]}, 右端={boundary_info['neumann_values'][1]}"
            )

        # 係数の情報
        if hasattr(self.solver, "coeffs"):
            coeffs = self.solver.coeffs
            print(f"係数 [a, b, c, d]: {coeffs}")
            boundary_info["coefficients"] = coeffs

        return boundary_info

    def analyze_matrix_structure(self) -> Dict[str, Any]:
        """
        左辺行列の構造と境界ブロック行列を分析

        Returns:
            行列構造の分析結果
        """
        print("\n=== 左辺行列の構造分析 ===")

        # 境界条件の状態
        dirichlet_enabled = self.solver.grid_config.is_dirichlet
        neumann_enabled = self.solver.grid_config.is_neumann

        # 内部ブロック行列を取得
        A, B, C = self.left_builder._build_interior_blocks(self.coeffs)

        # 境界ブロック行列を取得
        B0, C0, D0, ZR, AR, BR = self.left_builder._build_boundary_blocks(
            self.coeffs,
            dirichlet_enabled=dirichlet_enabled,
            neumann_enabled=neumann_enabled,
        )

        # 左辺行列全体の構造情報
        L_size = self.L.shape[0]
        L_nnz = jnp.count_nonzero(self.L)
        L_density = L_nnz / (L_size * L_size)

        # 特異値分解の計算
        U, s, Vh = jnp.linalg.svd(self.L, full_matrices=False)

        print("\n行列の基本特性:")
        print(f"サイズ: {L_size}x{L_size}")
        print(f"非ゼロ要素数: {L_nnz}")
        print(f"行列密度: {L_density:.6f}")

        print("\n特異値分析:")
        print(f"最大特異値: {s[0]}")
        print(f"最小特異値: {s[-1]}")
        print(f"条件数: {s[0] / s[-1]}")

        # 境界ブロック行列の詳細出力
        print("\n左境界ブロック行列 B0:")
        print(B0)
        print("\nB0の特性:")
        B0_det = jnp.linalg.det(B0)
        B0_cond = jnp.linalg.cond(B0)
        B0_rank = jnp.linalg.matrix_rank(B0)
        print(f"  行列式: {B0_det}")
        print(f"  条件数: {B0_cond}")
        print(f"  ランク: {B0_rank}")

        print("\n右境界ブロック行列 BR:")
        print(BR)
        print("\nBRの特性:")
        BR_det = jnp.linalg.det(BR)
        BR_cond = jnp.linalg.cond(BR)
        BR_rank = jnp.linalg.matrix_rank(BR)
        print(f"  行列式: {BR_det}")
        print(f"  条件数: {BR_cond}")
        print(f"  ランク: {BR_rank}")

        matrix_info = {
            "L_size": int(L_size),
            "L_nnz": int(L_nnz),
            "L_density": float(L_density),
            "singularValues": {
                "max": float(s[0]),
                "min": float(s[-1]),
                "condition_number": float(s[0] / s[-1]),
            },
            "matrix_blocks": {
                "A": A.tolist(),
                "B": B.tolist(),
                "C": C.tolist(),
                "B0": B0.tolist(),
                "C0": C0.tolist(),
                "D0": D0.tolist(),
                "ZR": ZR.tolist(),
                "AR": AR.tolist(),
                "BR": BR.tolist(),
            },
            "boundary_blocks_properties": {
                "B0": {
                    "determinant": float(B0_det),
                    "condition_number": float(B0_cond),
                    "rank": int(B0_rank),
                },
                "BR": {
                    "determinant": float(BR_det),
                    "condition_number": float(BR_cond),
                    "rank": int(BR_rank),
                },
            },
        }

        return matrix_info

    def test_with_function(self, test_func: TestFunction) -> Dict[str, Any]:
        """
        単一のテスト関数に対してテストを実行

        Args:
            test_func: テスト関数

        Returns:
            テスト結果の辞書
        """
        print(f"\n--- テスト関数: {test_func.name} ---")

        # グリッド設定
        n = self.grid_config.n_points
        h = self.grid_config.h
        xrange = [-1.0, 1.0]  # 計算範囲

        # x座標のグリッド点
        x_points = jnp.array([xrange[0] + i * h for i in range(n)])

        # 関数値と導関数を計算
        f_values = jnp.array([test_func.f(x) for x in x_points])
        df_values = jnp.array([test_func.df(x) for x in x_points])
        d2f_values = jnp.array([test_func.d2f(x) for x in x_points])
        d3f_values = jnp.array([test_func.d3f(x) for x in x_points])

        # 境界条件の値を計算
        dirichlet_values = [f_values[0], f_values[-1]]
        neumann_values = [df_values[0], df_values[-1]]

        # グリッド設定を更新
        test_grid_config = GridConfig(
            n_points=n,
            h=h,
            dirichlet_values=dirichlet_values,
            neumann_values=neumann_values,
        )

        # 係数に基づいて右辺関数値を計算
        a, b, c, d = self.coeffs if self.coeffs else [1.0, 0.0, 0.0, 0.0]
        rhs_values = a * f_values + b * df_values + c * d2f_values + d * d3f_values

        # 右辺ベクトルを構築
        rhs = self.right_builder.build_vector(
            test_grid_config,
            rhs_values,
            self.coeffs,
            dirichlet_enabled=True,
            neumann_enabled=True,
        )

        # 左辺行列を構築
        L_test = self.left_builder.build_matrix(
            test_grid_config,
            self.coeffs,
            dirichlet_enabled=True,
            neumann_enabled=True,
        )

        # 詳細情報を出力
        print(f"\n係数 [a={a}, b={b}, c={c}, d={d}] での右辺関数値:")
        print("  f = a*ψ + b*ψ' + c*ψ'' + d*ψ'''")
        print(
            f"  左端での右辺関数値: {a * f_values[0] + b * df_values[0] + c * d2f_values[0] + d * d3f_values[0]}"
        )
        print(
            f"  右端での右辺関数値: {a * f_values[-1] + b * df_values[-1] + c * d2f_values[-1] + d * d3f_values[-1]}"
        )

        # 右辺の境界ブロックの詳細出力
        print("\n右辺の境界ブロックの詳細:")
        print("左端の境界ブロック (インデックス):")
        left_idx = [0, 1, 2, 3]  # 左端の4要素
        for i in left_idx:
            print(f"  [{i}]: {rhs[i]}")

        right_idx = [(n - 1) * 4 + i for i in range(4)]  # 右端の4要素
        print("\n右端の境界ブロック (インデックス):")
        for i in right_idx:
            print(f"  [{i}]: {rhs[i]}")

        # 右辺ベクトルの詳細分析
        print("\n右辺ベクトルの分析:")
        print(f"全体の長さ: {len(rhs)}")
        print(f"最大値: {jnp.max(rhs)}")
        print(f"最小値: {jnp.min(rhs)}")
        print(f"平均値: {jnp.mean(rhs)}")
        print(f"標準偏差: {jnp.std(rhs)}")

        # L2ノルムの計算
        rhs_l2_norm = jnp.linalg.norm(rhs)
        print(f"L2ノルム: {rhs_l2_norm}")

        # 非ゼロ要素の数と割合
        rhs_nonzero = jnp.count_nonzero(rhs)
        rhs_nonzero_ratio = rhs_nonzero / len(rhs)
        print(f"非ゼロ要素数: {rhs_nonzero}")
        print(f"非ゼロ要素の割合: {rhs_nonzero_ratio:.4f}")

    def test_with_all_functions(self) -> Dict[str, Any]:
        """
        全てのテスト関数でテストを実行

        Returns:
            テスト結果の辞書
        """
        print("\n=== テスト関数による分析 ===")

        results = {}
        for test_func in self.test_functions:
            results[test_func.name] = self.test_with_function(test_func)

        return results

    def perform_diagnosis(
        self, visualize: bool = False, test_func_name: str = "Sine"
    ) -> Dict[str, Any]:
        """
        基本的な診断を実行

        Args:
            visualize: 可視化フラグ（現在未使用）
            test_func_name: 診断時に参照するテスト関数の名前

        Returns:
            診断結果の辞書
        """
        print(f"\n========== {self.solver_name}の診断 ==========")

        # 境界条件の設定情報を取得
        boundary_props = self.analyze_boundary_conditions()

        # 行列構造の分析
        matrix_props = self.analyze_matrix_structure()

        # テスト関数による分析
        function_tests = self.test_with_all_functions()

        # 結果を統合
        results = {
            "solver_name": self.solver_name,
            "grid_points": self.grid_config.n_points,
            "grid_spacing": self.grid_config.h,
            "boundary_properties": boundary_props,
            "matrix_properties": matrix_props,
            "function_tests": function_tests,
        }

        # 可視化フラグは現在未使用だが、対応のために追加
        if visualize:
            print("可視化は現在サポートされていません。")

        return results
