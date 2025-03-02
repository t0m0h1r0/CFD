"""
シンプル化されたCCD法ソルバーの診断モジュール

行列の基本的な特性を診断する機能を提供します。
"""

import jax.numpy as jnp
import os
from typing import Type, Dict, Any, Optional

from ccd_core import GridConfig, CCDLeftHandBuilder
from ccd_solver import CCDSolver
from visualization import visualize_matrix_properties


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

        # 左辺行列を計算
        coeffs = solver_kwargs.get("coeffs", None)
        self.L = self.left_builder.build_matrix(boundary_grid_config, coeffs)

    def analyze_matrix_properties(self, visualize: bool = False) -> Dict[str, float]:
        """
        行列の基本特性を分析

        Args:
            visualize: 可視化するかどうか

        Returns:
            行列特性の辞書
        """
        print("=== 行列の基本特性 ===")
        print(f"左辺行列Lのサイズ: {self.L.shape}")

        # 条件数
        cond_L = jnp.linalg.cond(self.L)
        print(f"左辺行列Lの条件数: {cond_L:.2e}")

        # 特異値
        s = jnp.linalg.svd(self.L, compute_uv=False)
        print(f"最大特異値: {s[0]:.2e}")
        print(f"最小特異値: {s[-1]:.2e}")
        print(f"特異値の比率(最大/最小): {s[0] / s[-1]:.2e}")

        # スケーリング後の特性（CompositesolverにはL_scaledとtransformerが存在する）
        if hasattr(self.solver, "transformer") and hasattr(
            self.solver.transformer, "L_scaled"
        ):
            print("\n=== スケーリング後の行列特性 ===")
            L_scaled = self.solver.transformer.L_scaled
            cond_L_scaled = jnp.linalg.cond(L_scaled)
            print(f"スケーリング後の条件数: {cond_L_scaled:.2e}")
            print(f"条件数の改善率: {cond_L / cond_L_scaled:.2f}倍")

            if visualize:
                os.makedirs("results", exist_ok=True)
                visualize_matrix_properties(
                    L_scaled,
                    f"スケーリング後の行列特性 ({self.solver_name})",
                    "results/scaled_matrix.png",
                )

        # 元の行列を可視化
        if visualize:
            os.makedirs("results", exist_ok=True)
            visualize_matrix_properties(
                self.L, "元の行列特性", "results/original_matrix.png"
            )

        # 結果を辞書に格納して返す
        results = {
            "condition_number": float(cond_L),
            "max_singular": float(s[0]),
            "min_singular": float(s[-1]),
            "singular_ratio": float(s[0] / s[-1]),
        }

        if hasattr(self.solver, "transformer") and hasattr(
            self.solver.transformer, "L_scaled"
        ):
            results.update(
                {
                    "scaled_condition_number": float(cond_L_scaled),
                    "condition_improvement": float(cond_L / cond_L_scaled),
                }
            )

        return results

    def analyze_structure(self) -> Dict[str, Any]:
        """
        行列構造の基本的な特性を分析

        Returns:
            構造特性の辞書
        """
        print("=== 行列構造の分析 ===")

        # スパース性の確認
        zeros = jnp.sum(jnp.abs(self.L) < 1e-10)
        sparsity = zeros / self.L.size
        print(f"疎行列度: {sparsity:.2%}")

        # 対角優位性の確認
        diag = jnp.abs(jnp.diag(self.L))
        off_diag_max = jnp.zeros_like(diag)

        for i in range(self.L.shape[0]):
            row = jnp.abs(self.L[i])
            row = row.at[i].set(0)  # 対角要素を除外
            off_diag_max = off_diag_max.at[i].set(jnp.max(row))

        diag_dominant = jnp.all(diag > off_diag_max)
        print(f"対角優位性: {'あり' if diag_dominant else 'なし'}")

        return {"sparsity": float(sparsity), "diag_dominant": bool(diag_dominant)}

    def perform_diagnosis(self, visualize: bool = False) -> Dict[str, Any]:
        """
        基本的な診断を実行

        Args:
            visualize: 可視化を行うかどうか

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
        if hasattr(self.solver, "coeffs"):
            print(f"係数: {self.solver.coeffs}")

        print("\n1. 行列の基本特性")
        print("-" * 40)
        matrix_props = self.analyze_matrix_properties(visualize)

        print("\n2. 行列構造の確認")
        print("-" * 40)
        structure_props = self.analyze_structure()

        print("\n========== 診断完了 ==========")

        # 境界条件情報
        boundary_info = {"has_dirichlet": has_dirichlet, "has_neumann": has_neumann}

        if hasattr(self.solver, "coeffs"):
            boundary_info["coeffs"] = self.solver.coeffs

        # 結果を統合
        results = {
            "solver_name": solver_name,
            "grid_points": self.grid_config.n_points,
            "grid_spacing": self.grid_config.h,
            "boundary_info": boundary_info,
            "matrix_properties": matrix_props,
            "structure_properties": structure_props,
        }

        return results
