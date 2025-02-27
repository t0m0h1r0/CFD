"""
CCD法ソルバーの診断モジュール

行列の特性や数値安定性などを診断する機能を提供します。
"""

import jax.numpy as jnp
import os
from typing import Type, Dict, Any, Optional

from ccd_core import GridConfig, LeftHandBlockBuilder
from ccd_solver import CCDSolver
from visualization import visualize_matrix_properties


class CCDSolverDiagnostics:
    """CCD法ソルバーの診断を行うクラス"""
    
    def __init__(
        self, 
        solver_class: Type[CCDSolver], 
        grid_config: GridConfig, 
        solver_kwargs: Optional[Dict[str, Any]] = None
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
        self.solver = solver_class(grid_config, **solver_kwargs)
        self.left_builder = LeftHandBlockBuilder()
        
        # 行列を計算
        self.L = self.left_builder.build_block(self.grid_config)
        
        # 右辺ベクトルを含む行列を生成せず、必要に応じてソルバーから直接情報を取得
    
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
        print(f"特異値の比率(最大/最小): {s[0]/s[-1]:.2e}")
        
        # 対称性の確認
        sym_diff = self.L - self.L.T
        max_asym = jnp.max(jnp.abs(sym_diff))
        print(f"対称性の最大誤差: {max_asym:.2e}")
        
        # 行と列のノルム
        row_norms = jnp.sqrt(jnp.sum(self.L * self.L, axis=1))
        col_norms = jnp.sqrt(jnp.sum(self.L * self.L, axis=0))
        
        print(f"行ノルムの最大値: {jnp.max(row_norms):.2e}")
        print(f"行ノルムの最小値: {jnp.min(row_norms):.2e}")
        print(f"行ノルムの最大/最小比: {jnp.max(row_norms)/jnp.min(row_norms):.2e}")
        
        print(f"列ノルムの最大値: {jnp.max(col_norms):.2e}")
        print(f"列ノルムの最小値: {jnp.min(col_norms):.2e}")
        print(f"列ノルムの最大/最小比: {jnp.max(col_norms)/jnp.min(col_norms):.2e}")
        
        # ソルバーに固有の行列がある場合は表示
        if hasattr(self.solver, 'L_scaled'):
            print("\n=== スケーリング後の行列特性 ===")
            L_scaled = self.solver.L_scaled
            cond_L_scaled = jnp.linalg.cond(L_scaled)
            print(f"スケーリング後の条件数: {cond_L_scaled:.2e}")
            print(f"条件数の改善率: {cond_L/cond_L_scaled:.2f}倍")
            
            # スケーリング後の行と列のノルム
            row_norms_scaled = jnp.sqrt(jnp.sum(L_scaled * L_scaled, axis=1))
            col_norms_scaled = jnp.sqrt(jnp.sum(L_scaled * L_scaled, axis=0))
            
            print(f"スケーリング後の行ノルムの最大/最小比: {jnp.max(row_norms_scaled)/jnp.min(row_norms_scaled):.2e}")
            print(f"スケーリング後の列ノルムの最大/最小比: {jnp.max(col_norms_scaled)/jnp.min(col_norms_scaled):.2e}")
            
            if visualize:
                # resultsディレクトリを作成
                os.makedirs("results", exist_ok=True)
                visualize_matrix_properties(L_scaled, f"Scaled Matrix Properties ({self.solver_name})", "results/scaled_matrix.png")
            
        if visualize:
            # resultsディレクトリを作成
            os.makedirs("results", exist_ok=True)
            visualize_matrix_properties(self.L, "Original Matrix Properties", "results/original_matrix.png")
        
        # 結果を辞書に格納して返す
        results = {
            "condition_number": float(cond_L),
            "max_singular": float(s[0]),
            "min_singular": float(s[-1]),
            "singular_ratio": float(s[0]/s[-1]),
            "max_asymmetry": float(max_asym),
            "row_norm_ratio": float(jnp.max(row_norms)/jnp.min(row_norms)),
            "col_norm_ratio": float(jnp.max(col_norms)/jnp.min(col_norms))
        }
        
        if hasattr(self.solver, 'L_scaled'):
            results.update({
                "scaled_condition_number": float(cond_L_scaled),
                "condition_improvement": float(cond_L/cond_L_scaled),
                "scaled_row_norm_ratio": float(jnp.max(row_norms_scaled)/jnp.min(row_norms_scaled)),
                "scaled_col_norm_ratio": float(jnp.max(col_norms_scaled)/jnp.min(col_norms_scaled))
            })
        
        return results

    def analyze_boundary_blocks(self):
        """境界ブロック行列の確認"""
        B0, C0, D0, ZR, AR, BR = self.left_builder._build_boundary_blocks()
        
        print("=== 境界ブロック行列の確認 ===")
        print("\n左境界:")
        print("B0 (左端):\n", B0)
        print("\nC0 (左端の次):\n", C0)
        print("\nD0 (左端の次の次):\n", D0)
        
        print("\n右境界:")
        print("BR (右端):\n", BR)
        print("\nAR (右端の手前):\n", AR)
        print("\nZR (右端の手前の手前):\n", ZR)
        
        # 境界条件の階数チェック
        left_block = jnp.concatenate([B0, C0, D0], axis=1)
        right_block = jnp.concatenate([ZR, AR, BR], axis=1)
        left_rank = jnp.linalg.matrix_rank(left_block)
        right_rank = jnp.linalg.matrix_rank(right_block)
        
        print("左境界ブロックの階数:", left_rank)
        print("右境界ブロックの階数:", right_rank)
        
        return {
            "left_boundary_rank": int(left_rank),
            "right_boundary_rank": int(right_rank)
        }

    def analyze_matrix_structure(self):
        """行列の構造特性を分析"""
        print("=== 行列の構造分析 ===")
        
        # ブロック構造の確認
        n = self.grid_config.n_points
        block_size = 4  # 新しいブロックサイズは4
        
        # 左端と右端のブロックを表示
        print("\n左端の4×12ブロック:")
        print(self.L[:block_size, :3*block_size])
        
        print("\n右端の4×12ブロック:")
        print(self.L[-block_size:, -3*block_size:])
        
        # スパース性の確認
        zeros = jnp.sum(jnp.abs(self.L) < 1e-10)
        sparsity = zeros / self.L.size
        print(f"\n疎行列度: {sparsity:.2%}")
        
        # 対角優位性の確認
        diag = jnp.abs(jnp.diag(self.L))
        off_diag_max = jnp.zeros_like(diag)
        diag_dominance_ratio = jnp.zeros_like(diag)
        
        for i in range(self.L.shape[0]):
            row = jnp.abs(self.L[i])
            row = row.at[i].set(0)  # 対角要素を除外
            off_diag_max = off_diag_max.at[i].set(jnp.max(row))
            if off_diag_max[i] > 0:
                diag_dominance_ratio = diag_dominance_ratio.at[i].set(diag[i] / off_diag_max[i])
        
        diag_dominant = jnp.all(diag > off_diag_max)
        avg_diag_dominance = jnp.mean(diag_dominance_ratio)
        
        print(f"対角優位性: {'あり' if diag_dominant else 'なし'}")
        print(f"平均対角優位比率: {avg_diag_dominance:.2f}")
        print(f"対角要素の最小値: {jnp.min(diag):.2e}")
        print(f"非対角要素の最大値: {jnp.max(off_diag_max):.2e}")
        
        # バンド幅の計算
        bandwidth = 0
        for i in range(self.L.shape[0]):
            row = jnp.abs(self.L[i])
            nonzero = jnp.where(row > 1e-10)[0]
            if len(nonzero) > 0:
                bandwidth = max(bandwidth, jnp.max(jnp.abs(nonzero - i)))
        
        print(f"行列のバンド幅: {bandwidth}")
        
        return {
            "sparsity": float(sparsity),
            "diag_dominant": bool(diag_dominant),
            "avg_diag_dominance": float(avg_diag_dominance),
            "min_diag_value": float(jnp.min(diag)),
            "max_offdiag_value": float(jnp.max(off_diag_max)),
            "bandwidth": int(bandwidth)
        }

    def perform_full_diagnosis(self, visualize: bool = False):
        """総合的な診断を実行
        
        Args:
            visualize: 可視化を行うかどうか
            
        Returns:
            診断結果のディクショナリ
        """
        solver_name = self.solver_class.__name__
        print(f"\n========== {solver_name}の診断 ==========")
        
        print("\n1. 行列の基本特性")
        print("-" * 40)
        matrix_props = self.analyze_matrix_properties(visualize)
        
        print("\n2. 境界条件の確認")
        print("-" * 40)
        boundary_props = self.analyze_boundary_blocks()
        
        print("\n3. 行列構造の確認")
        print("-" * 40)
        structure_props = self.analyze_matrix_structure()
        
        print("\n========== 診断完了 ==========")
        
        # 結果を統合
        results = {
            "solver_name": solver_name,
            "grid_points": self.grid_config.n_points,
            "grid_spacing": self.grid_config.h,
            "matrix_properties": matrix_props,
            "boundary_properties": boundary_props,
            "structure_properties": structure_props
        }
        
        return results