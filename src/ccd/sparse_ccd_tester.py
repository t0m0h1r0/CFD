"""
疎行列対応CCDメソッドテスターモジュール

疎行列を使用したCCDソルバー実装のテスト機能を提供します。
"""

import jax.numpy as jnp
import time
import os
from typing import Tuple, List, Type, Dict, Any, Optional

from grid_config import GridConfig
from sparse_ccd_solver import SparseCCDSolver
from test_functions import TestFunction, TestFunctionFactory
from visualization import visualize_derivative_results
from ccd_tester import CCDMethodTester


class SparseCCDMethodTester(CCDMethodTester):
    """
    疎行列を使用したCCD法のテストを実行するクラス
    
    CCDMethodTesterを継承し、疎行列ソルバーに対応させます。
    基本的な機能はCCDMethodTesterと同じですが、
    疎行列ソルバーの特性に最適化されています。
    """

    def __init__(
        self,
        solver_class: Type[SparseCCDSolver],
        grid_config: GridConfig,
        x_range: Tuple[float, float],
        solver_kwargs: Optional[Dict[str, Any]] = None,
        test_functions: Optional[List[TestFunction]] = None,
        coeffs: Optional[List[float]] = None,
    ):
        """
        Args:
            solver_class: テスト対象の疎行列CCDソルバークラス
            grid_config: グリッド設定
            x_range: x軸の範囲 (開始位置, 終了位置)
            solver_kwargs: ソルバーの初期化パラメータ
            test_functions: テスト関数のリスト (Noneの場合は標準関数セットを使用)
            coeffs: [a, b, c, d] 係数リスト。Noneの場合は[1, 0, 0, 0]を使用 (f = psi)
        """
        # CCDMethodTesterと同様の初期化
        super().__init__(
            solver_class, 
            grid_config, 
            x_range, 
            solver_kwargs, 
            test_functions, 
            coeffs
        )
        
        # メモリ使用量のレポートを追加
        self.report_memory_usage = True
    
    def compute_errors(
        self, test_func: TestFunction
    ) -> Tuple[float, float, float, float]:
        """
        各導関数の誤差を計算 - メモリ使用量計測を追加
        
        Args:
            test_func: テスト関数

        Returns:
            (psi'の誤差, psi''の誤差, psi'''の誤差, 計算時間)
        """
        # 親クラスのcompute_errorsメソッドを呼び出し
        result = super().compute_errors(test_func)
        return result
    
    def run_tests(
        self, prefix: str = "", visualize: bool = True
    ) -> Dict[str, Tuple[List[float], float]]:
        """
        すべてのテスト関数に対してテストを実行

        Args:
            prefix: 出力ファイルの接頭辞
            visualize: 可視化を行うかどうか

        Returns:
            テスト結果の辞書 {関数名: ([1階誤差, 2階誤差, 3階誤差], 計算時間)}
        """
        # 出力ディレクトリの作成
        os.makedirs("results", exist_ok=True)
        
        # 疎行列ソルバーであることを表示
        solver_name = self.solver_class.__name__
        print(f"疎行列ソルバー {solver_name} を使用したテストを実行します")
        
        # 親クラスのrun_testsメソッドを呼び出し
        results = super().run_tests(prefix, visualize)
        
        # 疎行列特有の情報を表示
        if hasattr(self.solver, 'L_sparse'):
            matrix_size = self.grid_config.n_points * 4
            # 理論的な疎行列の非ゼロ要素数の推定
            estimated_nnz = (3 * 4 * 4) * 2 + (3 * 4 * 4) * (self.grid_config.n_points - 2)
            density = estimated_nnz / (matrix_size * matrix_size)
            
            print("\n===== 疎行列情報 =====")
            print(f"行列サイズ: {matrix_size}x{matrix_size}")
            print(f"推定非ゼロ要素数: {estimated_nnz}")
            print(f"推定密度: {density:.2e}")
            print(f"メモリ削減率（推定）: {1.0 - density:.2%}")
        
        return results


# テスト用コード
if __name__ == "__main__":
    from sparse_ccd_solver import SparseCompositeSolver
    
    # グリッド設定
    grid_config = GridConfig(
        n_points=64,
        h=1.0/63,
        coeffs=[1.0, 0.0, 0.0, 0.0]
    )
    
    # スパーステスターの実行
    tester = SparseCCDMethodTester(
        SparseCompositeSolver,
        grid_config,
        (-1.0, 1.0),
        solver_kwargs={"scaling": "none", "regularization": "none"},
    )
    
    tester.run_tests(prefix="sparse_test_", visualize=True)
