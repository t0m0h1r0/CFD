"""
2次元CCD法ソルバーモジュール

2次元結合コンパクト差分法による微分計算クラスを提供します。
"""

import cupy as cp
import cupyx.scipy.sparse as cpx_sparse
import cupyx.scipy.sparse.linalg as cpx_spla
from typing import List, Optional, Dict, Any, Tuple, Union

from grid_config_2d import GridConfig2D
from matrix_builder_2d import CCDLeftHandBuilder2D
from vector_builder_2d import CCDRightHandBuilder2D
from result_extractor_2d import CCDResultExtractor2D
from system_builder_2d import CCDSystemBuilder2D

# 1次元のDirectSolverとIterativeSolverを再利用
from ccd_solver import DirectSolver, IterativeSolver


class CCDSolver2D:
    """2次元CCD法による偏微分計算クラス"""

    def __init__(
        self,
        grid_config: GridConfig2D,
        use_iterative: bool = False,
        enable_boundary_correction: bool = None,
        solver_kwargs: Optional[Dict[str, Any]] = None,
        system_builder: Optional[CCDSystemBuilder2D] = None,
        coeffs: Optional[List[float]] = None,
    ):
        """
        2次元CCDソルバーの初期化

        Args:
            grid_config: 2次元グリッド設定
            use_iterative: 反復法を使用するかどうか
            enable_boundary_correction: 境界補正を有効にするかどうか
            solver_kwargs: 線形ソルバーのパラメータ
            system_builder: システムビルダーのインスタンス
            coeffs: 係数 [a, b, c, d]
        """
        # CuPyデバイスの明示的選択
        cp.cuda.Device(0).use()

        self.grid_config = grid_config

        # 係数とフラグの設定
        if coeffs is not None:
            self.grid_config.coeffs = coeffs

        if enable_boundary_correction is not None:
            self.grid_config.enable_boundary_correction = enable_boundary_correction

        # 係数への参照
        self.coeffs = self.grid_config.coeffs

        # システムビルダーの初期化（なければ新規作成）
        if system_builder is None:
            matrix_builder = CCDLeftHandBuilder2D()
            vector_builder = CCDRightHandBuilder2D()
            result_extractor = CCDResultExtractor2D()
            system_builder = CCDSystemBuilder2D(
                matrix_builder, vector_builder, result_extractor
            )
        self.system_builder = system_builder

        # デフォルトの零ベクトルをCuPyに変換
        zero_values = cp.zeros((grid_config.nx_points, grid_config.ny_points), dtype=cp.float64)

        # システムビルダーを使用して行列を作成
        try:
            self.L, _ = self.system_builder.build_system(grid_config, zero_values)
        except Exception as e:
            print(f"Error building system: {e}")
            raise

        # 行列の疎性を調査して出力
        nnz = self.L.nnz
        size = self.L.shape[0] * self.L.shape[1]
        density = nnz / size

        # ソルバーの選択（CuPy対応）
        solver_kwargs = solver_kwargs or {}
        self.use_iterative = use_iterative
        self.solver = (
            IterativeSolver(**solver_kwargs) if use_iterative else DirectSolver()
        )

    def solve(
        self, f: Union[cp.ndarray, List[List[float]]]
    ) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
        """
        関数値fに対する2次元CCD方程式系を解き、関数とその各偏導関数を返す

        Args:
            f: グリッド点での関数値の2次元配列

        Returns:
            (f, f_x, f_y, f_xx, f_xy, f_yy)のタプル
        """
        # 入力をCuPy配列に変換
        f_cupy = cp.asarray(f)

        # 右辺ベクトルを構築
        _, b = self.system_builder.build_system(self.grid_config, f_cupy)

        # 線形方程式を解く
        solution = self.solver.solve(self.L, b)

        # 解から関数値と各階微分を抽出
        result = self.system_builder.extract_results(self.grid_config, solution)

        # メモリ管理: 不要になった大きな配列を明示的に解放
        del f_cupy, b, solution
        cp.get_default_memory_pool().free_all_blocks()

        return result