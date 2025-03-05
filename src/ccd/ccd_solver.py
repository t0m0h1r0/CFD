"""
CuPy対応結合コンパクト差分法ソルバーモジュール

結合コンパクト差分法による微分計算クラスを提供します。
"""

import cupy as cp
import cupyx.scipy.sparse.linalg as cpx_spla
import cupyx.scipy.sparse as cpx_sparse
from typing import List, Optional, Dict, Any

from grid_config import GridConfig
from system_builder import CCDSystemBuilder
from ccd_core import create_system_builder


class DirectSolver:
    """直接法による線形ソルバー（CuPy疎行列対応）"""

    def solve(self, L, b):
        """直接法で線形方程式を解く"""
        # 密行列を疎行列に変換
        if not isinstance(L, cpx_sparse.spmatrix):
            L = cpx_sparse.csr_matrix(L)

        return cpx_spla.spsolve(L, b)


class IterativeSolver:
    """反復法による線形ソルバー（CuPy対応）"""

    def __init__(
        self,
        tol: float = 1e-10,
        atol: float = 1e-10,
        restart: int = 50,
        maxiter: int = 1000,
    ):
        """
        初期化パラメータ

        Args:
            tol: 相対許容誤差
            atol: 絶対許容誤差
            restart: GMRESの再起動パラメータ
            maxiter: 最大反復回数
        """
        self.tol = tol
        self.atol = atol
        self.restart = restart
        self.maxiter = maxiter

    def solve(self, L, b):
        """GMRES法で線形方程式を解く"""
        # 密行列を疎行列に変換
        if not isinstance(L, cpx_sparse.spmatrix):
            L = cpx_sparse.csr_matrix(L)

        solution, _ = cpx_spla.gmres(
            L,
            b,
            tol=self.tol,
            atol=self.atol,
            restart=self.restart,
            maxiter=self.maxiter,
        )
        return solution


class CCDSolver:
    """結合コンパクト差分法による微分計算クラス（CuPy対応）"""

    def __init__(
        self,
        grid_config: GridConfig,
        use_iterative: bool = False,
        enable_boundary_correction: bool = None,
        solver_kwargs: Optional[Dict[str, Any]] = None,
        system_builder: Optional[CCDSystemBuilder] = None,
        coeffs: Optional[List[float]] = None,
    ):
        """
        CCDソルバーの初期化

        Args:
            grid_config: グリッド設定
            use_iterative: 反復法を使用するかどうか
            enable_boundary_correction: 境界補正を有効にするかどうか
            solver_kwargs: 線形ソルバーのパラメータ
            system_builder: CCDSystemBuilderのインスタンス
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

        # システムビルダーの初期化
        self.system_builder = (
            system_builder if system_builder else create_system_builder()
        )

        # 左辺行列の構築（CuPy配列に変換）
        # デフォルトの零ベクトルもCuPyに変換
        zero_vector = cp.zeros(grid_config.n_points)

        # システムビルダーを使用して行列を作成
        self.L, _ = self.system_builder.build_system(grid_config, zero_vector)

        # ソルバーの選択（CuPy対応）
        solver_kwargs = solver_kwargs or {}
        self.use_iterative = use_iterative
        self.solver = (
            IterativeSolver(**solver_kwargs) if use_iterative else DirectSolver()
        )

    def solve(self, f):
        """
        関数値fに対するCCD方程式系を解き、ψとその各階微分を返す

        Args:
            f: グリッド点での関数値

        Returns:
            (ψ, ψ', ψ'', ψ''')のタプル
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


class VectorizedCCDSolver(CCDSolver):
    """複数の関数に同時に適用可能なCCDソルバー（CuPy対応）"""

    def solve_batch(self, fs):
        """
        複数の関数値セットに対してCCD方程式系を一度に解く

        Args:
            fs: 関数値の配列 (batch_size, n_points)

        Returns:
            (ψs, ψ's, ψ''s, ψ'''s) 各要素は (batch_size, n_points) の形状
        """
        # NumPyまたはリストをCuPy配列に変換
        fs_cupy = cp.asarray(fs)

        # バッチ処理
        results = []
        for f in fs_cupy:
            results.append(self.solve(f))

        # メモリ管理
        del fs_cupy
        cp.get_default_memory_pool().free_all_blocks()

        # 結果を転置して元の形状に戻す
        return tuple(cp.stack(r) for r in zip(*results))
