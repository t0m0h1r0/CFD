"""
CCD法ソルバーモジュール

結合コンパクト差分法による微分計算クラスを提供します。
GridConfigの拡張境界条件管理を利用します。
"""

import jax.numpy as jnp
import jax
from functools import partial
from typing import Tuple, List, Optional, Protocol
import jax.scipy.sparse.linalg as jspl

from grid_config import GridConfig
from system_builder import CCDSystemBuilder
from ccd_core import create_system_builder


class LinearSolver(Protocol):
    """線形方程式系を解くためのプロトコル"""

    def solve(self, L: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """
        線形方程式 L*x = b を解く

        Args:
            L: 左辺行列
            b: 右辺ベクトル

        Returns:
            解ベクトル x
        """
        ...


class DirectSolver:
    """直接法による線形ソルバー"""

    def solve(self, L: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """直接法で線形方程式を解く"""
        return jnp.linalg.solve(L, b)


class IterativeSolver:
    """反復法による線形ソルバー"""

    def __init__(
        self,
        tol: float = 1e-10,
        atol: float = 1e-10,
        restart: int = 50,
        maxiter: int = 1000,
    ):
        """
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

    def solve(self, L: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """GMRES法で線形方程式を解く"""
        solution, info = jspl.gmres(
            L,
            b,
            tol=self.tol,
            atol=self.atol,
            restart=self.restart,
            maxiter=self.maxiter,
        )
        return solution


class CCDSolver:
    """結合コンパクト差分法による微分計算クラス"""

    def __init__(
        self,
        grid_config: GridConfig,
        coeffs: Optional[List[float]] = None,
        use_iterative: bool = False,
        solver_kwargs: Optional[dict] = None,
        system_builder: Optional[CCDSystemBuilder] = None,
        enable_boundary_correction: bool = None,
    ):
        """
        CCDソルバーの初期化

        Args:
            grid_config: グリッド設定
            coeffs: [a, b, c, d] 係数リスト。Noneの場合は[1, 0, 0, 0]を使用 (f = psi)
            use_iterative: 反復法を使用するかどうか
            solver_kwargs: 線形ソルバーのパラメータ
            system_builder: CCDSystemBuilderのインスタンス。Noneの場合は新規作成
            enable_boundary_correction: 境界補正を有効にするかどうか
        """
        self.grid_config = grid_config

        # 係数を設定 - grid_configに保存
        if coeffs is not None:
            self.grid_config.coeffs = coeffs
        
        # 境界補正の設定
        if enable_boundary_correction is not None:
            self.grid_config.enable_boundary_correction = enable_boundary_correction
        
        # 係数への参照を保持（後方互換性のため）
        self.coeffs = self.grid_config.coeffs

        # システムビルダーの初期化または使用
        self.system_builder = system_builder if system_builder else create_system_builder()

        # 左辺行列の構築
        self.L, _ = self.system_builder.build_system(
            grid_config,
            jnp.zeros(grid_config.n_points),
            self.coeffs,
        )

        # 行列の特性を分析して最適なソルバーを選択
        self._select_solver(use_iterative, solver_kwargs or {})

    def _select_solver(self, use_iterative: bool, solver_kwargs: dict):
        """ソルバーの選択

        Args:
            use_iterative: 反復法を使用するかどうか
            solver_kwargs: ソルバーのパラメータ
        """
        self.use_iterative = use_iterative

        if use_iterative:
            self.solver = IterativeSolver(**solver_kwargs)
        else:
            self.solver = DirectSolver()

    @partial(jax.jit, static_argnums=(0,))
    def solve(
        self, f: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        関数値fに対するCCD方程式系を解き、ψとその各階微分を返す

        Args:
            f: グリッド点での関数値

        Returns:
            (ψ, ψ', ψ'', ψ''')のタプル
        """
        # 右辺ベクトルを構築
        _, b = self.system_builder.build_system(
            self.grid_config,
            f,
            self.coeffs,
        )

        # 線形方程式を解く
        solution = self.solver.solve(self.L, b)

        # 解から関数値と各階微分を抽出して返す
        return self.system_builder.extract_results(self.grid_config, solution)


# ベクトル化対応版のCCDSolverクラス
class VectorizedCCDSolver(CCDSolver):
    """複数の関数に同時に適用可能なCCDソルバー"""

    @partial(jax.jit, static_argnums=(0,))
    def solve_batch(
        self, fs: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        複数の関数値セットに対してCCD方程式系を一度に解く

        Args:
            fs: 関数値の配列 (batch_size, n_points)

        Returns:
            (ψs, ψ's, ψ''s, ψ'''s) 各要素は (batch_size, n_points) の形状
        """
        # バッチサイズとグリッド点数を取得
        batch_size, n_points = fs.shape

        # 各バッチについて個別に計算する関数を定義
        def solve_one(f):
            return self.solve(f)

        # バッチ全体に対して適用 (vmap)
        return jax.vmap(solve_one)(fs)