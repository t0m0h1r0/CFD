"""
スパースCCDソルバーモジュール

疎行列を使用したメモリ効率の良いCCDソルバーを提供します。
"""

import jax
import jax.numpy as jnp
import jax.scipy.sparse as jsp
import jax.scipy.sparse.linalg as jspl
from functools import partial
from typing import Tuple, List, Optional, Dict, Any

from grid_config import GridConfig
from sparse_matrix_builder import SparseCCDLeftHandBuilder
from vector_builder import CCDRightHandBuilder
from result_extractor import CCDResultExtractor


class SparseCCDSolver:
    """疎行列を使用したCCDソルバー"""

    def __init__(
        self,
        grid_config: GridConfig,
        use_iterative: bool = True,
        enable_boundary_correction: bool = None,
        solver_kwargs: Optional[Dict[str, Any]] = None,
        coeffs: Optional[List[float]] = None,
    ):
        """
        スパースCCDソルバーの初期化
        
        Args:
            grid_config: グリッド設定
            use_iterative: 反復法を使用するかどうか (疎行列では推奨)
            enable_boundary_correction: 境界補正を有効にするかどうか
            solver_kwargs: 線形ソルバーのパラメータ
            coeffs: 係数 [a, b, c, d]
        """
        self.grid_config = grid_config

        # 係数とフラグの設定
        if coeffs is not None:
            self.grid_config.coeffs = coeffs
        
        if enable_boundary_correction is not None:
            self.grid_config.enable_boundary_correction = enable_boundary_correction
        
        # 係数への参照（アクセスしやすさのため）
        self.coeffs = self.grid_config.coeffs
        
        # ビルダーの初期化
        self.matrix_builder = SparseCCDLeftHandBuilder()
        self.vector_builder = CCDRightHandBuilder()
        self.result_extractor = CCDResultExtractor()
        
        # 左辺行列の構築（COOデータのみ保持）
        self.row_indices, self.col_indices, self.values, self.matrix_shape = \
            self.matrix_builder.build_matrix_coo_data(
                grid_config,
                dirichlet_enabled=grid_config.is_dirichlet,
                neumann_enabled=grid_config.is_neumann
            )
        
        # ソルバーパラメータ
        self.use_iterative = use_iterative
        solver_kwargs = solver_kwargs or {}
        
        # 反復ソルバーの設定
        self.tol = solver_kwargs.get('tol', 1e-10)
        self.atol = solver_kwargs.get('atol', 1e-10)
        self.maxiter = solver_kwargs.get('maxiter', 1000)
        self.restart = solver_kwargs.get('restart', 50)
        
        # スパース行列の構築
        self.build_matrix()

    def build_matrix(self):
        """
        スパース行列を構築
        """
        # BCOOスパース行列を構築
        indices = jnp.column_stack([self.row_indices, self.col_indices])
        self.L_sparse = jsp.bcoo_matrix((self.values, indices), shape=self.matrix_shape)
        
        # CRS形式に変換（行アクセスが速い）- 実際の計算で使用
        # 注意: JSP.BCSRMatrixは現状ではまだ機能が制限されているため、
        # BCOO形式を保持し、必要に応じて変換します。

    def solve_iterative(self, b: jnp.ndarray) -> jnp.ndarray:
        """
        反復法により線形方程式を解く
        
        Args:
            b: 右辺ベクトル
            
        Returns:
            解ベクトル
        """
        # スパース行列を使用して反復的に解く
        solution, info = jspl.gmres(
            self.L_sparse,
            b,
            tol=self.tol,
            atol=self.atol,
            restart=self.restart,
            maxiter=self.maxiter
        )
        
        # 注意: infoの値をチェックするべきだが、現在のJSPのGMRESはinfoを返さない
        return solution
    
    def solve_direct(self, b: jnp.ndarray) -> jnp.ndarray:
        """
        直接法により線形方程式を解く
        
        注意: スパース行列に対する直接法はJAXで十分にサポートされていないため、
        この関数は密行列に変換して解くことになり、大規模問題ではメモリ効率が悪化します。
        
        Args:
            b: 右辺ベクトル
            
        Returns:
            解ベクトル
        """
        # BCOOからdense行列に変換
        L_dense = self.L_sparse.todense()
        
        # 直接法で解く
        return jnp.linalg.solve(L_dense, b)

    @partial(jax.jit, static_argnums=(0,))
    def solve(
        self, f: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        スパース形式で関数値fに対するCCD方程式系を解き、ψとその各階微分を返す

        Args:
            f: グリッド点での関数値

        Returns:
            (ψ, ψ', ψ'', ψ''')のタプル
        """
        # 右辺ベクトルを構築
        b = self.vector_builder.build_vector(
            self.grid_config,
            f,
            dirichlet_enabled=self.grid_config.is_dirichlet,
            neumann_enabled=self.grid_config.is_neumann
        )
        
        # 適切なソルバーで線形方程式を解く
        if self.use_iterative:
            solution = self.solve_iterative(b)
        else:
            solution = self.solve_direct(b)
        
        # 解から関数値と各階微分を抽出して返す
        return self.result_extractor.extract_components(self.grid_config, solution)


class SparseCompositeSolver(SparseCCDSolver):
    """
    スケーリングと正則化を組み合わせたスパース統合ソルバー
    
    このクラスはcomposite_solver.pyのCCDCompositeSolverと同様の機能を
    スパース行列に拡張したものです。スケーリングと正則化の戦略はdense行列と
    共通のものが使用できますが、パフォーマンスのためにスパース行列向けに
    最適化することが理想的です。
    """
    
    def __init__(
        self,
        grid_config: GridConfig,
        scaling: str = "none",
        regularization: str = "none",
        scaling_params: Optional[Dict[str, Any]] = None,
        regularization_params: Optional[Dict[str, Any]] = None,
        use_direct_solver: bool = False,
        enable_boundary_correction: bool = None,
        coeffs: Optional[List[float]] = None,
        **kwargs,
    ):
        """
        初期化
        
        Args:
            grid_config: グリッド設定
            scaling: スケーリング戦略名
            regularization: 正則化戦略名
            scaling_params: スケーリングパラメータ
            regularization_params: 正則化パラメータ
            use_direct_solver: 直接法を使用するかどうか（疎行列では通常False推奨）
            enable_boundary_correction: 境界補正を有効にするかどうか
            coeffs: 微分係数 [a, b, c, d]
            **kwargs: 追加のパラメータ
        """
        from plugin_loader import PluginLoader
        from transformation_pipeline import TransformerFactory
        
        # プラグインの読み込み
        PluginLoader.load_plugins(verbose=False)
        
        # インスタンス変数の設定
        self.scaling = scaling.lower()
        self.regularization = regularization.lower()
        self.scaling_params = scaling_params or {}
        self.regularization_params = regularization_params or {}
        
        # 親クラスの初期化
        super().__init__(
            grid_config=grid_config,
            use_iterative=not use_direct_solver,
            enable_boundary_correction=enable_boundary_correction,
            coeffs=coeffs,
            solver_kwargs=kwargs
        )
        
        # ここで最初にスパース行列をdenseに変換
        L_dense = self.L_sparse.todense()
        
        # 変換パイプラインの構築（既存のパイプラインを使用）
        self.transformer = TransformerFactory.create_transformation_pipeline(
            L_dense,
            scaling=self.scaling,
            regularization=self.regularization,
            scaling_params=self.scaling_params,
            regularization_params=self.regularization_params,
        )
        
        # 行列の変換
        self.L_transformed, self.inverse_transform = self.transformer.transform_matrix(L_dense)
        
        # 変換された行列をスパース形式に戻す（オプション）
        # 注意: 一部の変換は行列の疎性構造を変える可能性があるため、場合によっては
        # 密行列のままで計算するほうが良いケースもあります
        if self.use_iterative and self.scaling == "none" and self.regularization == "none":
            # 変換なしの場合は元のスパース構造を維持
            pass
        else:
            # 重要: L_transformedはdense行列なので、ソルバーでdense行列用の解法を使用
            self.use_sparse_solver = False
            
    @partial(jax.jit, static_argnums=(0,))
    def solve(
        self, f: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        導関数を計算
        
        Args:
            f: グリッド点での関数値
            
        Returns:
            (ψ, ψ', ψ'', ψ''')のタプル
        """
        # 右辺ベクトルを計算
        rhs = self.vector_builder.build_vector(
            self.grid_config, 
            f,
            dirichlet_enabled=self.grid_config.is_dirichlet,
            neumann_enabled=self.grid_config.is_neumann
        )
        
        # 右辺ベクトルに変換を適用
        rhs_transformed = self.transformer.transform_rhs(rhs)
        
        # 線形方程式を解く
        if hasattr(self, 'use_sparse_solver') and not self.use_sparse_solver:
            # 密行列用のソルバーを使用
            solution_transformed = jnp.linalg.solve(self.L_transformed, rhs_transformed)
        else:
            # スパース行列用の反復ソルバーを使用
            solution_transformed, _ = jspl.gmres(
                self.L_sparse, 
                rhs_transformed,
                tol=self.tol,
                atol=self.atol,
                restart=self.restart,
                maxiter=self.maxiter
            )
        
        # 逆変換を適用
        solution = self.inverse_transform(solution_transformed)
        
        # 解ベクトルから結果を抽出
        return self.result_extractor.extract_components(self.grid_config, solution)
    
    @classmethod
    def available_scaling_methods(cls) -> List[str]:
        """利用可能なスケーリング手法を返す"""
        from plugin_loader import PluginLoader
        return PluginLoader.available_scaling_methods()
    
    @classmethod
    def available_regularization_methods(cls) -> List[str]:
        """利用可能な正則化手法を返す"""
        from plugin_loader import PluginLoader
        return PluginLoader.available_regularization_methods()

    @classmethod
    def load_plugins(cls, silent: bool = True) -> None:
        """
        プラグインを読み込む

        Args:
            silent: ログを抑制するかどうか
        """
        from plugin_loader import PluginLoader
        PluginLoader.load_plugins(verbose=not silent)