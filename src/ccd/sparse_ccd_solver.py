"""
CuPy対応疎行列CCDソルバーモジュール

メモリ効率の良い疎行列ソルバーをCuPyで実装
"""

import cupy as cp
import cupyx.scipy.sparse as cpx_sparse
import cupyx.scipy.sparse.linalg as cpx_spla
from typing import List, Optional, Dict, Any

from grid_config import GridConfig
from sparse_matrix_builder import SparseCCDLeftHandBuilder
from vector_builder import CCDRightHandBuilder
from result_extractor import CCDResultExtractor
from plugin_loader import PluginLoader
from transformation_pipeline import TransformationPipeline, TransformerFactory


class SparseCCDSolver:
    """疎行列を使用したCCD法ソルバー（CuPy対応）"""

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
            use_iterative: 反復法を使用するかどうか
            enable_boundary_correction: 境界補正を有効にするかどうか
            solver_kwargs: 線形ソルバーのパラメータ
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
        
        # ビルダーの初期化
        self.matrix_builder = SparseCCDLeftHandBuilder()
        self.vector_builder = CCDRightHandBuilder()
        self.result_extractor = CCDResultExtractor()
        
        # 左辺行列の構築
        row_indices, col_indices, values, matrix_shape = \
            self.matrix_builder.build_matrix_coo_data(
                grid_config,
                dirichlet_enabled=grid_config.is_dirichlet,
                neumann_enabled=grid_config.is_neumann
            )
        
        # CuPyのスパース行列に変換
        indices = cp.column_stack([row_indices, col_indices])
        self.L_sparse = cpx_sparse.bcoo_matrix((cp.asarray(values), indices), 
                                                shape=matrix_shape)
        
        # ソルバーパラメータ
        solver_kwargs = solver_kwargs or {}
        self.use_iterative = use_iterative
        
        # ソルバーパラメータ
        self.tol = solver_kwargs.get('tol', 1e-10)
        self.atol = solver_kwargs.get('atol', 1e-10)
        self.maxiter = solver_kwargs.get('maxiter', 1000)
        self.restart = solver_kwargs.get('restart', 50)

    def solve(self, f):
        """
        関数値fに対するCCD方程式系を解き、ψとその各階微分を返す

        Args:
            f: グリッド点での関数値

        Returns:
            (ψ, ψ', ψ'', ψ''')のタプル
        """
        # CuPy配列に変換
        f_cupy = cp.asarray(f)

        # 右辺ベクトルを構築
        b = self.vector_builder.build_vector(
            self.grid_config,
            f_cupy,
            dirichlet_enabled=self.grid_config.is_dirichlet,
            neumann_enabled=self.grid_config.is_neumann
        )
        
        # 反復法で解く
        solution, info = cpx_spla.gmres(
            self.L_sparse, 
            b, 
            tol=self.tol, 
            atol=self.atol, 
            restart=self.restart, 
            maxiter=self.maxiter
        )
        
        # 結果を抽出
        result = self.result_extractor.extract_components(self.grid_config, solution)

        # メモリ管理
        del f_cupy, b, solution
        cp.get_default_memory_pool().free_all_blocks()

        return result


class SparseCompositeSolver(SparseCCDSolver):
    """
    スケーリングと正則化を組み合わせた疎行列ソルバー
    """

    def __init__(
        self,
        grid_config: GridConfig,
        scaling: str = "none",
        regularization: str = "none",
        scaling_params: Optional[Dict[str, Any]] = None,
        regularization_params: Optional[Dict[str, Any]] = None,
        use_direct_solver: bool = False,  # 疎行列ソルバーではデフォルトで反復法を使用
        enable_boundary_correction: bool = None,
        coeffs: Optional[List[float]] = None,
        **kwargs,
    ):
        # プラグインの読み込み
        PluginLoader.load_plugins(verbose=False)

        # 親クラスの初期化
        super().__init__(
            grid_config,
            use_iterative=not use_direct_solver,
            enable_boundary_correction=enable_boundary_correction,
            solver_kwargs=kwargs,
            coeffs=coeffs,
        )

        # 変換パイプラインの初期化
        self.transformer = TransformerFactory.create_transformation_pipeline(
            self.L_sparse,
            scaling=scaling,
            regularization=regularization,
            scaling_params=scaling_params,
            regularization_params=regularization_params,
        )

        # 行列の変換
        self.L_transformed, self.inverse_transform = self.transformer.transform_matrix(self.L_sparse)

    def solve(self, f):
        # 親クラスの solve メソッドを呼び出し
        return super().solve(f)

    @classmethod
    def load_plugins(cls, silent: bool = True) -> None:
        """
        プラグインを読み込む

        Args:
            silent: ログを抑制するかどうか
        """
        PluginLoader.load_plugins(verbose=not silent)

    @classmethod
    def available_scaling_methods(cls) -> List[str]:
        """
        利用可能なスケーリング手法のリストを返す

        Returns:
            スケーリング手法名のリスト
        """
        return PluginLoader.available_scaling_methods()

    @classmethod
    def available_regularization_methods(cls) -> List[str]:
        """
        利用可能な正則化手法のリストを返す

        Returns:
            正則化手法名のリスト
        """
        return PluginLoader.available_regularization_methods()