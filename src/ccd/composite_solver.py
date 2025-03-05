"""
統合CCDソルバー

スケーリングと正則化を組み合わせたCCDソルバーを提供します。
"""

import cupy as cp
from typing import Dict, Any, Optional, List, Tuple

from grid_config import GridConfig
from ccd_solver import CCDSolver
from plugin_loader import PluginLoader
from transformation_pipeline import TransformerFactory


class CCDCompositeSolver(CCDSolver):
    """
    スケーリングと正則化を組み合わせた統合ソルバー（CuPy対応）
    """

    def __init__(
        self,
        grid_config: GridConfig,
        scaling: str = "none",
        regularization: str = "none",
        scaling_params: Optional[Dict[str, Any]] = None,
        regularization_params: Optional[Dict[str, Any]] = None,
        use_direct_solver: bool = True,
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
            use_direct_solver: 直接法を使用するかどうか
            enable_boundary_correction: 境界補正を有効にするかどうか
            coeffs: 微分係数 [a, b, c, d]
            **kwargs: 追加のパラメータ
        """
        # プラグインを読み込み
        PluginLoader.load_plugins(verbose=False)

        # インスタンス変数を設定
        self.scaling = scaling.lower()
        self.regularization = regularization.lower()
        self.scaling_params = scaling_params or {}
        self.regularization_params = regularization_params or {}

        # 係数と境界補正の設定
        if coeffs is not None:
            grid_config.coeffs = coeffs

        if enable_boundary_correction is not None:
            grid_config.enable_boundary_correction = enable_boundary_correction

        # 親クラスのコンストラクタを呼び出し
        solver_kwargs = kwargs.copy()
        solver_kwargs["use_iterative"] = not use_direct_solver

        super().__init__(grid_config, **solver_kwargs)

        # 変換パイプラインを初期化
        self.transformer = TransformerFactory.create_transformation_pipeline(
            self.L,
            scaling=self.scaling,
            regularization=self.regularization,
            scaling_params=self.scaling_params,
            regularization_params=self.regularization_params,
        )

        # 行列を変換
        self.L_transformed, self.inverse_transform = self.transformer.transform_matrix(
            self.L
        )

    def solve(
        self, f: cp.ndarray
    ) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
        """
        導関数を計算

        Args:
            f: グリッド点での関数値

        Returns:
            (ψ, ψ', ψ'', ψ''')のタプル
        """
        # 右辺ベクトルを計算
        _, rhs = self.system_builder.build_system(self.grid_config, f)

        # 右辺ベクトルに変換を適用
        rhs_transformed = self.transformer.transform_rhs(rhs)

        # 線形方程式を解く
        solution_transformed = self.solver.solve(self.L_transformed, rhs_transformed)

        # 逆変換を適用
        solution = self.inverse_transform(solution_transformed)

        # 解ベクトルから各成分を抽出
        return self.system_builder.extract_results(self.grid_config, solution)

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

    @classmethod
    def create_solver(
        cls,
        grid_config: GridConfig,
        scaling: str = "none",
        regularization: str = "none",
        params: Optional[Dict[str, Any]] = None,
        coeffs: Optional[List[float]] = None,
        enable_boundary_correction: bool = None,
        **kwargs,
    ) -> "CCDCompositeSolver":
        """
        パラメータを指定してソルバーを作成するファクトリーメソッド

        Args:
            grid_config: グリッド設定
            scaling: スケーリング戦略名
            regularization: 正則化戦略名
            params: パラメータの辞書
            coeffs: 微分係数 [a, b, c, d]
            enable_boundary_correction: 境界補正を有効にするかどうか
            **kwargs: 追加のパラメータ

        Returns:
            CCDCompositeSolver インスタンス
        """
        params = params or {}

        # 正規化処理
        scaling = scaling.lower()
        regularization = regularization.lower()

        # パラメータを分類
        scaling_params = {}
        regularization_params = {}

        # パラメータ情報を取得して振り分け
        try:
            scaling_info = PluginLoader.get_param_info(scaling)
            for param_name, param_value in params.items():
                if param_name in scaling_info:
                    scaling_params[param_name] = param_value
        except KeyError:
            pass

        try:
            regularization_info = PluginLoader.get_param_info(regularization)
            for param_name, param_value in params.items():
                if param_name in regularization_info:
                    regularization_params[param_name] = param_value
        except KeyError:
            pass

        return cls(
            grid_config=grid_config,
            scaling=scaling,
            regularization=regularization,
            scaling_params=scaling_params,
            regularization_params=regularization_params,
            coeffs=coeffs,
            enable_boundary_correction=enable_boundary_correction,
            **kwargs,
        )
