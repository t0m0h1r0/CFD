"""
2次元統合CCDソルバー

スケーリングと正則化を組み合わせた2次元CCDソルバーを提供します。
"""

import cupy as cp
import cupyx.scipy.sparse as cpx_sparse
from typing import Dict, Any, Optional, List, Tuple, Union

from grid_config_2d import GridConfig2D
from ccd_solver_2d import CCDSolver2D
from plugin_loader import PluginLoader
from transformation_pipeline import TransformerFactory


class CCDCompositeSolver2D(CCDSolver2D):
    """
    スケーリングと正則化を組み合わせた2次元統合ソルバー
    """

    def __init__(
        self,
        grid_config: GridConfig2D,
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
            grid_config: 2次元グリッド設定
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
        try:
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
        except Exception as e:
            print(f"Warning: Error during transformation setup: {e}")
            print("Falling back to no transformation")
            self.L_transformed = self.L
            self.inverse_transform = lambda x: x

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
        try:
            f_cupy = cp.asarray(f, dtype=cp.float64)
        except Exception as e:
            print(f"Error converting input to CuPy array: {e}")
            # NumPy配列に変換してからCuPyに変換
            import numpy as np
            f_numpy = np.asarray(f, dtype=np.float64)
            f_cupy = cp.array(f_numpy)
        
        # 右辺ベクトルを計算
        try:
            _, rhs = self.system_builder.build_system(self.grid_config, f_cupy)

            # 右辺ベクトルに変換を適用
            rhs_transformed = self.transformer.transform_rhs(rhs)

            # 線形方程式を解く
            solution_transformed = self.solver.solve(self.L_transformed, rhs_transformed)

            # 逆変換を適用
            solution = self.inverse_transform(solution_transformed)
        except Exception as e:
            print(f"Error in 2D composite solver: {e}")
            # 変換なしで直接解く
            print("Attempting to solve without transformations")
            _, rhs = self.system_builder.build_system(self.grid_config, f_cupy)
            solution = self.solver.solve(self.L, rhs)

        # 解ベクトルから各成分を抽出
        return self.system_builder.extract_results(self.grid_config, solution)

    @classmethod
    def create_solver(
        cls,
        grid_config: GridConfig2D,
        scaling: str = "none",
        regularization: str = "none",
        params: Optional[Dict[str, Any]] = None,
        coeffs: Optional[List[float]] = None,
        enable_boundary_correction: bool = None,
        **kwargs,
    ) -> "CCDCompositeSolver2D":
        """
        パラメータを指定して2次元ソルバーを作成するファクトリーメソッド

        Args:
            grid_config: 2次元グリッド設定
            scaling: スケーリング戦略名
            regularization: 正則化戦略名
            params: パラメータの辞書
            coeffs: 微分係数 [a, b, c, d]
            enable_boundary_correction: 境界補正を有効にするかどうか
            **kwargs: 追加のパラメータ

        Returns:
            CCDCompositeSolver2D インスタンス
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