"""
統合CCDソルバー

スケーリングと正則化を組み合わせたCCDソルバーを提供します。
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Dict, Any, Optional, List, Tuple

from ccd_core import (
    GridConfig, 
    CCDLeftHandBuilder, 
    CCDRightHandBuilder, 
    CCDResultExtractor, 
    CCDSystemBuilder
)
from ccd_solver import CCDSolver, DirectSolver
from plugin_loader import PluginLoader
from transformation_pipeline import TransformerFactory


class CCDCompositeSolver(CCDSolver):
    """
    スケーリングと正則化を組み合わせた統合ソルバー
    """
    
    def __init__(
        self, 
        grid_config: GridConfig,
        scaling: str = "none",
        regularization: str = "none",
        scaling_params: Optional[Dict[str, Any]] = None,
        regularization_params: Optional[Dict[str, Any]] = None,
        coeffs: Optional[List[float]] = None,
        use_direct_solver: bool = True
    ):
        """
        初期化
        
        Args:
            grid_config: グリッド設定
            scaling: スケーリング戦略名
            regularization: 正則化戦略名
            scaling_params: スケーリングパラメータ
            regularization_params: 正則化パラメータ
            coeffs: 微分係数 [a, b, c, d]
            use_direct_solver: 直接法を使用するかどうか
        """
        # プラグインを読み込み
        PluginLoader.load_plugins(verbose=False)
        
        # システムビルダーの初期化
        self.system_builder = CCDSystemBuilder(
            CCDLeftHandBuilder(),
            CCDRightHandBuilder(),
            CCDResultExtractor()
        )
        
        # 係数を設定
        self.coeffs = coeffs if coeffs is not None else [1.0, 0.0, 0.0, 0.0]
        
        # インスタンス変数を設定
        self.scaling = scaling.lower()
        self.regularization = regularization.lower()
        self.scaling_params = scaling_params or {}
        self.regularization_params = regularization_params or {}
        
        # グリッド設定を保存
        self.grid_config = grid_config
        
        # 左辺行列の構築
        self.L, _ = self.system_builder.build_system(grid_config, jnp.zeros(grid_config.n_points), self.coeffs)
        
        # 変換パイプラインを初期化
        self.transformer = TransformerFactory.create_transformation_pipeline(
            self.L,
            scaling=self.scaling,
            regularization=self.regularization,
            scaling_params=self.scaling_params,
            regularization_params=self.regularization_params
        )
        
        # 行列を変換
        self.L_transformed, self.inverse_transform = self.transformer.transform_matrix(self.L)
        
        # ソルバーを初期化
        self.solver = DirectSolver() if use_direct_solver else None
    
    @partial(jax.jit, static_argnums=(0,))
    def solve(self, f: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        導関数を計算
        
        Args:
            f: グリッド点での関数値
            
        Returns:
            (ψ, ψ', ψ'', ψ''')のタプル
        """
        # 右辺ベクトルを計算
        _, rhs = self.system_builder.build_system(self.grid_config, f, self.coeffs)
        
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
        coeffs: Optional[List[float]] = None
    ) -> 'CCDCompositeSolver':
        """
        パラメータを指定してソルバーを作成するファクトリーメソッド
        
        Args:
            grid_config: グリッド設定
            scaling: スケーリング戦略名
            regularization: 正則化戦略名
            params: パラメータの辞書
            coeffs: 微分係数 [a, b, c, d]
            
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
        
        # 各種パラメータ情報を取得
        scaling_info = {}
        regularization_info = {}
        
        try:
            scaling_info = PluginLoader.get_param_info(scaling)
        except KeyError:
            pass
        
        try:
            regularization_info = PluginLoader.get_param_info(regularization)
        except KeyError:
            pass
        
        # パラメータを適切に振り分け
        for param_name, param_value in params.items():
            if param_name in scaling_info:
                scaling_params[param_name] = param_value
            elif param_name in regularization_info:
                regularization_params[param_name] = param_value
        
        return cls(
            grid_config=grid_config,
            scaling=scaling,
            regularization=regularization,
            scaling_params=scaling_params,
            regularization_params=regularization_params,
            coeffs=coeffs
        )
