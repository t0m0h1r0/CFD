"""
統合されたCCDソルバーモジュール

プラグイン形式のスケーリングと正則化を組み合わせた単一の合成ソルバー実装を提供します。
"""

from jax import jit
import jax.numpy as jnp
from functools import partial
from typing import Dict, Any, Optional, List, Tuple, Type

from ccd_core import GridConfig
from ccd_solver import CCDSolver as BaseCCDSolver

# スケーリングと正則化の戦略とレジストリをインポート
from scaling_strategies_base import ScalingStrategy, scaling_registry
from regularization_strategies_base import RegularizationStrategy, regularization_registry

# プラグイン管理モジュールをインポート
import os
import importlib.util


class CCDCompositeSolver(BaseCCDSolver):
    """スケーリングと正則化を組み合わせた統合ソルバー"""
    
    def __init__(
        self, 
        grid_config: GridConfig,
        scaling: str = "none",
        regularization: str = "none",
        scaling_params: Optional[Dict[str, Any]] = None,
        regularization_params: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            grid_config: グリッド設定
            scaling: スケーリング戦略名
            regularization: 正則化戦略名
            scaling_params: スケーリングパラメータ
            regularization_params: 正則化パラメータ
        """
        # インスタンス変数を先に保存
        self.scaling = scaling.lower()
        self.regularization = regularization.lower()
        self.scaling_params = scaling_params or {}
        self.regularization_params = regularization_params or {}
        
        # 親クラスの初期化（行列構築など）
        super().__init__(grid_config)
    
    def _initialize_solver(self):
        """ソルバーの初期化 - スケーリングと正則化を適用"""
        # 行列の構築
        L = self.left_builder.build_block(self.grid_config)
        K = self.right_builder.build_block(self.grid_config)
        
        # スケーリングの適用
        try:
            scaling_class = scaling_registry.get(self.scaling)
            scaling_strategy = scaling_class(L, K, **self.scaling_params)
            L_scaled, K_scaled, self.inverse_scaling = scaling_strategy.apply_scaling()
        except KeyError:
            print(f"警告: '{self.scaling}' スケーリング戦略が見つかりません。スケーリングなしで続行します。")
            L_scaled, K_scaled = L, K
            self.inverse_scaling = lambda x: x
        
        # 正則化の適用
        try:
            regularization_class = regularization_registry.get(self.regularization)
            regularization_strategy = regularization_class(L_scaled, K_scaled, **self.regularization_params)
            self.L_reg, self.K_reg, self.solver_func = regularization_strategy.apply_regularization()
        except KeyError:
            print(f"警告: '{self.regularization}' 正則化戦略が見つかりません。正則化なしで続行します。")
            self.L_reg, self.K_reg = L_scaled, K_scaled
            self.solver_func = lambda rhs: jnp.linalg.solve(self.L_reg, rhs)
        
        # 処理後の行列を保存（診断用）
        self.L_scaled = L_scaled
        self.K_scaled = K_scaled
        
        # スケーリングと正則化の戦略名を保存
        self.scaling_name = self.scaling
        self.regularization_name = self.regularization

    @partial(jit, static_argnums=(0,))
    def solve(self, f: jnp.ndarray) -> jnp.ndarray:
        """
        導関数を計算
        
        Args:
            f: 関数値ベクトル (n,)
            
        Returns:
            X: 導関数ベクトル (3n,) - [f'_0, f''_0, f'''_0, f'_1, f''_1, f'''_1, ...]
        """
        # 右辺ベクトルを計算
        rhs = self.K_reg @ f
        
        # 正則化されたソルバー関数を使用して解を計算
        X_scaled = self.solver_func(rhs)
        
        # スケーリングの逆変換を適用
        X = self.inverse_scaling(X_scaled)
        
        return X
    
    @staticmethod
    def load_plugins():
        """
        スケーリングと正則化のプラグインを読み込む
        
        Returns:
            (利用可能なスケーリング戦略のリスト, 利用可能な正則化戦略のリスト)
        """
        # プロジェクトのルートディレクトリを検出
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # スケーリング戦略のディレクトリをスキャン
        scaling_dir = os.path.join(current_dir, 'scaling')
        if os.path.exists(scaling_dir) and os.path.isdir(scaling_dir):
            scaling_registry.scan_directory(scaling_dir)
        
        # 正則化戦略のディレクトリをスキャン
        regularization_dir = os.path.join(current_dir, 'regularization')
        if os.path.exists(regularization_dir) and os.path.isdir(regularization_dir):
            regularization_registry.scan_directory(regularization_dir)
        
        return scaling_registry.get_names(), regularization_registry.get_names()
    
    @classmethod
    def available_scaling_methods(cls) -> List[str]:
        """利用可能なスケーリング手法のリストを返す"""
        # プラグインをロード
        cls.load_plugins()
        return scaling_registry.get_names()
    
    @classmethod
    def available_regularization_methods(cls) -> List[str]:
        """利用可能な正則化手法のリストを返す"""
        # プラグインをロード
        cls.load_plugins()
        return regularization_registry.get_names()
    
    @classmethod
    def get_scaling_param_info(cls, scaling_name: str) -> Dict[str, Dict[str, Any]]:
        """
        スケーリング戦略のパラメータ情報を取得
        
        Args:
            scaling_name: スケーリング戦略名
            
        Returns:
            パラメータ情報の辞書
        """
        try:
            scaling_class = scaling_registry.get(scaling_name)
            return scaling_class.get_param_info()
        except KeyError:
            return {}
    
    @classmethod
    def get_regularization_param_info(cls, regularization_name: str) -> Dict[str, Dict[str, Any]]:
        """
        正則化戦略のパラメータ情報を取得
        
        Args:
            regularization_name: 正則化戦略名
            
        Returns:
            パラメータ情報の辞書
        """
        try:
            regularization_class = regularization_registry.get(regularization_name)
            return regularization_class.get_param_info()
        except KeyError:
            return {}
    
    @classmethod
    def create_solver(
        cls, 
        grid_config: GridConfig, 
        scaling: str = "none", 
        regularization: str = "none", 
        params: Optional[Dict[str, Any]] = None
    ) -> 'CCDCompositeSolver':
        """
        パラメータを指定してソルバーを作成するファクトリーメソッド
        
        Args:
            grid_config: グリッド設定
            scaling: スケーリング戦略名
            regularization: 正則化戦略名
            params: パラメータ辞書 {パラメータ名: 値, ...}
            
        Returns:
            設定されたCCDCompositeSolverインスタンス
        """
        # プラグインをロード
        cls.load_plugins()
        
        params = params or {}
        
        # スケーリングと正則化のパラメータを分離
        scaling_param_info = cls.get_scaling_param_info(scaling)
        regularization_param_info = cls.get_regularization_param_info(regularization)
        
        scaling_params = {}
        regularization_params = {}
        
        # パラメータを適切な辞書に振り分け
        for param_name, param_value in params.items():
            if param_name in scaling_param_info:
                scaling_params[param_name] = param_value
            elif param_name in regularization_param_info:
                regularization_params[param_name] = param_value
        
        return cls(
            grid_config=grid_config,
            scaling=scaling,
            regularization=regularization,
            scaling_params=scaling_params,
            regularization_params=regularization_params
        )


# 起動時にプラグインを自動的にロード
CCDCompositeSolver.load_plugins()
