"""
統合されたCCDソルバーモジュール

プラグイン形式のスケーリングと正則化を組み合わせた合成ソルバー実装を提供します。
スケーリングと正則化の適用順序と変換処理を改善しました。
"""

from jax import jit
import jax.numpy as jnp
from functools import partial
from typing import Dict, Any, Optional, List, Tuple
import os

from ccd_core import GridConfig
from ccd_solver import CCDSolver as BaseCCDSolver
from scaling_strategies_base import scaling_registry
from regularization_strategies_base import regularization_registry


class CCDCompositeSolver(BaseCCDSolver):
    """スケーリングと正則化を組み合わせた統合ソルバー"""
    
    # 静的クラス変数
    _plugins_loaded = False
    
    def __init__(
        self, 
        grid_config: GridConfig,
        scaling: str = "none",
        regularization: str = "none",
        scaling_params: Optional[Dict[str, Any]] = None,
        regularization_params: Optional[Dict[str, Any]] = None,
        coeffs: Optional[List[float]] = None
    ):
        """
        Args:
            grid_config: グリッド設定
            scaling: スケーリング戦略名
            regularization: 正則化戦略名
            scaling_params: スケーリングパラメータ
            regularization_params: 正則化パラメータ
            coeffs: [a, b, c, d] 係数リスト
        """
        # インスタンス変数を設定
        self.scaling = scaling.lower()
        self.regularization = regularization.lower()
        self.scaling_params = scaling_params or {}
        self.regularization_params = regularization_params or {}
        
        # デフォルトの変換関数を設定
        self.inverse_scaling = lambda x: x
        self.scale_rhs = lambda x: x
        self.transform_rhs = lambda x: x
        self.inverse_regularization = lambda x: x
        
        # プラグインを一度だけロード
        if not CCDCompositeSolver._plugins_loaded:
            self._load_plugins()
        
        # 親クラスの初期化
        super().__init__(grid_config, coeffs)
        
        # スケーリングと正則化の初期化
        self._initialize_scaling_and_regularization()
    
    def _initialize_scaling_and_regularization(self):
        """スケーリングと正則化を初期化"""
        L = self.L  # 親クラスで構築済みの左辺行列
        
        # スケーリングの適用
        try:
            scaling_class = scaling_registry.get(self.scaling)
            scaling_strategy = scaling_class(L, **self.scaling_params)
            L_scaled, self.inverse_scaling = scaling_strategy.apply_scaling()
            
            # 右辺ベクトルのスケーリング関数があれば取得
            if hasattr(scaling_strategy, 'scale_rhs'):
                self.scale_rhs = scaling_strategy.scale_rhs
        except KeyError:
            print(f"警告: '{self.scaling}' スケーリング戦略が見つかりません。スケーリングなしで続行します。")
            L_scaled = L
        
        # 正則化の適用
        try:
            regularization_class = regularization_registry.get(self.regularization)
            regularization_strategy = regularization_class(L_scaled, **self.regularization_params)
            self.L_reg, self.inverse_regularization = regularization_strategy.apply_regularization()
            
            # 右辺ベクトルの変換関数があれば取得
            if hasattr(regularization_strategy, 'transform_rhs'):
                self.transform_rhs = regularization_strategy.transform_rhs
        except KeyError:
            print(f"警告: '{self.regularization}' 正則化戦略が見つかりません。正則化なしで続行します。")
            self.L_reg = L_scaled
        
        # 診断用に保存
        self.L_scaled = L_scaled

    @partial(jit, static_argnums=(0,))
    def solve(self, f: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        導関数を計算
        
        Args:
            f: 関数値ベクトル (n,)
            
        Returns:
            関数値と導関数のタプル (psi, psi', psi'', psi''')
        """
        # 右辺ベクトルを計算
        rhs = self._build_right_hand_vector(f)
        
        # 順番に変換を適用:
        # 1. スケーリング
        rhs_scaled = self.scale_rhs(rhs)
        
        # 2. 正則化
        rhs_reg = self.transform_rhs(rhs_scaled)
        
        # 線形方程式を解く
        solution_reg = jnp.linalg.solve(self.L_reg, rhs_reg)
        
        # 逆の順で逆変換を適用:
        # 1. 正則化の逆変換
        solution_scaled = self.inverse_regularization(solution_reg)
        
        # 2. スケーリングの逆変換
        solution = self.inverse_scaling(solution_scaled)
        
        # 解ベクトルから各成分を抽出
        return self._extract_derivatives(solution)
    
    @staticmethod
    def _load_plugins():
        """スケーリングと正則化のプラグインを読み込む（内部メソッド）"""
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
        
        # プラグイン読み込み完了フラグを設定
        CCDCompositeSolver._plugins_loaded = True
    
    @classmethod
    def load_plugins(cls, silent: bool = False):
        """
        スケーリングと正則化のプラグインを読み込む（互換性のために維持）
        
        Args:
            silent: 出力抑制フラグ（簡素化版では無視）
        
        Returns:
            (利用可能なスケーリング戦略のリスト, 利用可能な正則化戦略のリスト)
        """
        if not cls._plugins_loaded:
            cls._load_plugins()
        
        return cls.available_scaling_methods(), cls.available_regularization_methods()

    @classmethod
    def available_scaling_methods(cls) -> List[str]:
        """利用可能なスケーリング手法のリストを返す"""
        if not cls._plugins_loaded:
            cls._load_plugins()
        return scaling_registry.get_names()
    
    @classmethod
    def available_regularization_methods(cls) -> List[str]:
        """利用可能な正則化手法のリストを返す"""
        if not cls._plugins_loaded:
            cls._load_plugins()
        return regularization_registry.get_names()
    
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
            params: パラメータ辞書
            coeffs: [a, b, c, d] 係数リスト
            
        Returns:
            設定されたCCDCompositeSolverインスタンス
        """
        if not cls._plugins_loaded:
            cls._load_plugins()
        
        params = params or {}
        
        # パラメータを分類
        scaling_params = {}
        regularization_params = {}
        
        # 簡易パラメータ振り分け
        for param_name, param_value in params.items():
            if param_name in ['alpha', 'iterations', 'threshold', 'threshold_ratio', 
                             'rank', 'damp', 'relaxation', 'l1_ratio', 'tol']:
                regularization_params[param_name] = param_value
            elif param_name in ['max_iter', 'tol']:
                scaling_params[param_name] = param_value
        
        return cls(
            grid_config=grid_config,
            scaling=scaling,
            regularization=regularization,
            scaling_params=scaling_params,
            regularization_params=regularization_params,
            coeffs=coeffs
        )