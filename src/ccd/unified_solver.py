"""
統合されたCCDソルバーモジュール

プラグイン形式のスケーリングと正則化を組み合わせた単一の合成ソルバー実装を提供します。
"""

from jax import jit
import jax.numpy as jnp
from functools import partial
from typing import Dict, Any, Optional, List, Tuple, Type

from ccd_core import GridConfig, LeftHandBlockBuilder
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
        
        # デフォルトのソルバー機能を設定
        self.inverse_scaling = lambda x: x
        self.solver_func = None  # 親クラス初期化後に設定
        
        # 親クラスの初期化（行列構築など）
        super().__init__(grid_config)
        
        # スケーリングと正則化の初期化（左辺行列Lが構築された後で実行）
        self._initialize_scaling_and_regularization()
    
    def _initialize_scaling_and_regularization(self):
        """スケーリングと正則化を初期化"""
        L = self.L  # 親クラスで構築済みの左辺行列
        
        # スケーリングの適用
        try:
            scaling_class = scaling_registry.get(self.scaling)
            scaling_strategy = scaling_class(L, **self.scaling_params)
            L_scaled, self.inverse_scaling = scaling_strategy.apply_scaling()
        except KeyError:
            print(f"警告: '{self.scaling}' スケーリング戦略が見つかりません。スケーリングなしで続行します。")
            L_scaled = L
            self.inverse_scaling = lambda x: x
        
        # 正則化の適用
        try:
            regularization_class = regularization_registry.get(self.regularization)
            # 不要な引数を削除
            regularization_strategy = regularization_class(L_scaled, **self.regularization_params)
            self.L_reg, self.inverse_regularization = regularization_strategy.apply_regularization()
        except KeyError:
            print(f"警告: '{self.regularization}' 正則化戦略が見つかりません。正則化なしで続行します。")
            self.L_reg = L_scaled
            self.inverse_regularization = lambda x: x
        
        # 処理後の行列を保存（診断用）
        self.L_scaled = L_scaled
        
        # スケーリングと正則化の戦略名を保存
        self.scaling_name = self.scaling
        self.regularization_name = self.regularization

    @partial(jit, static_argnums=(0,))
    def solve(self, f: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        導関数を計算
        
        Args:
            f: 関数値ベクトル (n,)
            
        Returns:
            関数値と導関数のタプル (f, f', f'', f''')
        """
        # 右辺ベクトルを計算（親クラスのメソッドを使用）
        rhs = self._build_right_hand_vector(f)
        
        # 正則化されたソルバー関数を使用して解を計算
        solution_scaled = jnp.linalg.solve(self.L_reg, rhs)
        
        # スケーリングの逆変換を適用
        solution = self.inverse_scaling(solution_scaled)
        
        # 解ベクトルから各成分を抽出
        return self._extract_derivatives(solution)
    
    @staticmethod
    def load_plugins(silent: bool = False):
        """
        スケーリングと正則化のプラグインを読み込む
        
        Args:
            silent: Trueの場合、出力を抑制
            
        Returns:
            (利用可能なスケーリング戦略のリスト, 利用可能な正則化戦略のリスト)
        """
        # 静かモードを設定
        if silent:
            scaling_registry.enable_silent_mode()
            regularization_registry.enable_silent_mode()
        else:
            scaling_registry.disable_silent_mode()
            regularization_registry.disable_silent_mode()
        
        # プラグインが既にロードされているかどうかを確認
        if not hasattr(CCDCompositeSolver, '_plugins_loaded'):
            # プロジェクトのルートディレクトリを検出
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # スケーリング戦略のディレクトリをスキャン
            scaling_dir = os.path.join(current_dir, 'scaling')
            if os.path.exists(scaling_dir) and os.path.isdir(scaling_dir):
                scaling_registry.scan_directory(scaling_dir)
            
            # 正則化戦略のディレクトリをスキャン
            regularization_dir = os.path.join(current_dir, 'regularization')
            if os.path.exists(regularization_dir) and os.path.isdir(regularization_dir):
                # デバッグ出力
                if not silent:
                    print(f"正則化ディレクトリのスキャン: {regularization_dir}")
                    print(f"ディレクトリ内のファイル: {os.listdir(regularization_dir)}")
                
                regularization_registry.scan_directory(regularization_dir)
            
            # フラグを設定してプラグインが読み込まれたことを記録
            CCDCompositeSolver._plugins_loaded = True
        
        # 通常モードに戻す
        if silent:
            scaling_registry.disable_silent_mode()
            regularization_registry.disable_silent_mode()
        
        return scaling_registry.get_names(), regularization_registry.get_names()

    @classmethod
    def available_scaling_methods(cls) -> List[str]:
        """利用可能なスケーリング手法のリストを返す"""
        # プラグインをロード（静かモード）
        cls.load_plugins(silent=True)
        return scaling_registry.get_names()
    
    @classmethod
    def available_regularization_methods(cls) -> List[str]:
        """利用可能な正則化手法のリストを返す"""
        # プラグインをロード（静かモード）
        cls.load_plugins(silent=True)
        return regularization_registry.get_names()
    
    @classmethod
    def display_available_methods(cls):
        """利用可能な手法を表示"""
        # プラグインをロード
        cls.load_plugins()
        
        # 利用可能な手法を表示
        print("=== 使用可能なスケーリング手法 ===")
        for method in sorted(cls.available_scaling_methods()):
            param_info = cls.get_scaling_param_info(method)
            if param_info:
                params = ", ".join([f"{k} ({v['help']}, デフォルト: {v['default']})" for k, v in param_info.items()])
                print(f"- {method} - パラメータ: {params}")
            else:
                print(f"- {method}")
        
        print("\n=== 使用可能な正則化手法 ===")
        for method in sorted(cls.available_regularization_methods()):
            # 特定の冗長な名前はスキップ
            if method in ['s_v_d', 't_s_v_d', 'l_s_q_r']:
                continue
                
            param_info = cls.get_regularization_param_info(method)
            if param_info:
                params = ", ".join([f"{k} ({v['help']}, デフォルト: {v['default']})" for k, v in param_info.items()])
                print(f"- {method} - パラメータ: {params}")
            else:
                print(f"- {method}")
    
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
        # プラグインをロード（静かモード）
        cls.load_plugins(silent=True)
        
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


# 起動時にプラグインを自動的にロード（サイレントモード）
CCDCompositeSolver.load_plugins(silent=True)