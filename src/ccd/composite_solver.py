"""
改良版統合CCDソルバーモジュール

プラグイン形式のスケーリングと正則化を組み合わせた合成ソルバー実装を提供します。
リファクタリングにより、SOLID原則に準拠した責任の分離を実現しています。
"""

from jax import jit
import jax.numpy as jnp
from functools import partial
from typing import Dict, Any, Optional, List, Tuple, Protocol
import os

from ccd_core import (
    GridConfig, 
    CCDLeftHandBuilder, 
    CCDRightHandBuilder, 
    CCDResultExtractor, 
    CCDSystemBuilder
)
from ccd_solver import CCDSolver, DirectSolver, IterativeSolver
from scaling_strategies_base import scaling_registry
from regularization_strategies_base import regularization_registry
from strategy_adapters import (
    ScalingStrategyFactory,
    RegularizationStrategyFactory
)

class MatrixTransformer(Protocol):
    """行列変換のプロトコル定義"""
    
    def transform_matrix(self, matrix: jnp.ndarray) -> Tuple[jnp.ndarray, callable]:
        """
        行列を変換し、逆変換関数を返す
        
        Args:
            matrix: 変換する行列
            
        Returns:
            変換後の行列、逆変換関数
        """
        ...
    
    def transform_rhs(self, rhs: jnp.ndarray) -> jnp.ndarray:
        """
        右辺ベクトルを変換する
        
        Args:
            rhs: 変換する右辺ベクトル
            
        Returns:
            変換後の右辺ベクトル
        """
        ...


class CompositeMatrixTransformer:
    """スケーリングと正則化を順に適用する複合変換器"""
    
    def __init__(
        self, 
        scaling_strategy: Optional[MatrixTransformer] = None, 
        regularization_strategy: Optional[MatrixTransformer] = None
    ):
        """
        Args:
            scaling_strategy: スケーリング戦略
            regularization_strategy: 正則化戦略
        """
        self.scaling_strategy = scaling_strategy
        self.regularization_strategy = regularization_strategy
        self.L_orig = None
        self.L_scaled = None
        self.L_reg = None
    
    def transform_matrix(self, matrix: jnp.ndarray) -> Tuple[jnp.ndarray, callable]:
        """
        スケーリングと正則化を順に適用し、逆変換関数を返す
        
        Args:
            matrix: 元の行列
            
        Returns:
            変換後の行列、逆変換関数
        """
        self.L_orig = matrix
        transformed_matrix = matrix
        inverse_funcs = []
        
        # スケーリングの適用
        if self.scaling_strategy:
            transformed_matrix, inverse_scaling = self.scaling_strategy.transform_matrix(transformed_matrix)
            inverse_funcs.append(inverse_scaling)
            self.L_scaled = transformed_matrix
        else:
            self.L_scaled = transformed_matrix
            inverse_funcs.append(lambda x: x)  # 恒等関数
        
        # 正則化の適用
        if self.regularization_strategy:
            transformed_matrix, inverse_regularization = self.regularization_strategy.transform_matrix(transformed_matrix)
            inverse_funcs.append(inverse_regularization)
        else:
            inverse_funcs.append(lambda x: x)  # 恒等関数
        
        self.L_reg = transformed_matrix
        
        # 逆変換関数の合成（正則化の逆変換を先に適用し、次にスケーリングの逆変換）
        def composite_inverse(x):
            for inverse_func in reversed(inverse_funcs):
                x = inverse_func(x)
            return x
        
        return transformed_matrix, composite_inverse
    
    def transform_rhs(self, rhs: jnp.ndarray) -> jnp.ndarray:
        """
        右辺ベクトルに変換を適用
        
        Args:
            rhs: 右辺ベクトル
            
        Returns:
            変換後の右辺ベクトル
        """
        transformed_rhs = rhs
        
        # スケーリングの適用
        if self.scaling_strategy:
            transformed_rhs = self.scaling_strategy.transform_rhs(transformed_rhs)
        
        # 正則化の適用
        if self.regularization_strategy:
            transformed_rhs = self.regularization_strategy.transform_rhs(transformed_rhs)
        
        return transformed_rhs


class CCDCompositeSolver(CCDSolver):
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
        coeffs: Optional[List[float]] = None,
        use_iterative: bool = False,
        solver_kwargs: Optional[dict] = None
    ):
        """
        Args:
            grid_config: グリッド設定
            scaling: スケーリング戦略名
            regularization: 正則化戦略名
            scaling_params: スケーリングパラメータ
            regularization_params: 正則化パラメータ
            coeffs: [a, b, c, d] 係数リスト
            use_iterative: 反復法を使用するかどうか
            solver_kwargs: 線形ソルバーのパラメータ
        """
        # プラグインを一度だけロード
        if not CCDCompositeSolver._plugins_loaded:
            self._load_plugins()
        
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
        
        # 左辺行列の構築
        self.L, _ = self.system_builder.build_system(grid_config, jnp.zeros(grid_config.n_points), self.coeffs)
        
        # 変換器を初期化
        self.transformer = self._initialize_transformer()
        
        # 行列を変換
        self.L_transformed, self.inverse_transform = self.transformer.transform_matrix(self.L)
        
        # ソルバーの選択
        if use_iterative:
            self.solver = IterativeSolver(**(solver_kwargs or {}))
        else:
            self.solver = DirectSolver()
        
        # グリッド設定を保存
        self.grid_config = grid_config
    
    def _initialize_transformer(self) -> CompositeMatrixTransformer:
        """スケーリングと正則化の変換器を初期化"""
        scaling_strategy = None
        regularization_strategy = None
        
        # スケーリングの初期化
        try:
            if self.scaling != "none":
                # アダプターファクトリーを使用して適切なアダプターを生成
                scaling_strategy = ScalingStrategyFactory.create_adapter(
                    self.scaling, self.L, **self.scaling_params
                )
        except KeyError:
            print(f"警告: '{self.scaling}' スケーリング戦略が見つかりません。スケーリングなしで続行します。")
        
        # 正則化の初期化
        try:
            if self.regularization != "none":
                # アダプターファクトリーを使用して適切なアダプターを生成
                regularization_strategy = RegularizationStrategyFactory.create_adapter(
                    self.regularization, self.L, **self.regularization_params
                )
        except KeyError:
            print(f"警告: '{self.regularization}' 正則化戦略が見つかりません。正則化なしで続行します。")
        
        return CompositeMatrixTransformer(scaling_strategy, regularization_strategy)
    
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
        _, rhs = self.system_builder.build_system(self.grid_config, f, self.coeffs)
        
        # 右辺ベクトルに変換を適用
        rhs_transformed = self.transformer.transform_rhs(rhs)
        
        # 線形方程式を解く
        solution_transformed = self.solver.solve(self.L_transformed, rhs_transformed)
        
        # 逆変換を適用
        solution = self.inverse_transform(solution_transformed)
        
        # 解ベクトルから各成分を抽出
        return self.system_builder.extract_results(self.grid_config, solution)
    
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
        スケーリングと正則化のプラグインを読み込む
        
        Args:
            silent: 出力抑制フラグ
        
        Returns:
            (利用可能なスケーリング戦略のリスト, 利用可能な正則化戦略のリスト)
        """
        if silent:
            scaling_registry.enable_silent_mode()
            regularization_registry.enable_silent_mode()
        
        if not cls._plugins_loaded:
            cls._load_plugins()
        
        if silent:
            scaling_registry.disable_silent_mode()
            regularization_registry.disable_silent_mode()
        
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
        coeffs: Optional[List[float]] = None,
        use_iterative: bool = False,
        solver_kwargs: Optional[dict] = None
    ) -> 'CCDCompositeSolver':
        """
        パラメータを指定してソルバーを作成するファクトリーメソッド
        
        Args:
            grid_config: グリッド設定
            scaling: スケーリング戦略名
            regularization: 正則化戦略名
            params: パラメータ辞書
            coeffs: [a, b, c, d] 係数リスト
            use_iterative: 反復法を使用するかどうか
            solver_kwargs: 線形ソルバーのパラメータ
            
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
            coeffs=coeffs,
            use_iterative=use_iterative,
            solver_kwargs=solver_kwargs
        )