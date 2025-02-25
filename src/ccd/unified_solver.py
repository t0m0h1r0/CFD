"""
統合されたCCDソルバーモジュール

スケーリングと正則化を組み合わせた単一の合成ソルバー実装を提供します。
"""

from jax import jit
import jax.numpy as jnp
from functools import partial
from typing import Type, Dict, Any, Optional, List, Callable, Tuple

from ccd_core import GridConfig
from ccd_solver import CCDSolver

# 基底クラスのインポート
from scaling_strategies_base import ScalingStrategy, NoneScaling
from regularization_strategies_base import RegularizationStrategy, NoneRegularization

# 基本的なスケーリング戦略のインポート
from scaling_strategies_basic import NormalizationScaling, RehuScaling

# 高度なスケーリング戦略のインポート
from scaling_strategies_advanced import (
    EqualizationScaling, IterativeScaling, VanDerSluisScaling,
    DiagonalDominanceScaling, SquareSumScaling, MaxElementScaling
)

# 基本的な正則化戦略のインポート
from regularization_strategies_basic import TikhonovRegularization

# SVDベースの正則化戦略のインポート
from regularization_strategies_svd import SVDRegularization, TSVDRegularization

# 反復的な正則化戦略のインポート
from regularization_strategies_iterative import (
    LandweberRegularization, PrecomputedLandweberRegularization, LSQRRegularization
)

# 高度な正則化戦略のインポート
from regularization_strategies_advanced import (
    TotalVariationRegularization, L1Regularization, ElasticNetRegularization
)


class CCDCompositeSolver(CCDSolver):
    """スケーリングと正則化を組み合わせた統合ソルバー"""
    
    # 利用可能なスケーリングと正則化の戦略のマッピング
    SCALING_STRATEGIES = {
        "none": NoneScaling,
        "normalization": NormalizationScaling,
        "rehu": RehuScaling,
        "equalization": EqualizationScaling,
        "iterative": IterativeScaling,
        "van_der_sluis": VanDerSluisScaling,
        "diagonal_dominance": DiagonalDominanceScaling,
        "square_sum": SquareSumScaling,
        "max_element": MaxElementScaling
    }
    
    REGULARIZATION_STRATEGIES = {
        "none": NoneRegularization,
        "tikhonov": TikhonovRegularization,
        "svd": SVDRegularization,
        "tsvd": TSVDRegularization,
        "landweber": LandweberRegularization,
        "precomputed_landweber": PrecomputedLandweberRegularization,
        "lsqr": LSQRRegularization,
        "total_variation": TotalVariationRegularization,
        "l1": L1Regularization,
        "elastic_net": ElasticNetRegularization
    }
    
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
            scaling: スケーリング戦略
            regularization: 正則化戦略
            scaling_params: スケーリングパラメータ
            regularization_params: 正則化パラメータ
        """
        # インスタンス変数を先に保存
        self.scaling = scaling
        self.regularization = regularization
        self.scaling_params = scaling_params or {}
        self.regularization_params = regularization_params or {}
        
        # 親クラスの初期化（行列構築など）
        super().__init__(grid_config)
    
    def _get_scaling_strategy(self, L: jnp.ndarray, K: jnp.ndarray) -> ScalingStrategy:
        """スケーリング戦略オブジェクトを取得"""
        strategy_class = self.SCALING_STRATEGIES.get(self.scaling, NoneScaling)
        
        # 特定のスケーリング戦略に対する特別な処理
        if strategy_class == IterativeScaling:
            max_iter = self.scaling_params.get("max_iter", 10)
            tol = self.scaling_params.get("tol", 1e-8)
            return strategy_class(L, K, max_iter, tol)
        
        return strategy_class(L, K)
    
    def _get_regularization_strategy(self, L: jnp.ndarray, K: jnp.ndarray) -> RegularizationStrategy:
        """正則化戦略オブジェクトを取得"""
        strategy_class = self.REGULARIZATION_STRATEGIES.get(self.regularization, NoneRegularization)
        
        # 正則化戦略のパラメータを取得
        if strategy_class == TikhonovRegularization:
            alpha = self.regularization_params.get("alpha", 1e-6)
            return strategy_class(L, K, alpha)
        elif strategy_class == SVDRegularization:
            threshold = self.regularization_params.get("threshold", 1e-10)
            return strategy_class(L, K, threshold)
        elif strategy_class == TSVDRegularization:
            rank = self.regularization_params.get("rank", None)
            threshold_ratio = self.regularization_params.get("threshold_ratio", 1e-5)
            return strategy_class(L, K, rank, threshold_ratio)
        elif strategy_class in (LandweberRegularization, PrecomputedLandweberRegularization):
            iterations = self.regularization_params.get("iterations", 20)
            relaxation = self.regularization_params.get("relaxation", 0.1)
            return strategy_class(L, K, iterations, relaxation)
        elif strategy_class == LSQRRegularization:
            iterations = self.regularization_params.get("iterations", 20)
            damp = self.regularization_params.get("damp", 0)
            return strategy_class(L, K, iterations, damp)
        elif strategy_class in (TotalVariationRegularization, L1Regularization):
            alpha = self.regularization_params.get("alpha", 1e-4)
            iterations = self.regularization_params.get("iterations", 50)
            tol = self.regularization_params.get("tol", 1e-6)
            return strategy_class(L, K, alpha, iterations, tol)
        elif strategy_class == ElasticNetRegularization:
            alpha = self.regularization_params.get("alpha", 1e-4)
            l1_ratio = self.regularization_params.get("l1_ratio", 0.5)
            iterations = self.regularization_params.get("iterations", 100)
            tol = self.regularization_params.get("tol", 1e-6)
            return strategy_class(L, K, alpha, l1_ratio, iterations, tol)
        else:
            return strategy_class(L, K)
    
    def _initialize_solver(self):
        """ソルバーの初期化 - スケーリングと正則化を適用"""
        # 行列の構築
        L = self.left_builder.build_block(self.grid_config)
        K = self.right_builder.build_block(self.grid_config)
        
        # スケーリングの適用
        scaling_strategy = self._get_scaling_strategy(L, K)
        L_scaled, K_scaled, self.inverse_scaling = scaling_strategy.apply_scaling()
        
        # 正則化の適用
        regularization_strategy = self._get_regularization_strategy(L_scaled, K_scaled)
        self.L_reg, self.K_reg, self.solver_func = regularization_strategy.apply_regularization()
        
        # 処理後の行列を保存（診断用）
        self.L_scaled = L_scaled
        self.K_scaled = K_scaled

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
    
    # 標準的な設定を持つソルバーを作成するファクトリーメソッド
    @classmethod
    def create_basic_solver(cls, grid_config: GridConfig) -> 'CCDCompositeSolver':
        """基本的なCCDソルバーを作成（スケーリングと正則化なし）"""
        return cls(grid_config, scaling="none", regularization="none")
    
    @classmethod
    def create_normalization_solver(cls, grid_config: GridConfig) -> 'CCDCompositeSolver':
        """ノルマライゼーションスケーリングを適用したソルバーを作成"""
        return cls(grid_config, scaling="normalization", regularization="none")
    
    @classmethod
    def create_rehu_solver(cls, grid_config: GridConfig) -> 'CCDCompositeSolver':
        """Rehuスケーリングを適用したソルバーを作成"""
        return cls(grid_config, scaling="rehu", regularization="none")
    
    @classmethod
    def create_equalization_solver(cls, grid_config: GridConfig) -> 'CCDCompositeSolver':
        """均等化スケーリングを適用したソルバーを作成"""
        return cls(grid_config, scaling="equalization", regularization="none")
    
    @classmethod
    def create_iterative_solver(cls, grid_config: GridConfig, max_iter: int = 10, tol: float = 1e-8) -> 'CCDCompositeSolver':
        """反復的スケーリング（Sinkhorn-Knopp法）を適用したソルバーを作成"""
        return cls(
            grid_config, 
            scaling="iterative", 
            regularization="none", 
            scaling_params={"max_iter": max_iter, "tol": tol}
        )
    
    @classmethod
    def create_van_der_sluis_solver(cls, grid_config: GridConfig) -> 'CCDCompositeSolver':
        """Van der Sluisスケーリングを適用したソルバーを作成"""
        return cls(grid_config, scaling="van_der_sluis", regularization="none")
    
    @classmethod
    def create_diagonal_dominance_solver(cls, grid_config: GridConfig) -> 'CCDCompositeSolver':
        """対角優位スケーリングを適用したソルバーを作成"""
        return cls(grid_config, scaling="diagonal_dominance", regularization="none")
    
    @classmethod
    def create_square_sum_solver(cls, grid_config: GridConfig) -> 'CCDCompositeSolver':
        """二乗和スケーリングを適用したソルバーを作成"""
        return cls(grid_config, scaling="square_sum", regularization="none")
    
    @classmethod
    def create_max_element_solver(cls, grid_config: GridConfig) -> 'CCDCompositeSolver':
        """最大成分スケーリングを適用したソルバーを作成"""
        return cls(grid_config, scaling="max_element", regularization="none")
    
    @classmethod
    def create_tikhonov_solver(cls, grid_config: GridConfig, alpha: float = 1e-6) -> 'CCDCompositeSolver':
        """Tikhonov正則化を適用したソルバーを作成"""
        return cls(
            grid_config, 
            scaling="none", 
            regularization="tikhonov", 
            regularization_params={"alpha": alpha}
        )
    
    @classmethod
    def create_svd_solver(cls, grid_config: GridConfig, threshold: float = 1e-10) -> 'CCDCompositeSolver':
        """SVD切断法を適用したソルバーを作成"""
        return cls(
            grid_config, 
            scaling="none", 
            regularization="svd", 
            regularization_params={"threshold": threshold}
        )
    
    @classmethod
    def create_tsvd_solver(cls, grid_config: GridConfig, rank: int = None, threshold_ratio: float = 1e-5) -> 'CCDCompositeSolver':
        """TSVD（切断特異値分解）を適用したソルバーを作成"""
        return cls(
            grid_config, 
            scaling="none", 
            regularization="tsvd", 
            regularization_params={"rank": rank, "threshold_ratio": threshold_ratio}
        )
    
    @classmethod
    def create_landweber_solver(
        cls, grid_config: GridConfig, iterations: int = 20, relaxation: float = 0.1
    ) -> 'CCDCompositeSolver':
        """Landweber反復法を適用したソルバーを作成"""
        return cls(
            grid_config, 
            scaling="none", 
            regularization="landweber", 
            regularization_params={"iterations": iterations, "relaxation": relaxation}
        )
    
    @classmethod
    def create_precomputed_landweber_solver(
        cls, grid_config: GridConfig, iterations: int = 20, relaxation: float = 0.1
    ) -> 'CCDCompositeSolver':
        """事前計算型Landweber反復法を適用したソルバーを作成"""
        return cls(
            grid_config, 
            scaling="none", 
            regularization="precomputed_landweber", 
            regularization_params={"iterations": iterations, "relaxation": relaxation}
        )
    
    @classmethod
    def create_lsqr_solver(
        cls, grid_config: GridConfig, iterations: int = 20, damp: float = 0
    ) -> 'CCDCompositeSolver':
        """LSQR法を適用したソルバーを作成"""
        return cls(
            grid_config, 
            scaling="none", 
            regularization="lsqr", 
            regularization_params={"iterations": iterations, "damp": damp}
        )
    
    @classmethod
    def create_total_variation_solver(
        cls, grid_config: GridConfig, alpha: float = 1e-4, iterations: int = 50, tol: float = 1e-6
    ) -> 'CCDCompositeSolver':
        """Total Variation 正則化を適用したソルバーを作成"""
        return cls(
            grid_config, 
            scaling="none", 
            regularization="total_variation", 
            regularization_params={"alpha": alpha, "iterations": iterations, "tol": tol}
        )
    
    @classmethod
    def create_l1_solver(
        cls, grid_config: GridConfig, alpha: float = 1e-4, iterations: int = 100, tol: float = 1e-6
    ) -> 'CCDCompositeSolver':
        """L1正則化（LASSO）を適用したソルバーを作成"""
        return cls(
            grid_config, 
            scaling="none", 
            regularization="l1", 
            regularization_params={"alpha": alpha, "iterations": iterations, "tol": tol}
        )
    
    @classmethod
    def create_elastic_net_solver(
        cls, grid_config: GridConfig, alpha: float = 1e-4, l1_ratio: float = 0.5, 
        iterations: int = 100, tol: float = 1e-6
    ) -> 'CCDCompositeSolver':
        """Elastic Net 正則化を適用したソルバーを作成"""
        return cls(
            grid_config, 
            scaling="none", 
            regularization="elastic_net", 
            regularization_params={
                "alpha": alpha, 
                "l1_ratio": l1_ratio, 
                "iterations": iterations, 
                "tol": tol
            }
        )
    
    @classmethod
    def available_scaling_methods(cls) -> List[str]:
        """利用可能なスケーリング手法のリストを返す"""
        return list(cls.SCALING_STRATEGIES.keys())
    
    @classmethod
    def available_regularization_methods(cls) -> List[str]:
        """利用可能な正則化手法のリストを返す"""
        return list(cls.REGULARIZATION_STRATEGIES.keys())


# 後方互換性のために従来のクラス名をエイリアスとして提供
CCDSolver = CCDCompositeSolver.create_basic_solver
NormalizationScalingSolver = CCDCompositeSolver.create_normalization_solver
RehuScalingSolver = CCDCompositeSolver.create_rehu_solver
EqualizationScalingSolver = CCDCompositeSolver.create_equalization_solver
IterativeScalingSolver = CCDCompositeSolver.create_iterative_solver
VanDerSluisScalingSolver = CCDCompositeSolver.create_van_der_sluis_solver
DiagonalDominanceScalingSolver = CCDCompositeSolver.create_diagonal_dominance_solver
SquareSumScalingSolver = CCDCompositeSolver.create_square_sum_solver
MaxElementScalingSolver = CCDCompositeSolver.create_max_element_solver

TikhonovRegularizedSolver = CCDCompositeSolver.create_tikhonov_solver
SVDRegularizedSolver = CCDCompositeSolver.create_svd_solver
TSVDRegularizedSolver = CCDCompositeSolver.create_tsvd_solver
LandweberIterativeSolver = CCDCompositeSolver.create_landweber_solver
PrecomputedLandweberSolver = CCDCompositeSolver.create_precomputed_landweber_solver
LSQRRegularizedSolver = CCDCompositeSolver.create_lsqr_solver
TotalVariationRegularizedSolver = CCDCompositeSolver.create_total_variation_solver
L1RegularizedSolver = CCDCompositeSolver.create_l1_solver
ElasticNetRegularizedSolver = CCDCompositeSolver.create_elastic_net_solver

# 合成ソルバーのエイリアス
CompositeSolver = CCDCompositeSolver