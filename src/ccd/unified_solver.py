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


class ScalingStrategy:
    """スケーリング戦略の基底クラス"""
    
    def __init__(self, L: jnp.ndarray, K: jnp.ndarray):
        """
        Args:
            L: スケーリングする行列
            K: スケーリングする右辺行列
        """
        self.L = L
        self.K = K
    
    def apply_scaling(self) -> Tuple[jnp.ndarray, jnp.ndarray, Callable]:
        """スケーリングを適用し、スケーリングされた行列と逆変換関数を返す"""
        # デフォルトはスケーリングなし
        return self.L, self.K, lambda x: x


class NoneScaling(ScalingStrategy):
    """スケーリングなし"""
    pass


class NormalizationScaling(ScalingStrategy):
    """行と列のノルムによるスケーリング"""
    
    def apply_scaling(self):
        # 行と列のL2ノルムを計算
        row_norms = jnp.sqrt(jnp.sum(self.L * self.L, axis=1))
        col_norms = jnp.sqrt(jnp.sum(self.L * self.L, axis=0))
        
        # スケーリング行列を作成
        D = jnp.diag(1.0 / jnp.sqrt(row_norms * col_norms))
        
        # スケーリングを適用
        L_scaled = D @ self.L @ D
        K_scaled = D @ self.K
        
        # 逆変換関数
        def inverse_scaling(X_scaled):
            return D @ X_scaled
        
        return L_scaled, K_scaled, inverse_scaling


class RehuScaling(ScalingStrategy):
    """Rehu法によるスケーリング"""
    
    def apply_scaling(self):
        # 行列の各行と列の最大絶対値を計算
        max_values_row = jnp.max(jnp.abs(self.L), axis=1)
        max_values_col = jnp.max(jnp.abs(self.L), axis=0)
        
        # スケーリング行列を作成
        D_row = jnp.diag(1.0 / jnp.sqrt(max_values_row))
        D_col = jnp.diag(1.0 / jnp.sqrt(max_values_col))
        
        # スケーリングを適用
        L_scaled = D_row @ self.L @ D_col
        K_scaled = D_row @ self.K
        
        # 逆変換関数
        def inverse_scaling(X_scaled):
            return D_col @ X_scaled
        
        return L_scaled, K_scaled, inverse_scaling


class EqualizationScaling(ScalingStrategy):
    """均等化スケーリング"""
    
    def apply_scaling(self):
        # 1. 行の均等化
        row_max = jnp.max(jnp.abs(self.L), axis=1)
        D_row = jnp.diag(1.0 / row_max)
        L_row_eq = D_row @ self.L
        K_row_eq = D_row @ self.K
        
        # 2. 列の均等化
        col_max = jnp.max(jnp.abs(L_row_eq), axis=0)
        D_col = jnp.diag(1.0 / col_max)
        
        # 3. スケーリングを適用
        L_scaled = L_row_eq @ D_col
        K_scaled = K_row_eq
        
        # 逆変換関数
        def inverse_scaling(X_scaled):
            return D_col @ X_scaled
        
        return L_scaled, K_scaled, inverse_scaling


class IterativeScaling(ScalingStrategy):
    """反復的スケーリング（Sinkhorn-Knopp法）"""
    
    def __init__(self, L: jnp.ndarray, K: jnp.ndarray, max_iter: int = 10, tol: float = 1e-8):
        """
        Args:
            L: スケーリングする行列
            K: スケーリングする右辺行列
            max_iter: 最大反復回数
            tol: 収束判定閾値
        """
        super().__init__(L, K)
        self.max_iter = max_iter
        self.tol = tol
    
    def apply_scaling(self):
        """Sinkhorn-Knopp法による反復的スケーリングを適用"""
        # 行列の絶対値を取得
        A = jnp.abs(self.L)
        
        # 行と列のスケーリングベクトルを初期化
        d_row = jnp.ones(A.shape[0])
        d_col = jnp.ones(A.shape[1])
        
        D_row = jnp.diag(d_row)
        D_col = jnp.diag(d_col)
        
        # 反復的にスケーリングを適用
        for _ in range(self.max_iter):
            # 行のスケーリング
            row_sums = jnp.sum(D_row @ A @ D_col, axis=1)
            d_row_new = 1.0 / jnp.sqrt(row_sums)
            D_row_new = jnp.diag(d_row_new)
            
            # 列のスケーリング
            col_sums = jnp.sum(D_row_new @ A @ D_col, axis=0)
            d_col_new = 1.0 / jnp.sqrt(col_sums)
            D_col_new = jnp.diag(d_col_new)
            
            # 収束判定
            if (jnp.max(jnp.abs(d_row_new - d_row)) < self.tol and 
                jnp.max(jnp.abs(d_col_new - d_col)) < self.tol):
                break
            
            d_row, d_col = d_row_new, d_col_new
            D_row, D_col = D_row_new, D_col_new
        
        # 最終的なスケーリング行列を保存
        self.D_row = D_row
        self.D_col = D_col
        
        # スケーリングを適用
        L_scaled = self.D_row @ self.L @ self.D_col
        K_scaled = self.D_row @ self.K
        
        # 逆変換関数
        def inverse_scaling(X_scaled):
            return self.D_col @ X_scaled
        
        return L_scaled, K_scaled, inverse_scaling


class RegularizationStrategy:
    """正則化戦略の基底クラス"""
    
    def __init__(self, L: jnp.ndarray, K: jnp.ndarray):
        """
        Args:
            L: 正則化する行列
            K: 正則化する右辺行列
        """
        self.L = L
        self.K = K
    
    def apply_regularization(self) -> Tuple[jnp.ndarray, jnp.ndarray, Callable]:
        """正則化を適用し、正則化された行列とソルバー関数を返す"""
        # デフォルトは正則化なし - 標準的な行列の解法
        def solver_func(rhs):
            return jnp.linalg.solve(self.L, rhs)
        
        return self.L, self.K, solver_func


class NoneRegularization(RegularizationStrategy):
    """正則化なし"""
    pass


class TikhonovRegularization(RegularizationStrategy):
    """Tikhonov正則化"""
    
    def __init__(self, L: jnp.ndarray, K: jnp.ndarray, alpha: float = 1e-6):
        super().__init__(L, K)
        self.alpha = alpha
    
    def apply_regularization(self):
        n = self.L.shape[0]
        # 単位行列を生成
        I = jnp.eye(n)
        # 行列を正則化
        L_reg = self.L + self.alpha * I
        
        # ソルバー関数
        def solver_func(rhs):
            return jnp.linalg.solve(L_reg, rhs)
        
        return L_reg, self.K, solver_func


class LandweberRegularization(RegularizationStrategy):
    """Landweber反復法による正則化"""
    
    def __init__(self, L: jnp.ndarray, K: jnp.ndarray, iterations: int = 20, relaxation: float = 0.1):
        super().__init__(L, K)
        self.iterations = iterations
        self.relaxation = relaxation
        self.L_T = L.T
    
    def apply_regularization(self):
        # 行列のスペクトルノルムを概算
        s_max = jnp.linalg.norm(self.L, ord=2)
        
        # 緩和パラメータを安全な範囲に調整
        omega = min(self.relaxation, 1.9 / (s_max ** 2))
        
        # ソルバー関数
        def solver_func(rhs):
            # 初期解を0に設定
            x = jnp.zeros_like(rhs)
            
            # Landweber反復
            for _ in range(self.iterations):
                # 残差: r = rhs - L @ x
                residual = rhs - self.L @ x
                # 反復更新: x = x + omega * L^T @ residual
                x = x + omega * (self.L_T @ residual)
            
            return x
        
        return self.L, self.K, solver_func


class PrecomputedLandweberRegularization(RegularizationStrategy):
    """事前計算型のLandweber反復法による正則化"""
    
    def __init__(self, L: jnp.ndarray, K: jnp.ndarray, iterations: int = 20, relaxation: float = 0.1):
        super().__init__(L, K)
        self.iterations = iterations
        self.relaxation = relaxation
        self.L_T = L.T
    
    def apply_regularization(self):
        # 行列のスペクトルノルムを概算
        s_max = jnp.linalg.norm(self.L, ord=2)
        
        # 緩和パラメータを安全な範囲に調整
        omega = min(self.relaxation, 1.9 / (s_max ** 2))
        
        n = self.L.shape[0]
        I = jnp.eye(n)
        
        # Landweber反復のマトリックス形式
        LTL = self.L_T @ self.L
        M = I - omega * LTL
        
        # M^n を計算
        M_power = I
        for _ in range(self.iterations):
            M_power = M_power @ M
        
        # 最終的な変換行列を計算
        I_minus_M_power = I - M_power
        
        # LTLの擬似逆行列を計算
        U, s, Vh = jnp.linalg.svd(LTL, full_matrices=False)
        threshold = jnp.max(s) * 1e-10
        s_inv = jnp.where(s > threshold, 1.0 / s, 0.0)
        LTL_pinv = Vh.T @ jnp.diag(s_inv) @ U.T
        
        # 最終的な変換行列
        transform_matrix = I_minus_M_power @ LTL_pinv @ self.L_T
        
        # ソルバー関数
        def solver_func(rhs):
            return transform_matrix @ rhs
        
        return self.L, self.K, solver_func


class SVDRegularization(RegularizationStrategy):
    """SVD切断法による正則化"""
    
    def __init__(self, L: jnp.ndarray, K: jnp.ndarray, threshold: float = 1e-10):
        super().__init__(L, K)
        self.threshold = threshold
    
    def apply_regularization(self):
        # 特異値分解を実行
        U, s, Vh = jnp.linalg.svd(self.L, full_matrices=False)
        
        # 特異値のフィルタリング
        s_filtered = jnp.where(s > self.threshold, s, self.threshold)
        
        # 擬似逆行列を計算
        pinv = Vh.T @ jnp.diag(1.0 / s_filtered) @ U.T
        
        # ソルバー関数
        def solver_func(rhs):
            return pinv @ rhs
        
        return self.L, self.K, solver_func


class CCDCompositeSolver(CCDSolver):
    """スケーリングと正則化を組み合わせた統合ソルバー"""
    
    # 利用可能なスケーリングと正則化の戦略のマッピング
    SCALING_STRATEGIES = {
        "none": NoneScaling,
        "normalization": NormalizationScaling,
        "rehu": RehuScaling,
        "equalization": EqualizationScaling,
        "iterative": IterativeScaling
    }
    
    REGULARIZATION_STRATEGIES = {
        "none": NoneRegularization,
        "tikhonov": TikhonovRegularization,
        "landweber": LandweberRegularization,
        "precomputed_landweber": PrecomputedLandweberRegularization,
        "svd": SVDRegularization
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
            scaling: スケーリング戦略 ("none", "normalization", "rehu", "equalization")
            regularization: 正則化戦略 ("none", "tikhonov", "landweber", "precomputed_landweber", "svd")
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
        
        # IterativeScaling の場合はパラメータを渡す
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
        elif strategy_class in (LandweberRegularization, PrecomputedLandweberRegularization):
            iterations = self.regularization_params.get("iterations", 20)
            relaxation = self.regularization_params.get("relaxation", 0.1)
            return strategy_class(L, K, iterations, relaxation)
        elif strategy_class == SVDRegularization:
            threshold = self.regularization_params.get("threshold", 1e-10)
            return strategy_class(L, K, threshold)
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
        """導関数を計算
        
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
    def create_iterative_solver(cls, grid_config: GridConfig, max_iter: int = 10, tol: float = 1e-8) -> 'CCDCompositeSolver':
        """反復的スケーリング（Sinkhorn-Knopp法）を適用したソルバーを作成"""
        return cls(
            grid_config, 
            scaling="iterative", 
            regularization="none", 
            scaling_params={"max_iter": max_iter, "tol": tol}
        )
    
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
IterativeScalingSolver = CCDCompositeSolver.create_iterative_solver
TikhonovRegularizedSolver = CCDCompositeSolver.create_tikhonov_solver
LandweberIterativeSolver = CCDCompositeSolver.create_landweber_solver
PrecomputedLandweberSolver = CCDCompositeSolver.create_precomputed_landweber_solver

# 合成ソルバーのエイリアス
CompositeSolver = CCDCompositeSolver