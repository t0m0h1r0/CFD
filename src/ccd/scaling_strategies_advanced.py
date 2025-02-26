"""
高度なスケーリング戦略モジュール

CCD法の高度なスケーリング戦略（Equalization, Iterative, Van der Sluis, Diagonal Dominance等）を提供します。
"""

import jax.numpy as jnp

from scaling_strategies_base import ScalingStrategy


class EqualizationScaling(ScalingStrategy):
    """均等化スケーリング"""
    
    def apply_scaling(self):
        """
        均等化スケーリングを適用
        
        各行と列の最大絶対値を1にスケーリングします。
        """
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
        """
        Sinkhorn-Knopp法による反復的スケーリングを適用
        
        行と列の和を反復的に均等化する手法で、数値的に安定したスケーリングを実現します。
        """
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


class VanDerSluisScaling(ScalingStrategy):
    """Van der Sluis スケーリング"""
    
    def apply_scaling(self):
        """
        Van der Sluis スケーリングを適用
        
        各列を2-ノルムで正規化するスケーリング手法です。
        行列の列間でスケールが大きく異なる場合に特に効果的です。
        """
        # 各列の2-ノルムを計算
        col_norms = jnp.sqrt(jnp.sum(self.L * self.L, axis=0))
        
        # スケーリング行列を作成（列のみスケーリング）
        D_col = jnp.diag(1.0 / col_norms)
        
        # スケーリングを適用
        L_scaled = self.L @ D_col
        K_scaled = self.K
        
        # 逆変換関数
        def inverse_scaling(X_scaled):
            return D_col @ X_scaled
        
        return L_scaled, K_scaled, inverse_scaling


class DiagonalDominanceScaling(ScalingStrategy):
    """対角優位スケーリング"""
    
    def apply_scaling(self):
        """
        対角優位スケーリングを適用
        
        対角要素が1になるようスケーリングし、他の要素は相対的に小さくなります。
        数値解法の収束特性の改善に役立ちます。
        """
        n = self.L.shape[0]
        diag_elements = jnp.diag(self.L)
        
        # 対角要素が0の場合に備えて小さな値を加える
        diag_elements = jnp.where(jnp.abs(diag_elements) < 1e-10, 1e-10, diag_elements)
        
        # スケーリング行列を作成
        D = jnp.diag(1.0 / diag_elements)
        
        # スケーリングを適用
        L_scaled = D @ self.L @ D
        K_scaled = D @ self.K
        
        # 逆変換関数
        def inverse_scaling(X_scaled):
            return D @ X_scaled
        
        return L_scaled, K_scaled, inverse_scaling


class SquareSumScaling(ScalingStrategy):
    """二乗和スケーリング（Sum of Squares Scaling）"""
    
    def apply_scaling(self):
        """
        二乗和スケーリングを適用
        
        各行と列の要素の二乗和が等しくなるようスケーリングします。
        特異値分解の前処理として効果的です。
        """
        # 1. 行の二乗和を計算
        row_sqr_sums = jnp.sqrt(jnp.sum(self.L * self.L, axis=1))
        D_row = jnp.diag(1.0 / row_sqr_sums)
        L_row_scaled = D_row @ self.L
        
        # 2. 列の二乗和を計算
        col_sqr_sums = jnp.sqrt(jnp.sum(L_row_scaled * L_row_scaled, axis=0))
        D_col = jnp.diag(1.0 / col_sqr_sums)
        
        # 3. スケーリングを適用
        L_scaled = L_row_scaled @ D_col
        K_scaled = D_row @ self.K
        
        # 逆変換関数
        def inverse_scaling(X_scaled):
            return D_col @ X_scaled
        
        return L_scaled, K_scaled, inverse_scaling


class MaxElementScaling(ScalingStrategy):
    """最大成分スケーリング（Max Element Scaling）"""
    
    def apply_scaling(self):
        """
        最大成分スケーリングを適用
        
        行列全体の最大絶対値が1になるようスケーリングします。
        非常にシンプルなスケーリング手法です。
        """
        # 行列全体の最大絶対値を取得
        max_abs_value = jnp.max(jnp.abs(self.L))
        
        # スケーリングを適用
        L_scaled = self.L / max_abs_value
        K_scaled = self.K / max_abs_value
        
        # 逆変換関数
        def inverse_scaling(X_scaled):
            return X_scaled  # この方式では逆変換の必要なし
        
        return L_scaled, K_scaled, inverse_scaling