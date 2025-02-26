"""
基本的なスケーリング戦略モジュール

CCD法の基本的なスケーリング戦略（Normalization, Rehu）を提供します。
"""

import jax.numpy as jnp

from scaling_strategies_base import ScalingStrategy


class NormalizationScaling(ScalingStrategy):
    """行と列のノルムによるスケーリング"""
    
    def apply_scaling(self):
        """
        行と列のL2ノルムでスケーリングを適用する
        
        各要素を行と列のL2ノルムの平方根で割ることでスケーリング
        """
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
        """
        Rehu法（行と列の最大絶対値）によるスケーリングを適用
        
        各行と列の最大絶対値の平方根でスケーリングする効果的な方法
        """
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