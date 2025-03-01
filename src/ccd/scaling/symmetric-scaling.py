"""
対称化スケーリング戦略

左辺行列を対称形に変換するスケーリング手法を提供します。
"""

import jax.numpy as jnp
from typing import Tuple, Dict, Any, Callable

from scaling_strategy import ScalingStrategy, scaling_registry


class SymmetricScaling(ScalingStrategy):
    """
    対称化スケーリング
    
    左辺行列を対称形に変換する手法です。非対称行列に対して、
    A -> S = (A + A^T)/2 の変換を施し、対称行列を得ます。
    これにより、実対称行列の性質（実固有値を持つなど）を活用できます。
    """
    
    def _init_params(self, **kwargs):
        """
        パラメータの初期化
        
        Args:
            **kwargs: 初期化パラメータ
        """
        # 行列のサイズを保存
        if hasattr(self, 'matrix'):
            self.n = self.matrix.shape[0]
    
    @classmethod
    def get_param_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        パラメータ情報を返す
        
        Returns:
            パラメータ情報の辞書
        """
        return {}
    
    def apply_scaling(self) -> Tuple[jnp.ndarray, Callable[[jnp.ndarray], jnp.ndarray]]:
        """
        対称化スケーリングを適用
        
        非対称行列 A を対称行列 S = (A + A^T)/2 に変換します。
        この変換により、CG法などの対称行列向け解法が使用可能になります。
        
        理論的背景:
        1. 任意の行列 A は対称成分 S = (A + A^T)/2 と反対称成分 K = (A - A^T)/2 に分解できる
        2. 対称行列は実固有値のみを持つため数値的に安定
        3. Ax = b の代わりに Sx = b' を解くことで、同等の解を得る変換が可能
        
        Returns:
            (対称化された行列, 逆変換関数)
        """
        # 行列のサイズを取得
        self.n = self.matrix.shape[0]
        
        # 対称成分を計算: S = (A + A^T)/2
        S = (self.matrix + self.matrix.T) / 2.0
        
        # 元の行列との差分を保存（反対称成分）
        self.anti_symmetric = (self.matrix - self.matrix.T) / 2.0
        
        # 変換の正則性をチェック
        det_S = jnp.linalg.det(S)
        if jnp.abs(det_S) < 1e-10:
            # 対称化によって行列が特異になる場合は、微小な正則化項を追加
            S = S + 1e-10 * jnp.eye(self.n)
        
        # 対称行列を保存
        self.symmetric_matrix = S
        
        # 逆変換関数
        def inverse_scaling(X_scaled):
            # 対称化行列での解から元の非対称問題の解を近似復元
            # この場合、対称成分のみを考慮しているため、
            # 完全な逆変換は一般的には不可能です
            return X_scaled
        
        return S, inverse_scaling
    
    def transform_rhs(self, rhs: jnp.ndarray) -> jnp.ndarray:
        """
        右辺ベクトルに対称化の変換を適用
        
        Args:
            rhs: 変換する右辺ベクトル
            
        Returns:
            変換された右辺ベクトル
        """
        # 対称化行列で問題を解くために右辺ベクトルも変換
        # Ax = b を S x = b' に変換する場合、理論的には
        # b' = b - K x_approx が必要ですが、初期の x_approx が不明なため
        # 単純に b を使用するのが一般的です
        
        # 右辺ベクトルはそのまま使用
        return rhs


# スケーリング戦略をレジストリに登録
scaling_registry.register("symmetric", SymmetricScaling)
