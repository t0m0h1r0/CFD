"""
相似変換による対称化スケーリング戦略

相似変換を用いて左辺行列を対称形に変換する高度なスケーリング手法を提供します。
"""

import jax.numpy as jnp
from typing import Tuple, Dict, Any, Callable

from scaling_strategy import ScalingStrategy, scaling_registry


class SimilaritySymmetricScaling(ScalingStrategy):
    """
    相似変換による対称化スケーリング
    
    相似変換 S = D^(-1/2) A D^(1/2) を用いて非対称行列Aを対称行列Sに変換します。
    ここでDは対角行列で、適切に選ぶことで対称化が可能です。
    この手法はより洗練された対称化手法であり、特定の条件下では元の問題と完全に等価です。
    """
    
    def _init_params(self, **kwargs):
        """
        パラメータの初期化
        
        Args:
            **kwargs: 初期化パラメータ
        """
        self.method = kwargs.get('method', 'equilibration')
        self.tol = kwargs.get('tol', 1e-10)
        self.max_iter = kwargs.get('max_iter', 10)
    
    @classmethod
    def get_param_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        パラメータ情報を返す
        
        Returns:
            パラメータ情報の辞書
        """
        return {
            'method': {
                'type': str,
                'default': 'equilibration',
                'help': '対称化の方法（equilibration または diag_dominant）'
            },
            'tol': {
                'type': float,
                'default': 1e-10,
                'help': '収束判定閾値'
            },
            'max_iter': {
                'type': int,
                'default': 10,
                'help': '最大反復回数'
            }
        }
    
    def apply_scaling(self) -> Tuple[jnp.ndarray, Callable[[jnp.ndarray], jnp.ndarray]]:
        """
        相似変換による対称化スケーリングを適用
        
        数学的背景:
        1. 行列A∈R^(n×n)と正の対角行列D∈R^(n×n)に対して、相似変換
           S = D^(-1/2) A D^(1/2) が対称行列となるための必要十分条件は、
           A D が対称行列となることです。
        
        2. 実用的な方法では、反復的に以下を計算します:
           - Row scaling: r_i = 1/sqrt(∑_j |a_ij|)
           - Column scaling: c_j = 1/sqrt(∑_i |a_ij|)
           - 対角行列 D = diag(r_1,...,r_n) を構築
           - 変換 A' = D A D を適用
        
        Returns:
            (対称化された行列, 逆変換関数)
        """
        # 行列のコピーを作成
        A = self.matrix.copy()
        n = A.shape[0]
        
        if self.method == 'equilibration':
            # 平衡化による対称化（Ruiz法に基づく）
            d = jnp.ones(n)  # 対角スケーリング係数
            
            # 行と列のスケーリング（固定反復回数）
            for _ in range(self.max_iter):
                # 行のスケーリング
                row_norms = jnp.sqrt(jnp.sum(A * A, axis=1))
                row_norms = jnp.where(row_norms < self.tol, 1.0, row_norms)
                Dr = jnp.diag(1.0 / jnp.sqrt(row_norms))
                A = Dr @ A @ Dr
                
                # スケーリング係数の更新
                d = d * jnp.diag(Dr)
            
            # 最終的な対角スケーリング行列
            D = jnp.diag(d)
            
        else:  # diag_dominant
            # 対角優位性に基づく対称化
            # 各行の非対角要素と対角要素の比率に基づくスケーリング
            diag_elements = jnp.diag(A)
            diag_elements = jnp.where(jnp.abs(diag_elements) < self.tol, self.tol, diag_elements)
            
            # 対角行列の構築（対角優位性を増強）
            D_vals = jnp.ones(n)
            for i in range(n):
                row = A[i, :]
                row_without_diag = jnp.concatenate([row[:i], row[i+1:]])
                max_off_diag = jnp.max(jnp.abs(row_without_diag))
                if max_off_diag > jnp.abs(diag_elements[i]):
                    D_vals = D_vals.at[i].set(jnp.sqrt(max_off_diag / jnp.abs(diag_elements[i])))
            
            D = jnp.diag(D_vals)
        
        # D^(-1/2) の計算
        D_neg_half = jnp.diag(1.0 / jnp.sqrt(jnp.diag(D)))
        
        # D^(1/2) の計算
        D_half = jnp.diag(jnp.sqrt(jnp.diag(D)))
        
        # 相似変換による対称化: S = D^(-1/2) A D^(1/2)
        S = D_neg_half @ A @ D_half
        
        # 対称性を強制（数値誤差対策）
        S = (S + S.T) / 2.0
        
        # スケーリング行列を保存
        self.scaling_matrix_row = D_neg_half
        self.scaling_matrix_col = D_half
        
        # 逆変換関数
        def inverse_scaling(X_scaled):
            # 対称化問題の解から元の問題の解への変換
            # x = D^(1/2) x_sym
            return D_half @ X_scaled
        
        return S, inverse_scaling
    
    def transform_rhs(self, rhs: jnp.ndarray) -> jnp.ndarray:
        """
        右辺ベクトルに対称化の変換を適用
        
        相似変換 S = D^(-1/2) A D^(1/2) に対応して、
        右辺ベクトル b も b' = D^(-1/2) b に変換する必要があります。
        
        Args:
            rhs: 変換する右辺ベクトル
            
        Returns:
            変換された右辺ベクトル
        """
        # 行スケーリングのみを右辺ベクトルに適用
        if hasattr(self, 'scaling_matrix_row'):
            return self.scaling_matrix_row @ rhs
        return rhs


# スケーリング戦略をレジストリに登録
scaling_registry.register("similarity_symmetric", SimilaritySymmetricScaling)
