# preconditioner/jacobi.py
"""
ヤコビ前処理

このモジュールは、行列の対角成分の逆数を用いた
シンプルで効率的な前処理手法を提供します。
"""

import numpy as np
from .base import BasePreconditioner

class JacobiPreconditioner(BasePreconditioner):
    """ヤコビ前処理（対角成分の逆数）"""
    
    def __init__(self, epsilon=1e-14):
        """
        初期化
        
        Args:
            epsilon: ゼロ除算回避用の小さな値
        """
        super().__init__()
        self.epsilon = epsilon
        self.diag_vals = None
    
    def setup(self, A):
        """
        ヤコビ前処理の設定
        
        Args:
            A: システム行列
            
        Returns:
            self: メソッドチェーン用
        """
        # 対角成分を抽出
        if hasattr(A, 'diagonal'):
            diag = A.diagonal()
        elif hasattr(A, 'toarray'):
            diag = A.toarray().diagonal()
        else:
            diag = np.diag(A)
        
        # CuPy/JAX配列をNumPyに変換
        if hasattr(diag, 'get'):
            diag = diag.get()
        elif 'jax' in str(type(diag)):
            diag = np.array(diag)
        
        # 対角成分の逆数を計算（ゼロ除算回避）
        self.diag_vals = 1.0 / (diag + (np.abs(diag) < self.epsilon) * self.epsilon)
        return self
    
    def __call__(self, b):
        """
        前処理を適用 (D⁻¹b)
        
        Args:
            b: ベクトル
            
        Returns:
            前処理したベクトル
        """
        if self.diag_vals is None:
            return b
            
        # ベクトル形式に応じて処理
        if 'cupy' in str(type(b)):  # CuPy
            import cupy as cp
            return cp.multiply(cp.array(self.diag_vals), b)
        elif 'jax' in str(type(b)):  # JAX
            import jax.numpy as jnp
            return jnp.multiply(jnp.array(self.diag_vals), b)
        else:  # NumPy
            return np.multiply(self.diag_vals, b)
    
    @property
    def description(self):
        """前処理器の説明"""
        return f"ヤコビ前処理（対角成分の逆数、epsilon={self.epsilon}）"