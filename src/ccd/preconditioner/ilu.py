# preconditioner/ilu.py
"""
不完全LU分解前処理

このモジュールは、システム行列の不完全LU分解に基づく
効率的な前処理手法を提供します。
"""

import numpy as np
from .base import BasePreconditioner

class ILUPreconditioner(BasePreconditioner):
    """不完全LU分解前処理"""
    
    def __init__(self, fill_factor=10, drop_tol=1e-4):
        """
        初期化
        
        Args:
            fill_factor: 充填因子（メモリ使用量制御）
            drop_tol: 切り捨て許容値（精度制御）
        """
        super().__init__()
        self.fill_factor = fill_factor
        self.drop_tol = drop_tol
        self.ilu = None
    
    def setup(self, A):
        """
        不完全LU分解前処理の設定
        
        Args:
            A: システム行列
            
        Returns:
            self: メソッドチェーン用
        """
        try:
            from scipy.sparse.linalg import spilu
            from scipy.sparse import csc_matrix
            
            # CSC形式に変換（必要な場合）
            if hasattr(A, 'tocsc'):
                A_csc = A.tocsc()
            elif hasattr(A, 'toarray'):
                A_csc = csc_matrix(A.toarray())
            else:
                A_csc = csc_matrix(A)
            
            # CuPy/JAX配列をNumPyに変換
            if hasattr(A_csc, 'get'):
                A_csc = A_csc.get()
            
            # 不完全LU分解を計算
            self.ilu = spilu(A_csc, 
                           fill_factor=self.fill_factor,
                           drop_tol=self.drop_tol)
            return self
            
        except Exception as e:
            print(f"ILU分解エラー: {e}")
            return self
    
    def __call__(self, b):
        """
        前処理を適用（近似解を返す）
        
        Args:
            b: 右辺ベクトル
            
        Returns:
            前処理したベクトル
        """
        if self.ilu is None:
            return b
            
        # NumPy配列に変換
        if hasattr(b, 'get'):  # CuPy
            b_np = b.get()
        elif 'jax' in str(type(b)):  # JAX
            b_np = np.array(b)
        else:
            b_np = b
            
        # ILUを使って近似解を計算
        x_np = self.ilu.solve(b_np)
        
        # 元の形式に戻す
        if 'cupy' in str(type(b)):  # CuPy
            import cupy as cp
            return cp.array(x_np)
        elif 'jax' in str(type(b)):  # JAX
            import jax.numpy as jnp
            return jnp.array(x_np)
        else:
            return x_np
    
    @property
    def description(self):
        """前処理器の説明"""
        return f"不完全LU分解前処理 (fill_factor={self.fill_factor}, drop_tol={self.drop_tol})"