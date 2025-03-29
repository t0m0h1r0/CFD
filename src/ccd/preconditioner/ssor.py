# preconditioner/ssor.py
"""
対称的逐次緩和法前処理 (SSOR)

このモジュールは、対称的逐次緩和法を用いた
効率的な前処理手法を提供します。
"""

import numpy as np
from scipy import sparse
from .base import BasePreconditioner

class SSORPreconditioner(BasePreconditioner):
    """対称的逐次緩和法前処理"""
    
    def __init__(self, omega=1.0, epsilon=1e-14):
        """
        初期化
        
        Args:
            omega: 緩和パラメータ (0 < omega < 2)
            epsilon: ゼロ除算回避用の小さな値
        """
        super().__init__()
        self.omega = max(0.0, min(omega, 2.0))  # 範囲を制限
        self.epsilon = epsilon
        self.D = None
        self.L = None
        self.U = None
    
    def setup(self, A):
        """SSOR前処理の設定"""
        # CSR行列に変換
        if hasattr(A, 'tocsr'):
            A_csr = A.tocsr()
        elif hasattr(A, 'toarray'):
            A_csr = sparse.csr_matrix(A.toarray())
        else:
            A_csr = sparse.csr_matrix(A)
        
        # CuPy/JAX配列の場合はNumPyに変換
        if hasattr(A_csr, 'get'):
            A_csr = A_csr.get()
        
        n = A_csr.shape[0]
        
        # 対角・下三角・上三角部分に分解
        D = sparse.diags(A_csr.diagonal())
        L = sparse.tril(A_csr, k=-1)
        U = sparse.triu(A_csr, k=1)
        
        # ゼロ除算回避
        D_data = D.diagonal()
        D_data = D_data + (np.abs(D_data) < self.epsilon) * self.epsilon
        D = sparse.diags(D_data)
        
        # 前処理に必要な行列を保存
        self.D = D
        self.L = L
        self.U = U
        
        return self
    
    def __call__(self, b):
        """
        前処理を適用（近似解を返す）
        
        Args:
            b: 右辺ベクトル
            
        Returns:
            前処理したベクトル
        """
        if self.D is None or self.L is None or self.U is None:
            return b
            
        # NumPy配列に変換
        if hasattr(b, 'get'):  # CuPy
            b_np = b.get()
        elif 'jax' in str(type(b)):  # JAX
            b_np = np.array(b)
        else:
            b_np = b
            
        try:
            # SSOR前処理を適用 (D/ω)^(-1) * (I + ω*D^(-1)*L)^(-1) * (I + ω*D^(-1)*U)^(-1)
            # 前方代入
            D_inv = sparse.diags(1.0 / self.D.diagonal())
            L_term = sparse.eye(self.D.shape[0]) + self.omega * D_inv @ self.L
            U_term = sparse.eye(self.D.shape[0]) + self.omega * D_inv @ self.U
            
            # 反復的な前方/後方代入（簡略化した近似解法）
            temp = b_np.copy()
            # 前方代入 (I + ω*D^(-1)*L) * z = b
            temp = sparse.linalg.spsolve_triangular(L_term, temp, lower=True)
            # 対角スケーリング z = (D/ω)^(-1) * z
            temp = (self.omega / self.D.diagonal()) * temp
            # 後方代入 (I + ω*D^(-1)*U) * x = z
            x_np = sparse.linalg.spsolve_triangular(U_term, temp, lower=False)
            
            # 元の形式に戻す
            if 'cupy' in str(type(b)):  # CuPy
                import cupy as cp
                return cp.array(x_np)
            elif 'jax' in str(type(b)):  # JAX
                import jax.numpy as jnp
                return jnp.array(x_np)
            else:
                return x_np
                
        except Exception as e:
            print(f"SSOR前処理適用エラー: {e}")
            return b
    
    @property
    def description(self):
        """前処理器の説明"""
        return f"対称的逐次緩和法前処理 (omega={self.omega})"