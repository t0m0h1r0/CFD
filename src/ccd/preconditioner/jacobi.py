"""
ヤコビ前処理

このモジュールは、行列の対角成分の逆数を用いた
単純かつ効率的な前処理手法を提供します。
"""

import numpy as np
from .base import BasePreconditioner

class JacobiPreconditioner(BasePreconditioner):
    """ヤコビ前処理（対角成分の逆数を使用）"""
    
    def __init__(self, epsilon=1e-14):
        """
        初期化
        
        Args:
            epsilon: ゼロ除算回避用の小さな値
        """
        super().__init__()
        self.epsilon = epsilon
    
    def setup(self, A):
        """
        ヤコビ前処理の設定
        
        Args:
            A: システム行列
            
        Returns:
            self: メソッドチェーン用
        """
        try:
            # 対角成分の抽出
            if hasattr(A, 'diagonal'):
                diag = A.diagonal()
            elif hasattr(A, 'toarray'):
                # diagonal()メソッドがない疎行列用
                diag = A.toarray().diagonal()
            else:
                # 密行列用
                diag = np.diag(A)
            
            # CuPy配列の場合
            if hasattr(diag, 'get'):
                diag = diag.get()  # NumPyに変換
            
            # 対角成分の逆数計算（ゼロ除算回避）
            inv_diag = 1.0 / (diag + (np.abs(diag) < self.epsilon) * self.epsilon)
            
            # 対角行列または演算子として保存
            if self.is_sparse(A):
                # 疎行列用
                try:
                    from scipy import sparse
                    self.M = sparse.diags(inv_diag)
                except ImportError:
                    # フォールバック
                    self.M = np.diag(inv_diag)
            else:
                # 密行列用
                self.M = np.diag(inv_diag)
        
        except Exception as e:
            print(f"ヤコビ前処理設定エラー: {e}")
            self.M = None
        
        return self
    
    @property
    def name(self):
        """前処理名を返す"""
        return "JacobiPreconditioner"
    
    @property
    def description(self):
        """前処理の説明を返す"""
        return "ヤコビ前処理（対角成分の逆数）"