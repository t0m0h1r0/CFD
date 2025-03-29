"""
単位行列前処理

このモジュールは、前処理なし（単位行列）の基本的な前処理クラスを提供します。
主にデフォルト前処理として、または前処理効果の基準比較用に使用されます。
"""

import numpy as np
import scipy.sparse as sp
from .base import BasePreconditioner

class IdentityPreconditioner(BasePreconditioner):
    """単位行列前処理（実質的に前処理なし）"""
    
    def __init__(self):
        """初期化"""
        super().__init__()
        self.M = None
    
    def setup(self, A):
        """
        単位行列前処理の設定
        
        Args:
            A: システム行列
            
        Returns:
            self: メソッドチェーン用
        """
        try:
            # 行列サイズを取得
            if hasattr(A, 'shape'):
                n = A.shape[0]
                
                # 明示的に単位行列を作成
                # 注: 単位行列を直接設定することで、視覚化に使用できるようにする
                if hasattr(A, 'format') or 'sparse' in str(type(A)):
                    # 疎行列用の単位行列
                    self.M = sp.eye(n, format='csr')
                else:
                    # 密行列用の単位行列
                    self.M = np.eye(n)
                
                print(f"単位行列前処理を設定しました (サイズ: {n}x{n})")
            else:
                print("警告: 行列の形状が取得できません")
                self.M = None
        except Exception as e:
            print(f"単位行列前処理設定エラー: {e}")
            self.M = None
            
        return self
    
    def __call__(self, b):
        """
        単位行列前処理を適用（実質的に変更なし）
        
        Args:
            b: 右辺ベクトル
            
        Returns:
            前処理適用後のベクトル（変更なし）
        """
        # 単位行列なので入力をそのまま返す
        return b
    
    @property
    def name(self):
        """前処理名を返す"""
        return "IdentityPreconditioner"
    
    @property
    def description(self):
        """前処理の説明を返す"""
        return "単位行列前処理（実質的に前処理なし）"