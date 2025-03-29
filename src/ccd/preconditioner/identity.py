"""
単位行列前処理

このモジュールは、前処理なし（単位行列）の基本的な前処理クラスを提供します。
主にデフォルト前処理として、または前処理効果の基準比較用に使用されます。
"""

from .base import BasePreconditioner

class IdentityPreconditioner(BasePreconditioner):
    """単位行列前処理（実質的に前処理なし）"""
    
    def __init__(self):
        """初期化"""
        super().__init__()
    
    def setup(self, A):
        """
        単位行列前処理の設定（実質的に何もしない）
        
        Args:
            A: システム行列
            
        Returns:
            self: メソッドチェーン用
        """
        # 単位行列前処理は何も設定しない
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
        return b
    
    @property
    def name(self):
        """前処理名を返す"""
        return "IdentityPreconditioner"
    
    @property
    def description(self):
        """前処理の説明を返す"""
        return "単位行列前処理（実質的に前処理なし）"