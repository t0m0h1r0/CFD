# preconditioner/identity.py
"""
単位行列前処理

このモジュールは、前処理なし（単位行列）の基本的な前処理クラスを提供します。
主にデフォルト前処理として、または前処理効果のベースラインとして使用されます。
"""

from .base import BasePreconditioner

class IdentityPreconditioner(BasePreconditioner):
    """単位行列前処理（実質的に前処理なし）"""
    
    def setup(self, A):
        """
        単位行列前処理の設定（何もしない）
        
        Args:
            A: システム行列
            
        Returns:
            self: メソッドチェーン用
        """
        # 単位行列は設定不要
        return self
    
    def __call__(self, b):
        """
        前処理を適用（変更なし）
        
        Args:
            b: ベクトル
            
        Returns:
            入力と同じベクトル
        """
        return b
    
    @property
    def description(self):
        """前処理器の説明"""
        return "単位行列前処理（実質的に前処理なし）"