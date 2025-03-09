import cupy as cp
import cupyx.scipy.sparse as sp
from .base import BaseScaling

class SymmetricScaling(BaseScaling):
    """対称スケーリング手法"""
    
    def scale(self, A, b):
        """
        対角要素を使ってAを対称的にスケーリング (D^-1/2 * A * D^-1/2)
        
        Args:
            A: システム行列（スパース）
            b: 右辺ベクトル
            
        Returns:
            tuple: (scaled_A, scaled_b, scale_info)
        """
        # 対角要素を取得
        diag = A.diagonal()
        
        # ゼロ除算を避けるため、ゼロを1に置き換え
        diag = cp.where(cp.abs(diag) < 1e-15, 1.0, diag)
        
        # D^-1/2 を計算
        D_sqrt_inv = cp.sqrt(1.0 / cp.abs(diag))
        D_sqrt_inv_mat = sp.diags(D_sqrt_inv)
        
        # Aをスケーリング
        scaled_A = D_sqrt_inv_mat @ A @ D_sqrt_inv_mat
        
        # bをスケーリング
        scaled_b = D_sqrt_inv_mat @ b
        
        # アンスケーリング用の情報を保存
        scale_info = {'D_sqrt_inv': D_sqrt_inv}
        
        return scaled_A, scaled_b, scale_info
    
    def unscale(self, x, scale_info):
        """
        解ベクトルをアンスケーリング
        
        Args:
            x: 解ベクトル
            scale_info: D_sqrt_invを含むスケーリング情報
            
        Returns:
            unscaled_x: アンスケーリングされた解ベクトル
        """
        D_sqrt_inv = scale_info['D_sqrt_inv']
        unscaled_x = x / D_sqrt_inv
        return unscaled_x
    
    @property
    def name(self):
        return "SymmetricScaling"
    
    @property
    def description(self):
        return "対角要素を使用した対称スケーリング (D^-1/2 * A * D^-1/2)"