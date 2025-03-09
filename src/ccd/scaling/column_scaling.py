import cupy as cp
import cupyx.scipy.sparse as sp
from .base import BaseScaling

class ColumnScaling(BaseScaling):
    """列スケーリング手法"""
    
    def __init__(self, norm_type=2):
        """
        ノルム型を指定して初期化
        
        Args:
            norm_type: 列スケーリングに使用するノルム型（デフォルト: 2-ノルム）
        """
        self.norm_type = norm_type
    
    def scale(self, A, b):
        """
        Aの各列をそのノルムでスケーリング
        
        Args:
            A: システム行列（スパース）
            b: 右辺ベクトル
            
        Returns:
            tuple: (scaled_A, scaled_b, scale_info)
        """
        # 列ノルムを計算
        col_norms = cp.zeros(A.shape[1])
        for j in range(A.shape[1]):
            col = A[:, j].toarray().flatten()
            col_norms[j] = cp.linalg.norm(col, ord=self.norm_type)
        
        # ゼロ除算を避けるため、ゼロを1に置き換え
        col_norms = cp.where(col_norms < 1e-15, 1.0, col_norms)
        
        # スケーリング対角行列を作成
        D_inv = sp.diags(1.0 / col_norms)
        
        # Aをスケーリング
        scaled_A = A @ D_inv
        
        # bは変更しない
        scaled_b = b
        
        # アンスケーリング用の情報を保存
        scale_info = {'col_norms': col_norms}
        
        return scaled_A, scaled_b, scale_info
    
    def unscale(self, x, scale_info):
        """
        解ベクトルをアンスケーリング
        
        Args:
            x: 解ベクトル
            scale_info: 列ノルムを含むスケーリング情報
            
        Returns:
            unscaled_x: アンスケーリングされた解ベクトル
        """
        col_norms = scale_info['col_norms']
        unscaled_x = x * col_norms
        return unscaled_x
    
    @property
    def name(self):
        return "ColumnScaling"
    
    @property
    def description(self):
        return "各列をそのノルムでスケーリング"