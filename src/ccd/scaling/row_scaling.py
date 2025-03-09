import cupy as cp
import cupyx.scipy.sparse as sp
from .base import BaseScaling

class RowScaling(BaseScaling):
    """行スケーリング手法"""
    
    def __init__(self, norm_type=float('inf')):
        """
        ノルム型を指定して初期化
        
        Args:
            norm_type: 行スケーリングに使用するノルム型（デフォルト: 無限大ノルム）
        """
        self.norm_type = norm_type
    
    def scale(self, A, b):
        """
        Aの各行をそのノルムでスケーリングし、bも対応して調整
        
        Args:
            A: システム行列（スパース）
            b: 右辺ベクトル
            
        Returns:
            tuple: (scaled_A, scaled_b, scale_info)
        """
        # 行ノルムを計算
        row_norms = cp.zeros(A.shape[0])
        for i in range(A.shape[0]):
            row = A[i, :].toarray().flatten()
            row_norms[i] = cp.linalg.norm(row, ord=self.norm_type)
        
        # ゼロ除算を避けるため、ゼロを1に置き換え
        row_norms = cp.where(row_norms < 1e-15, 1.0, row_norms)
        
        # スケーリング対角行列を作成
        D_inv = sp.diags(1.0 / row_norms)
        
        # AとbをスケーリングF
        scaled_A = D_inv @ A
        scaled_b = D_inv @ b
        
        # アンスケーリング用の情報を保存
        scale_info = {'row_norms': row_norms}
        
        return scaled_A, scaled_b, scale_info
    
    def unscale(self, x, scale_info):
        """
        行スケーリングは解ベクトルに影響しない
        
        Args:
            x: 解ベクトル
            scale_info: スケーリング情報
            
        Returns:
            unscaled_x: 元のx（アンスケーリング不要）
        """
        # 行スケーリングは解を変更しないので、そのまま返す
        return x
    
    @property
    def name(self):
        return "RowScaling"
    
    @property
    def description(self):
        return "各行をそのノルムでスケーリング"