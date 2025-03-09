import cupy as cp
import cupyx.scipy.sparse as sp
from .base import BaseScaling

class EquilibrationScaling(BaseScaling):
    """行と列のスケーリングを組み合わせた平衡化スケーリング"""
    
    def __init__(self, max_iterations=10, tolerance=1e-8):
        """
        反復パラメータを指定して初期化
        
        Args:
            max_iterations: 平衡化の最大反復回数
            tolerance: 収束許容誤差
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def scale(self, A, b):
        """
        反復的平衡化スケーリングを適用
        
        Args:
            A: システム行列（スパース）
            b: 右辺ベクトル
            
        Returns:
            tuple: (scaled_A, scaled_b, scale_info)
        """
        m, n = A.shape
        scaled_A = A.copy()
        
        # スケーリングベクトルを初期化
        row_scale = cp.ones(m)
        col_scale = cp.ones(n)
        
        # 反復処理
        for _ in range(self.max_iterations):
            # 行と列のノルムを取得
            row_norms = cp.zeros(m)
            for i in range(m):
                row = scaled_A[i, :].toarray().flatten()
                row_norms[i] = cp.linalg.norm(row, ord=float('inf'))
            
            col_norms = cp.zeros(n)
            for j in range(n):
                col = scaled_A[:, j].toarray().flatten()
                col_norms[j] = cp.linalg.norm(col, ord=float('inf'))
            
            # 収束確認
            if (cp.abs(row_norms - 1.0) < self.tolerance).all() and (cp.abs(col_norms - 1.0) < self.tolerance).all():
                break
            
            # スケーリングベクトルを更新
            row_scale_update = cp.where(row_norms < 1e-15, 1.0, 1.0 / cp.sqrt(row_norms))
            col_scale_update = cp.where(col_norms < 1e-15, 1.0, 1.0 / cp.sqrt(col_norms))
            
            row_scale *= row_scale_update
            col_scale *= col_scale_update
            
            # スケーリングを適用
            DR = sp.diags(row_scale_update)
            DC = sp.diags(col_scale_update)
            scaled_A = DR @ scaled_A @ DC
        
        # bをスケーリング
        scaled_b = b * row_scale
        
        # アンスケーリング用の情報を保存
        scale_info = {'row_scale': row_scale, 'col_scale': col_scale}
        
        return scaled_A, scaled_b, scale_info
    
    def unscale(self, x, scale_info):
        """
        解ベクトルをアンスケーリング
        
        Args:
            x: 解ベクトル
            scale_info: row_scaleとcol_scaleを含むスケーリング情報
            
        Returns:
            unscaled_x: アンスケーリングされた解ベクトル
        """
        col_scale = scale_info['col_scale']
        unscaled_x = x / col_scale
        return unscaled_x
    
    @property
    def name(self):
        return "EquilibrationScaling"
    
    @property
    def description(self):
        return "行と列のノルムをバランスさせる反復的平衡化スケーリング"