"""
Ruhe対角スケーリングの実装

このスケーリング手法は、Axel Ruheの反復的手法を実装して対角スケーリングにより
行列の条件数を改善します。この方法は行と列のノルムがほぼ等しくなるように
スケーリング係数を反復的に計算します。
"""

from typing import Dict, Any, Tuple
import numpy as np
import scipy.sparse as sp
from .base import BaseScaling


class RuheScaling(BaseScaling):
    """
    Ruhe対角スケーリング手法
    
    この手法は行と列のノルムがほぼ等しくなるよう行列を反復的に平衡化し、
    多くの問題タイプの条件数を大幅に改善します。
    
    参考文献:
        Ruhe, A. (1980). "Perturbation bounds for means of eigenvalues and invariant subspaces."
    """
    
    def __init__(self, max_iterations=10, tolerance=1e-6, norm_type=2):
        """
        Ruheスケーリングアルゴリズムを初期化
        
        Args:
            max_iterations: アルゴリズムの最大反復回数
            tolerance: 行/列ノルム比の収束許容誤差
            norm_type: 使用するノルムの種類 (2: ユークリッド、float('inf'): 最大)
        """
        super().__init__()
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.norm_type = norm_type
        
    def scale(self, A, b) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Ruhe対角スケーリングを行列Aと右辺ベクトルbに適用
        
        Args:
            A: スケーリングするシステム行列
            b: 右辺ベクトル
            
        Returns:
            tuple: (scaled_A, scaled_b, scale_info)
        """
        m, n = A.shape
        
        # スケーリングベクトルを1で初期化
        d_row = np.ones(m, dtype='float64')
        d_col = np.ones(n, dtype='float64')
        
        # 作業用にAのコピーを作成
        scaled_A = A.copy() if hasattr(A, 'copy') else A
        
        # Aが疎行列形式かどうかを判定
        is_csr = hasattr(scaled_A, 'format') and scaled_A.format == 'csr'
        is_csc = hasattr(scaled_A, 'format') and scaled_A.format == 'csc'
        
        # 反復スケーリングアルゴリズム
        for iteration in range(self.max_iterations):
            # 行と列のノルムを計算
            row_norms = self._compute_row_norms(scaled_A, is_csr)
            col_norms = self._compute_column_norms(scaled_A, is_csc)
            
            # 収束チェック
            row_norm_avg = np.mean(row_norms)
            col_norm_avg = np.mean(col_norms)
            
            # ノルムがほぼ等しい場合は終了
            if abs(row_norm_avg - col_norm_avg) < self.tolerance:
                break
                
            # スケーリング係数を計算
            alpha_row = np.sqrt(col_norm_avg / np.maximum(row_norm_avg, 1e-15))
            alpha_col = np.sqrt(row_norm_avg / np.maximum(col_norm_avg, 1e-15))
            
            # スケーリングベクトルを更新
            d_row_update = np.power(row_norms, -0.5) * alpha_row
            d_col_update = np.power(col_norms, -0.5) * alpha_col
            
            # 数値的安定性の保護
            d_row_update = np.where(np.isfinite(d_row_update), d_row_update, 1.0)
            d_col_update = np.where(np.isfinite(d_col_update), d_col_update, 1.0)
            
            # 累積スケーリング係数を更新
            d_row *= d_row_update
            d_col *= d_col_update
            
            # 対角スケーリング行列を構築
            D_row = sp.diags(d_row_update)
            D_col = sp.diags(d_col_update)
            
            # 行列にスケーリングを適用
            scaled_A = D_row @ scaled_A @ D_col
        
        # 右辺ベクトルをスケーリング
        scaled_b = b * d_row
        
        # スケーリングされた行列、スケーリングされた右辺、スケーリング情報を返す
        return scaled_A, scaled_b, {'row_scale': d_row, 'col_scale': d_col}
        
    def _compute_row_norms(self, A, is_csr=False):
        """行列Aの行ノルムを効率的に計算"""
        m = A.shape[0]
        row_norms = np.zeros(m)
        
        if is_csr:
            # CSRフォーマットでは行へのアクセスに効率的なindptrを使用
            for i in range(m):
                start, end = A.indptr[i], A.indptr[i+1]
                if end > start:
                    row_data = A.data[start:end]
                    if self.norm_type == float('inf'):
                        row_norms[i] = np.max(np.abs(row_data))
                    else:
                        row_norms[i] = np.linalg.norm(row_data, ord=self.norm_type)
        else:
            # 一般的なケース
            for i in range(m):
                row = A[i, :]
                if hasattr(row, 'toarray'):
                    row = row.toarray().flatten()
                row_norms[i] = np.linalg.norm(row, ord=self.norm_type)
        
        return row_norms
    
    def _compute_column_norms(self, A, is_csc=False):
        """行列Aの列ノルムを効率的に計算"""
        n = A.shape[1]
        col_norms = np.zeros(n)
        
        if is_csc:
            # CSCフォーマットでは列へのアクセスに効率的なindptrを使用
            for j in range(n):
                start, end = A.indptr[j], A.indptr[j+1]
                if end > start:
                    col_data = A.data[start:end]
                    if self.norm_type == float('inf'):
                        col_norms[j] = np.max(np.abs(col_data))
                    else:
                        col_norms[j] = np.linalg.norm(col_data, ord=self.norm_type)
        else:
            # 一般的なケース
            for j in range(n):
                col = A[:, j]
                if hasattr(col, 'toarray'):
                    col = col.toarray().flatten()
                col_norms[j] = np.linalg.norm(col, ord=self.norm_type)
        
        return col_norms
    
    def unscale(self, x, scale_info: Dict[str, Any]):
        """
        解ベクトルをアンスケーリング
        
        Args:
            x: アンスケーリングする解ベクトル
            scale_info: scaleメソッドからのスケーリング情報
            
        Returns:
            アンスケーリングされた解ベクトル
        """
        col_scale = scale_info.get('col_scale')
        if col_scale is None:
            return x
        return x / col_scale
    
    def scale_b_only(self, b, scale_info: Dict[str, Any]):
        """
        右辺ベクトルのみをスケーリング
        
        Args:
            b: スケーリングする右辺ベクトル
            scale_info: scaleメソッドからのスケーリング情報
            
        Returns:
            スケーリングされた右辺ベクトル
        """
        row_scale = scale_info.get('row_scale')
        if row_scale is None:
            return b
        return b * row_scale
    
    @property
    def name(self) -> str:
        return "RuheScaling"
    
    @property
    def description(self) -> str:
        return "行列の条件数を改善するRuheの反復的対角スケーリング"