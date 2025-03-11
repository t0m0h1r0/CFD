from typing import Dict, Any, Tuple, Union, Optional
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
    
    def scale(self, A: Union[sp.spmatrix, cp.ndarray], b: cp.ndarray) -> Tuple[Union[sp.spmatrix, cp.ndarray], cp.ndarray, Dict[str, Any]]:
        """
        Aの各行をそのノルムでスケーリングし、bも対応して調整
        
        Args:
            A: システム行列（スパース）
            b: 右辺ベクトル
            
        Returns:
            tuple: (scaled_A, scaled_b, scale_info)
        """
        # 行ノルムを計算（効率化された実装）
        row_norms = cp.zeros(A.shape[0])
        
        # CSRフォーマットでより効率的に計算
        if hasattr(A, 'format') and A.format == 'csr':
            for i in range(A.shape[0]):
                start = A.indptr[i]
                end = A.indptr[i+1]
                # 直接データ配列から効率的にノルムを計算
                if self.norm_type == float('inf'):
                    row_norms[i] = cp.max(cp.abs(A.data[start:end])) if end > start else 0.0
                else:
                    row_norms[i] = cp.linalg.norm(A.data[start:end], ord=self.norm_type) if end > start else 0.0
        else:
            # 以前のコード（フォールバック）
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
    
    def unscale(self, x: cp.ndarray, scale_info: Dict[str, Any]) -> cp.ndarray:
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
    def name(self) -> str:
        return "RowScaling"
    
    @property
    def description(self) -> str:
        return "各行をそのノルムでスケーリング"