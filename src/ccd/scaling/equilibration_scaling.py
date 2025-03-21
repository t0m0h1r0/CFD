"""
行と列のスケーリングを組み合わせた平衡化スケーリング実装（汎用版）
"""

from typing import Dict, Any, Tuple
from .base import BaseScaling


class EquilibrationScaling(BaseScaling):
    """行と列のスケーリングを組み合わせた平衡化スケーリング: A → D_r A D_c, b → D_r b"""
    
    def __init__(self, max_iterations=5, tolerance=1e-8):
        """
        反復パラメータを指定して初期化
        
        Args:
            max_iterations: 平衡化の最大反復回数
            tolerance: 収束許容誤差
        """
        super().__init__()
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def scale(self, A, b) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        反復的平衡化スケーリングを適用（汎用版）
        
        Args:
            A: システム行列 (NumPyまたはCuPy)
            b: 右辺ベクトル (NumPyまたはCuPy)
            
        Returns:
            tuple: (scaled_A, scaled_b, scale_info)
        """
        # 配列モジュールの取得（NumPyまたはCuPy）
        xp = self._get_array_module(A)
        m, n = A.shape
        
        # スケーリングベクトルを初期化
        row_scale = xp.ones(m, dtype=A.dtype if hasattr(A, 'dtype') else None)
        col_scale = xp.ones(n, dtype=A.dtype if hasattr(A, 'dtype') else None)
        
        # 元の行列をコピー
        scaled_A = A.copy() if hasattr(A, 'copy') else A.copy()
        
        # 最大許容スケーリング係数
        max_scale_factor = 1e3
        
        # 反復処理
        for iter_idx in range(self.max_iterations):
            # 行および列のノルムを計算
            row_norms = self._compute_norms(scaled_A, axis=1)
            col_norms = self._compute_norms(scaled_A, axis=0)
            
            # 収束確認
            if (xp.abs(row_norms - 1.0) < self.tolerance).all() and \
               (xp.abs(col_norms - 1.0) < self.tolerance).all():
                break
            
            # スケーリング係数を更新
            row_scale_update = 1.0 / xp.sqrt(self._maximum(row_norms, 1e-10))
            col_scale_update = 1.0 / xp.sqrt(self._maximum(col_norms, 1e-10))
            
            # 係数を制限
            if hasattr(xp, 'clip'):
                row_scale_update = xp.clip(row_scale_update, 1.0/max_scale_factor, max_scale_factor)
                col_scale_update = xp.clip(col_scale_update, 1.0/max_scale_factor, max_scale_factor)
            else:
                # フォールバック
                row_scale_update = xp.minimum(xp.maximum(row_scale_update, 1.0/max_scale_factor), max_scale_factor)
                col_scale_update = xp.minimum(xp.maximum(col_scale_update, 1.0/max_scale_factor), max_scale_factor)
            
            # 合計スケーリング係数を更新
            row_scale *= row_scale_update
            col_scale *= col_scale_update
            
            # 合計係数も制限
            if hasattr(xp, 'clip'):
                row_scale = xp.clip(row_scale, 1.0/max_scale_factor, max_scale_factor)
                col_scale = xp.clip(col_scale, 1.0/max_scale_factor, max_scale_factor)
            else:
                row_scale = xp.minimum(xp.maximum(row_scale, 1.0/max_scale_factor), max_scale_factor)
                col_scale = xp.minimum(xp.maximum(col_scale, 1.0/max_scale_factor), max_scale_factor)
            
            # 行列のスケーリング
            D_row = self._diags(row_scale_update)
            D_col = self._diags(col_scale_update)
            scaled_A = D_row @ scaled_A @ D_col
        
        # 右辺ベクトルのスケーリング
        scaled_b = b * row_scale
        
        return scaled_A, scaled_b, {'row_scale': row_scale, 'col_scale': col_scale}
    
    def _compute_norms(self, A, axis=1):
        """
        行列の各行または列のノルム（無限大ノルム）を計算
        
        Args:
            A: 行列 (NumPyまたはCuPy)
            axis: 0=列方向、1=行方向
            
        Returns:
            norms: 各行または列のノルム
        """
        xp = self._get_array_module(A)
        is_sparse = self._is_sparse(A)
        m, n = A.shape
        
        if axis == 1:  # 行ノルム
            norms = self._zeros(m, dtype=A.dtype if hasattr(A, 'dtype') else None, array_ref=A)
            
            if is_sparse and hasattr(A, 'format'):
                if A.format == 'csr':
                    # CSR形式は行方向のアクセスが効率的
                    for i in range(m):
                        start, end = A.indptr[i], A.indptr[i+1]
                        if end > start:
                            norms[i] = xp.max(xp.abs(A.data[start:end]))
                else:
                    # CSC形式などは通常の方法で
                    for i in range(m):
                        row = A[i, :]
                        if hasattr(row, 'toarray'):
                            row = row.toarray().flatten()
                        if row.size > 0:
                            norms[i] = xp.max(xp.abs(row))
            else:
                # 密行列の場合
                if hasattr(xp, 'max') and hasattr(xp, 'abs'):
                    # 効率的な方法があれば使用
                    norms = xp.max(xp.abs(A), axis=1)
                else:
                    # フォールバック
                    for i in range(m):
                        norms[i] = xp.max(xp.abs(A[i, :]))
        
        else:  # 列ノルム
            norms = self._zeros(n, dtype=A.dtype if hasattr(A, 'dtype') else None, array_ref=A)
            
            if is_sparse and hasattr(A, 'format'):
                if A.format == 'csc':
                    # CSC形式は列方向のアクセスが効率的
                    for j in range(n):
                        start, end = A.indptr[j], A.indptr[j+1]
                        if end > start:
                            norms[j] = xp.max(xp.abs(A.data[start:end]))
                else:
                    # CSR形式などは通常の方法で
                    for j in range(n):
                        col = A[:, j]
                        if hasattr(col, 'toarray'):
                            col = col.toarray().flatten()
                        if col.size > 0:
                            norms[j] = xp.max(xp.abs(col))
            else:
                # 密行列の場合
                if hasattr(xp, 'max') and hasattr(xp, 'abs'):
                    # 効率的な方法があれば使用
                    norms = xp.max(xp.abs(A), axis=0)
                else:
                    # フォールバック
                    for j in range(n):
                        norms[j] = xp.max(xp.abs(A[:, j]))
        
        return norms
    
    def unscale(self, x, scale_info: Dict[str, Any]):
        """解ベクトルをアンスケーリング"""
        col_scale = scale_info.get('col_scale')
        if col_scale is not None:
            return x / col_scale
        return x
    
    def scale_b_only(self, b, scale_info: Dict[str, Any]):
        """右辺ベクトルbのみをスケーリング"""
        row_scale = scale_info.get('row_scale')
        if row_scale is not None:
            return b * row_scale
        return b
    
    @property
    def name(self) -> str:
        return "EquilibrationScaling"
    
    @property
    def description(self) -> str:
        return "行と列のノルムをバランスさせる反復的平衡化スケーリング"