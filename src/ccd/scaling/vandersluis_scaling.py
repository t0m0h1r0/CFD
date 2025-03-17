"""
Van der Sluis対角スケーリングの実装

このスケーリング手法はVan der Sluisの最適手法を実装し、行スケーリングにより
行列の条件数を改善します。特に条件数の悪い行列に効果的で、多くの問題クラスに
対して理論的に良好な結果を生み出すことが証明されています。
"""

from typing import Dict, Any, Tuple
from .base import BaseScaling


class VanDerSluisScaling(BaseScaling):
    """
    Van der Sluis対角スケーリング
    
    このスケーリング手法は行ノルムの幾何平均を使用して最適なスケーリング係数を
    計算します。これは多くの行列クラスにおいて（ある意味で）条件数を最小化する
    ことが証明されています。
    
    参考文献:
        Van der Sluis, A. (1969). "Condition numbers and equilibration of matrices."
        Numerische Mathematik, 14(1), 14-23.
    """
    
    def __init__(self, norm_type=2, backend='numpy'):
        """
        Van der Sluisスケーリングを初期化
        
        Args:
            norm_type: 使用するノルムの種類 (2: ユークリッド、float('inf'): 最大)
            backend: 計算バックエンド ('numpy', 'cupy', 'jax')
        """
        super().__init__(backend)
        self.norm_type = norm_type
    
    def scale(self, A, b) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Van der Sluisの最適行スケーリングを行列Aと右辺ベクトルbに適用
        
        Args:
            A: スケーリングするシステム行列
            b: 右辺ベクトル
            
        Returns:
            tuple: (scaled_A, scaled_b, scale_info)
        """
        m, n = A.shape
        
        # 効率的に行ノルムを計算
        is_csr = hasattr(A, 'format') and A.format == 'csr'
        row_norms = self._compute_row_norms(A, is_csr)
        
        # 安定性のためにゼロおよび小さい値を1に置き換え
        row_norms = self.array_utils.where(row_norms < 1e-15, 1.0, row_norms)
        
        # 最適スケーリング係数を計算
        # van der Sluis法では、各行は1/row_normでスケーリングされる
        row_scale = 1.0 / row_norms
        
        # 極端に条件数の悪い行列の場合、ダンピングを適用
        # これによって極端な場合の破滅的な桁落ちを避ける
        max_scale = self.array_utils.max(row_scale)
        min_scale = self.array_utils.min(row_scale)
        if max_scale / min_scale > 1e10:
            row_scale = self.array_utils.sqrt(row_scale)  # スケーリングを緩和
        
        # 対角スケーリング行列を構築
        D_row = self.array_utils.diags(row_scale)
        
        # 行列と右辺ベクトルにスケーリングを適用
        scaled_A = D_row @ A
        scaled_b = D_row @ b
        
        return scaled_A, scaled_b, {'row_scale': row_scale}
    
    def _compute_row_norms(self, A, is_csr=False):
        """行列Aの行ノルムを効率的に計算"""
        m = A.shape[0]
        row_norms = self.array_utils.zeros(m)
        
        if is_csr:
            # CSRフォーマットでは行へのアクセスに効率的なindptrを使用
            for i in range(m):
                start, end = A.indptr[i], A.indptr[i+1]
                if end > start:
                    row_data = A.data[start:end]
                    if self.norm_type == float('inf'):
                        row_norms[i] = self.array_utils.max(self.array_utils.abs(row_data))
                    else:
                        row_norms[i] = self.array_utils.linalg_norm(row_data, ord=self.norm_type)
        else:
            # 一般的なケース
            for i in range(m):
                row = A[i, :]
                if hasattr(row, 'toarray'):
                    row = row.toarray().flatten()
                row_norms[i] = self.array_utils.linalg_norm(row, ord=self.norm_type)
        
        return row_norms
    
    def unscale(self, x, scale_info: Dict[str, Any]):
        """
        解ベクトルをアンスケーリング
        
        Args:
            x: アンスケーリングする解ベクトル
            scale_info: scaleメソッドからのスケーリング情報
            
        Returns:
            アンスケーリングされた解ベクトル
        """
        # Van der Sluisスケーリングは行のみに影響し、解ベクトルには影響しない
        return x
    
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
        return "VanDerSluisScaling"
    
    @property
    def description(self) -> str:
        return "条件数を最小化するVan der Sluisの最適行スケーリング"
