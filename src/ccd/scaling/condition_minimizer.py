"""
条件数を最小化するスケーリング手法
"""

from typing import Dict, Any, Tuple
import logging
from .base import BaseScaling
from .equilibration_scaling import EquilibrationScaling


class ConditionMinimizer(BaseScaling):
    """条件数を最小化するスケーリング手法"""
    
    def __init__(self, max_iterations=3, power_range=None, backend='numpy'):
        """
        パラメータを指定して初期化
        
        Args:
            max_iterations: 最小化プロセスの最大反復回数
            power_range: スケーリング係数のべき乗範囲（Noneの場合は自動）
            backend: 計算バックエンド ('numpy', 'cupy', 'jax')
        """
        super().__init__(backend)
        self.max_iterations = max_iterations
        self.power_range = power_range or [-2, -1, -0.5, 0.5, 1, 2]
        self._logger = logging.getLogger(__name__)
        
    def scale(self, A, b) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        条件数を改善するためにAをスケーリング
        
        Args:
            A: システム行列
            b: 右辺ベクトル
            
        Returns:
            tuple: (scaled_A, scaled_b, scale_info)
        """
        m, n = A.shape
        if m != n:
            # 非正方行列の場合は平衡化を使用
            eq_scaler = EquilibrationScaling(backend=self.array_utils.backend)
            return eq_scaler.scale(A, b)
        
        # まず平衡化スケーリングを基本として使用
        eq_scaler = EquilibrationScaling(max_iterations=2, backend=self.array_utils.backend)
        scaled_A, scaled_b, scale_info = eq_scaler.scale(A, b)
        
        # スケーリング係数を取得
        row_scale = scale_info.get('row_scale', self.array_utils.ones(m))
        col_scale = scale_info.get('col_scale', self.array_utils.ones(n))
        
        try:
            # 初期条件数を安全に推定
            initial_cond = self._estimate_condition_number(scaled_A)
            if initial_cond == float('inf') or self.array_utils.isnan(initial_cond):
                return scaled_A, scaled_b, scale_info
            
            best_cond = initial_cond
            best_A = scaled_A
            best_b = scaled_b
            best_scale_info = scale_info.copy()
            
            # 様々なべき乗でスケーリングを試す
            for power in self.power_range:
                try:
                    # 新しいスケーリング係数
                    new_row_scale = self.array_utils.power(row_scale, power)
                    new_col_scale = self.array_utils.power(col_scale, power)
                    
                    # スケーリング行列を作成
                    D_row = self.array_utils.diags(new_row_scale)
                    D_col = self.array_utils.diags(new_col_scale)
                    
                    # 試行用スケーリング適用
                    test_A = D_row @ A @ D_col
                    test_b = D_row @ b
                    
                    # 条件数をチェック
                    test_cond = self._estimate_condition_number(test_A)
                    
                    if test_cond < best_cond and test_cond > 0 and not self.array_utils.isnan(test_cond):
                        best_cond = test_cond
                        best_A = test_A
                        best_b = test_b
                        best_scale_info = {
                            'row_scale': new_row_scale,
                            'col_scale': new_col_scale
                        }
                except Exception:
                    continue
            
            return best_A, best_b, best_scale_info
            
        except Exception:
            # 何か問題が発生した場合は平衡化の結果を返す
            return scaled_A, scaled_b, scale_info
    
    def _estimate_condition_number(self, A):
        """効率的かつ安全な条件数推定"""
        try:
            n = A.shape[0]
            
            if n > 1000:
                # 大規模行列用：疎行列反復法
                try:
                    if self.array_utils.backend == 'cupy':
                        from cupyx.scipy.sparse.linalg import svds
                        # 最大特異値と最小特異値を推定
                        u, s_max, vh = svds(A, k=1, which='LM')
                        u, s_min, vh = svds(A, k=1, which='SM')
                        return float(s_max[0] / s_min[0])
                    elif self.array_utils.backend == 'numpy':
                        from scipy.sparse.linalg import svds
                        u, s_max, vh = svds(A, k=1, which='LM')
                        u, s_min, vh = svds(A, k=1, which='SM')
                        return float(s_max[0] / s_min[0])
                except:
                    pass
            
            # 小規模行列または代替方法
            if hasattr(A, 'toarray'):
                A_dense = A.toarray()
            else:
                A_dense = A
                
            # JAXの場合、別の方法が必要かもしれない
            if self.array_utils.backend == 'jax':
                # JAXのSVDを使う
                import jax.numpy as jnp
                try:
                    s = jnp.linalg.svd(A_dense, compute_uv=False)
                    return float(s[0] / s[-1])
                except:
                    pass
                    
            return float(self.array_utils.get_array_module().linalg.cond(A_dense))
        except:
            return float('inf')
    
    def unscale(self, x, scale_info: Dict[str, Any]):
        """解ベクトルをアンスケーリング"""
        col_scale = scale_info.get('col_scale')
        if col_scale is None:
            return x
        return x / col_scale
    
    def scale_b_only(self, b, scale_info: Dict[str, Any]):
        """右辺ベクトルbのみをスケーリング"""
        row_scale = scale_info.get('row_scale')
        if row_scale is not None:
            return b * row_scale
        return b
    
    @property
    def name(self) -> str:
        return "ConditionMinimizer"
    
    @property
    def description(self) -> str:
        return "行列の条件数を最小化することを目的としたスケーリング"
