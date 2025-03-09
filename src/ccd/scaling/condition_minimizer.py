import cupy as cp
import cupyx.scipy.sparse as sp
from .base import BaseScaling
from .equilibration_scaling import EquilibrationScaling

class ConditionMinimizer(BaseScaling):
    """条件数を最小化するスケーリング手法"""
    
    def __init__(self, max_iterations=5):
        """
        パラメータを指定して初期化
        
        Args:
            max_iterations: 最小化プロセスの最大反復回数
        """
        self.max_iterations = max_iterations
    
    def scale(self, A, b):
        """
        条件数を改善するためにAをスケーリング
        
        Args:
            A: システム行列（スパース）
            b: 右辺ベクトル
            
        Returns:
            tuple: (scaled_A, scaled_b, scale_info)
        """
        m, n = A.shape
        if m != n:
            # 非正方行列の場合は平衡化を使用
            return EquilibrationScaling().scale(A, b)
        
        # 正方行列の場合はより高度な手法を使用
        # まず平衡化を基本として使用
        scaled_A, scaled_b, scale_info = EquilibrationScaling().scale(A, b)
        
        # 追加のスケーリング反復で改善を試みる
        row_scale = scale_info.get('row_scale', cp.ones(m))
        col_scale = scale_info.get('col_scale', cp.ones(n))
        
        # 条件数推定値を最小化する追加反復
        try:
            # 条件数推定のために密行列に変換
            # (大きな行列の場合はコストがかかる可能性がある)
            A_dense = scaled_A.toarray()
            
            # 初期条件数を推定
            initial_cond = cp.linalg.cond(A_dense)
            best_cond = initial_cond
            best_A = scaled_A
            best_b = scaled_b
            
            # 2のべき乗でスケーリングを試す
            for i in range(1, self.max_iterations+1):
                # 新しいスケーリング行列を作成
                power_scale = 2**(-i)
                D_row = sp.diags(cp.power(row_scale, power_scale))
                D_col = sp.diags(cp.power(col_scale, power_scale))
                
                # スケーリングを適用
                test_A = D_row @ A @ D_col
                test_b = D_row @ b
                
                # 条件数をチェック
                test_cond = cp.linalg.cond(test_A.toarray())
                
                if test_cond < best_cond:
                    best_cond = test_cond
                    best_A = test_A
                    best_b = test_b
                    # スケーリング情報を更新
                    scale_info['row_scale'] = cp.power(row_scale, power_scale) * row_scale
                    scale_info['col_scale'] = cp.power(col_scale, power_scale) * col_scale
            
            print(f"条件数の改善: {initial_cond:.2e} -> {best_cond:.2e}")
            return best_A, best_b, scale_info
            
        except Exception as e:
            # 何か問題が発生した場合は平衡化の結果を返す
            print(f"条件数最小化でエラーが発生しました: {e}")
            return scaled_A, scaled_b, scale_info
    
    def unscale(self, x, scale_info):
        """
        解ベクトルをアンスケーリング
        
        Args:
            x: 解ベクトル
            scale_info: スケーリング情報
            
        Returns:
            unscaled_x: アンスケーリングされた解ベクトル
        """
        col_scale = scale_info.get('col_scale', cp.ones(len(x)))
        unscaled_x = x / col_scale
        return unscaled_x
    
    @property
    def name(self):
        return "ConditionMinimizer"
    
    @property
    def description(self):
        return "行列の条件数を最小化することを目的としたスケーリング"