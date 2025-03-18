"""
行列の性質に基づいて最適なスケーリング手法を選択する適応的スケーリング

このスケーリング手法は行列の特性を分析し、最も適切なスケーリング戦略を
自動的に選択・組み合わせます。数値的安定性とパフォーマンスを最適化します。
"""

from typing import Dict, Any, Tuple
import logging
import numpy as np
import scipy.sparse as sp
from .base import BaseScaling
from .row_scaling import RowScaling
from .column_scaling import ColumnScaling
from .symmetric_scaling import SymmetricScaling
from .equilibration_scaling import EquilibrationScaling


class AdaptiveScaling(BaseScaling):
    """
    適応的スケーリング手法
    
    行列の特性を分析し、行スケーリング、列スケーリング、対称スケーリング、
    または平衡化スケーリングのいずれが最も効果的かを判断して適用します。
    """
    
    def __init__(self):
        """初期化"""
        super().__init__()
        self._logger = logging.getLogger(__name__)
        self.selected_strategy = None
        self.strategy_info = {}
        
    def scale(self, A, b) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        行列Aを分析し、最適なスケーリング戦略を適用
        
        Args:
            A: システム行列
            b: 右辺ベクトル
            
        Returns:
            tuple: (scaled_A, scaled_b, scale_info)
        """
        # 行列の特性を分析
        properties = self._analyze_matrix(A)
        
        # 分析結果に基づいてスケーリング戦略を選択
        strategy, strategy_name = self._select_strategy(properties)
        
        # 選択された戦略を適用
        scaled_A, scaled_b, scale_info = strategy.scale(A, b)
        
        # 選択された戦略に関する情報を保存
        self.selected_strategy = strategy_name
        self.strategy_info = properties
        
        # scale_infoにメタデータを追加
        scale_info['selected_strategy'] = strategy_name
        scale_info['matrix_properties'] = properties
        
        return scaled_A, scaled_b, scale_info
    
    def _analyze_matrix(self, A) -> Dict[str, Any]:
        """
        スケーリング戦略の選択を導くための行列特性分析
        
        Args:
            A: 分析する行列
            
        Returns:
            行列特性の辞書
        """
        properties = {}
        m, n = A.shape
        
        # 基本的な特性
        properties['is_square'] = m == n
        properties['size'] = max(m, n)
        properties['shape'] = (m, n)
        
        # 疎性分析
        if hasattr(A, 'nnz'):
            properties['nnz'] = A.nnz
            properties['density'] = A.nnz / (m * n)
        else:
            if hasattr(A, 'toarray'):
                nnz = np.count_nonzero(A.toarray())
            else:
                nnz = np.count_nonzero(A)
            properties['nnz'] = nnz
            properties['density'] = nnz / (m * n)
        
        # 対称性チェック（正方行列の場合）
        if properties['is_square']:
            if hasattr(A, 'toarray'):
                # 疎行列の場合、サンプルの要素をチェック
                sample_size = min(1000, m)
                indices = np.random.choice(m, sample_size, replace=False)
                
                symmetry_violations = 0
                for i in indices:
                    for j in indices:
                        if i < j:  # 上三角部分のみチェック
                            if A[i, j] != A[j, i]:
                                symmetry_violations += 1
                
                properties['is_symmetric'] = symmetry_violations < sample_size * 0.05
            else:
                # 密行列の場合
                try:
                    properties['is_symmetric'] = np.allclose(A, A.T)
                except AttributeError:
                    properties['is_symmetric'] = False
        else:
            properties['is_symmetric'] = False
            
        # 行と列の変動を分析
        row_norms = self._compute_row_norms(A)
        col_norms = self._compute_column_norms(A)
        
        properties['row_norm_max'] = float(np.max(row_norms))
        properties['row_norm_min'] = float(np.min(row_norms))
        properties['row_norm_ratio'] = float(properties['row_norm_max'] / max(properties['row_norm_min'], 1e-15))
        
        properties['col_norm_max'] = float(np.max(col_norms))
        properties['col_norm_min'] = float(np.min(col_norms))
        properties['col_norm_ratio'] = float(properties['col_norm_max'] / max(properties['col_norm_min'], 1e-15))
        
        # 対角優位性（正方行列の場合）
        if properties['is_square']:
            try:
                diag = A.diagonal()
                diag_abs = np.abs(diag)
                
                if hasattr(A, 'format') and A.format == 'csr':
                    # CSR行列の場合
                    row_sums = np.zeros_like(diag)
                    for i in range(m):
                        start, end = A.indptr[i], A.indptr[i+1]
                        row_data = np.abs(A.data[start:end])
                        row_sums[i] = np.sum(row_data) - diag_abs[i]
                else:
                    # 一般的なケース
                    if hasattr(A, 'toarray'):
                        A_abs = np.abs(A.toarray())
                    else:
                        A_abs = np.abs(A)
                    row_sums = np.sum(A_abs, axis=1) - diag_abs
                
                # 対角要素と非対角要素の比率
                # 数値の安定性のためのエラー処理
                diag_ratios = np.where(row_sums < 1e-15, 0.0, diag_abs / row_sums)
                
                properties['diag_dominance'] = float(np.mean(diag_ratios))
            except Exception as e:
                self._logger.warning(f"対角優位性分析中にエラー: {e}")
                properties['diag_dominance'] = 0.0
        
        return properties
    
    def _select_strategy(self, properties) -> Tuple[BaseScaling, str]:
        """
        行列特性に基づいて最適なスケーリング戦略を選択
        
        Args:
            properties: 行列特性の辞書
            
        Returns:
            tuple: (scaling_strategy, strategy_name)
        """
        # 深刻な条件不良の行列には平衡化を使用
        if (properties.get('row_norm_ratio', 0) > 1e8 or 
            properties.get('col_norm_ratio', 0) > 1e8):
            return EquilibrationScaling(max_iterations=5), "EquilibrationScaling"
        
        # 対称行列には対称スケーリングが最適
        if properties.get('is_symmetric', False):
            return SymmetricScaling(), "SymmetricScaling"
        
        # 対角優位行列には行スケーリングが有効
        if properties.get('diag_dominance', 0) > 0.8:
            return RowScaling(), "RowScaling"
        
        # 行と列の変動を比較
        row_ratio = properties.get('row_norm_ratio', 1.0)
        col_ratio = properties.get('col_norm_ratio', 1.0)
        
        if row_ratio > 10 * col_ratio:
            return RowScaling(), "RowScaling"
        elif col_ratio > 10 * row_ratio:
            return ColumnScaling(), "ColumnScaling"
        
        # デフォルトでは平衡化を使用
        return EquilibrationScaling(), "EquilibrationScaling"
    
    def _compute_row_norms(self, A):
        """行列Aの行ノルムを計算"""
        m = A.shape[0]
        row_norms = np.zeros(m)
        
        if hasattr(A, 'format') and A.format == 'csr':
            for i in range(m):
                start, end = A.indptr[i], A.indptr[i+1]
                if end > start:
                    row_norms[i] = np.linalg.norm(A.data[start:end])
        else:
            # その他の形式
            for i in range(m):
                row = A[i, :]
                if hasattr(row, 'toarray'):
                    row = row.toarray().flatten()
                row_norms[i] = np.linalg.norm(row)
        
        return row_norms
    
    def _compute_column_norms(self, A):
        """行列Aの列ノルムを計算"""
        n = A.shape[1]
        col_norms = np.zeros(n)
        
        if hasattr(A, 'format') and A.format == 'csc':
            for j in range(n):
                start, end = A.indptr[j], A.indptr[j+1]
                if end > start:
                    col_norms[j] = np.linalg.norm(A.data[start:end])
        else:
            # その他の形式
            for j in range(n):
                col = A[:, j]
                if hasattr(col, 'toarray'):
                    col = col.toarray().flatten()
                col_norms[j] = np.linalg.norm(col)
        
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
        # 選択された戦略を抽出
        selected_strategy = scale_info.get('selected_strategy')
        
        # 適切なアンスケーラーを作成し、そのunscaleメソッドを呼び出す
        if selected_strategy == "RowScaling":
            return RowScaling().unscale(x, scale_info)
        elif selected_strategy == "ColumnScaling":
            return ColumnScaling().unscale(x, scale_info)
        elif selected_strategy == "SymmetricScaling":
            return SymmetricScaling().unscale(x, scale_info)
        elif selected_strategy == "EquilibrationScaling":
            return EquilibrationScaling().unscale(x, scale_info)
        
        # 選択された戦略が不明の場合のフォールバック処理
        # 列スケーリングの場合
        col_scale = scale_info.get('col_scale')
        if col_scale is not None:
            return x / col_scale
        
        # 対称スケーリングの場合
        D_sqrt_inv = scale_info.get('D_sqrt_inv')
        if D_sqrt_inv is not None:
            return x * D_sqrt_inv
            
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
        # 選択された戦略を抽出
        selected_strategy = scale_info.get('selected_strategy')
        
        # 適切なスケーラーを作成し、そのscale_b_onlyメソッドを呼び出す
        if selected_strategy == "RowScaling":
            return RowScaling().scale_b_only(b, scale_info)
        elif selected_strategy == "ColumnScaling":
            return ColumnScaling().scale_b_only(b, scale_info)
        elif selected_strategy == "SymmetricScaling":
            return SymmetricScaling().scale_b_only(b, scale_info)
        elif selected_strategy == "EquilibrationScaling":
            return EquilibrationScaling().scale_b_only(b, scale_info)
        
        # 選択された戦略が不明の場合のフォールバック処理
        # 行スケーリングの場合
        row_scale = scale_info.get('row_scale')
        if row_scale is not None:
            return b * row_scale
        
        # 対称スケーリングの場合
        D_sqrt_inv = scale_info.get('D_sqrt_inv')
        if D_sqrt_inv is not None:
            return b * D_sqrt_inv
            
        return b
    
    @property
    def name(self) -> str:
        return "AdaptiveScaling"
    
    @property
    def description(self) -> str:
        return "最適な戦略を自動的に選択する適応的スケーリング"