"""
行列の性質に基づいて最適なスケーリング手法を選択する適応的スケーリング

このスケーリング手法は行列の特性を分析し、最も適切なスケーリング戦略を
自動的に選択・組み合わせます。数値的安定性とパフォーマンスを最適化します。
"""

from typing import Dict, Any, Tuple
import logging
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
    
    def __init__(self, backend='numpy'):
        """
        初期化
        
        Args:
            backend: 計算バックエンド ('numpy', 'cupy', 'jax')
        """
        super().__init__(backend)
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
                nnz = self.array_utils.get_array_module().count_nonzero(A.toarray())
            else:
                nnz = self.array_utils.get_array_module().count_nonzero(A)
            properties['nnz'] = nnz
            properties['density'] = nnz / (m * n)
        
        # 対称性チェック（正方行列の場合）
        if properties['is_square']:
            if hasattr(A, 'toarray'):
                # 疎行列の場合、サンプルの要素をチェック
                sample_size = min(1000, m)
                xp = self.array_utils.get_array_module()
                indices = xp.random.choice(m, sample_size, replace=False)
                
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
                    xp = self.array_utils.get_array_module()
                    properties['is_symmetric'] = xp.allclose(A, A.T)
                except AttributeError:
                    properties['is_symmetric'] = False
        else:
            properties['is_symmetric'] = False
            
        # 行と列の変動を分析
        row_norms = self._compute_row_norms(A)
        col_norms = self._compute_column_norms(A)
        
        properties['row_norm_max'] = float(self.array_utils.max(row_norms))
        properties['row_norm_min'] = float(self.array_utils.min(row_norms))
        properties['row_norm_ratio'] = float(properties['row_norm_max'] / max(properties['row_norm_min'], 1e-15))
        
        properties['col_norm_max'] = float(self.array_utils.max(col_norms))
        properties['col_norm_min'] = float(self.array_utils.min(col_norms))
        properties['col_norm_ratio'] = float(properties['col_norm_max'] / max(properties['col_norm_min'], 1e-15))
        
        # 対角優位性（正方行列の場合）
        if properties['is_square']:
            try:
                diag = A.diagonal()
                diag_abs = self.array_utils.abs(diag)
                
                if hasattr(A, 'format') and A.format == 'csr':
                    # CSR行列の場合
                    row_sums = self.array_utils.zeros_like(diag)
                    for i in range(m):
                        start, end = A.indptr[i], A.indptr[i+1]
                        row_data = self.array_utils.abs(A.data[start:end])
                        row_sums[i] = self.array_utils.sum(row_data) - diag_abs[i]
                else:
                    # 一般的なケース
                    if hasattr(A, 'toarray'):
                        A_abs = self.array_utils.abs(A.toarray())
                    else:
                        A_abs = self.array_utils.abs(A)
                    row_sums = self.array_utils.sum(A_abs, axis=1) - diag_abs
                
                # 対角要素と非対角要素の比率
                # 数値の安定性のためのエラー処理
                diag_ratios = self.array_utils.where(row_sums < 1e-15, 0.0, diag_abs / row_sums)
                
                properties['diag_dominance'] = float(self.array_utils.mean(diag_ratios))
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
        # スケーリングオブジェクトを作成する際に自身のバックエンドを使用
        backend = self.array_utils.backend
        
        # 深刻な条件不良の行列には平衡化を使用
        if (properties.get('row_norm_ratio', 0) > 1e8 or 
            properties.get('col_norm_ratio', 0) > 1e8):
            return EquilibrationScaling(max_iterations=5, backend=backend), "EquilibrationScaling"
        
        # 対称行列には対称スケーリングが最適
        if properties.get('is_symmetric', False):
            return SymmetricScaling(backend=backend), "SymmetricScaling"
        
        # 対角優位行列には行スケーリングが有効
        if properties.get('diag_dominance', 0) > 0.8:
            return RowScaling(backend=backend), "RowScaling"
        
        # 行と列の変動を比較
        row_ratio = properties.get('row_norm_ratio', 1.0)
        col_ratio = properties.get('col_norm_ratio', 1.0)
        
        if row_ratio > 10 * col_ratio:
            return RowScaling(backend=backend), "RowScaling"
        elif col_ratio > 10 * row_ratio:
            return ColumnScaling(backend=backend), "ColumnScaling"
        
        # デフォルトでは平衡化を使用
        return EquilibrationScaling(backend=backend), "EquilibrationScaling"
    
    def _compute_row_norms(self, A):
        """行列Aの行ノルムを計算"""
        m = A.shape[0]
        row_norms = self.array_utils.zeros(m)
        
        if hasattr(A, 'format') and A.format == 'csr':
            for i in range(m):
                start, end = A.indptr[i], A.indptr[i+1]
                if end > start:
                    row_norms[i] = self.array_utils.linalg_norm(A.data[start:end])
        else:
            for i in range(m):
                row = A[i, :]
                if hasattr(row, 'toarray'):
                    row = row.toarray().flatten()
                row_norms[i] = self.array_utils.linalg_norm(row)
        
        return row_norms
    
    def _compute_column_norms(self, A):
        """行列Aの列ノルムを計算"""
        n = A.shape[1]
        col_norms = self.array_utils.zeros(n)
        
        if hasattr(A, 'format') and A.format == 'csc':
            for j in range(n):
                start, end = A.indptr[j], A.indptr[j+1]
                if end > start:
                    col_norms[j] = self.array_utils.linalg_norm(A.data[start:end])
        else:
            for j in range(n):
                col = A[:, j]
                if hasattr(col, 'toarray'):
                    col = col.toarray().flatten()
                col_norms[j] = self.array_utils.linalg_norm(col)
        
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
        backend = self.array_utils.backend
        
        # 適切なアンスケーリング方法に転送
        if selected_strategy == "RowScaling":
            return RowScaling(backend=backend).unscale(x, scale_info)
        elif selected_strategy == "ColumnScaling":
            return ColumnScaling(backend=backend).unscale(x, scale_info)
        elif selected_strategy == "SymmetricScaling":
            return SymmetricScaling(backend=backend).unscale(x, scale_info)
        elif selected_strategy == "EquilibrationScaling":
            return EquilibrationScaling(backend=backend).unscale(x, scale_info)
        
        # デフォルト：列スケーリングが適用されている可能性
        col_scale = scale_info.get('col_scale')
        if col_scale is not None:
            return x / col_scale
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
        backend = self.array_utils.backend
        
        # 適切なスケーリング方法に転送
        if selected_strategy == "RowScaling":
            return RowScaling(backend=backend).scale_b_only(b, scale_info)
        elif selected_strategy == "ColumnScaling":
            return ColumnScaling(backend=backend).scale_b_only(b, scale_info)
        elif selected_strategy == "SymmetricScaling":
            return SymmetricScaling(backend=backend).scale_b_only(b, scale_info)
        elif selected_strategy == "EquilibrationScaling":
            return EquilibrationScaling(backend=backend).scale_b_only(b, scale_info)
        
        # デフォルト：行スケーリング
        row_scale = scale_info.get('row_scale')
        if row_scale is not None:
            return b * row_scale
        return b
    
    @property
    def name(self) -> str:
        return "AdaptiveScaling"
    
    @property
    def description(self) -> str:
        return "最適な戦略を自動的に選択する適応的スケーリング"
