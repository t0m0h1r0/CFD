"""
代数的マルチグリッド前処理

このモジュールは、マルチグリッド手法を用いた
高効率な前処理手法を提供します。
特に楕円型PDEに対して優れた性能を発揮します。
"""

import numpy as np
import scipy.sparse as sp
from .base import BasePreconditioner
import pyamg

class AMGPreconditioner(BasePreconditioner):
    """代数的マルチグリッド前処理"""
    
    def __init__(self, max_levels=10, cycle_type='V', strength_threshold=0.0):
        """
        初期化
        
        Args:
            max_levels: マルチグリッドの最大レベル数
            cycle_type: サイクルタイプ ('V' or 'W')
            strength_threshold: 粗視化のための閾値
        """
        super().__init__()
        self.max_levels = max_levels
        self.cycle_type = cycle_type
        self.strength_threshold = strength_threshold
        self.ml = None
        self.solver = None
    
    def setup(self, A):
        """
        AMG前処理の設定
        
        Args:
            A: システム行列
            
        Returns:
            self: メソッドチェーン用
        """
        # PyAMGが使えない場合のフォールバック用に単純な対角前処理を準備
        try:
            # 行列をCSRフォーマットに変換（必須）
            if hasattr(A, 'tocsr'):
                A_csr = A.tocsr()
            elif hasattr(A, 'format') and A.format == 'csr':
                A_csr = A
            else:
                # 密行列を疎行列に変換
                A_csr = sp.csr_matrix(A)
            
            # 行列に対してチェックと修正
            self._ensure_valid_matrix(A_csr)
            
            try:
                from pyamg import smoothed_aggregation_solver
                
                # まず最もシンプルな呼び出しを試す（パラメータなし）
                try:
                    print("PyAMG: 基本設定を試みています...")
                    self.ml = smoothed_aggregation_solver(A_csr)
                except Exception as e1:
                    print(f"基本設定失敗: {e1}")
                    
                    # 次に、単純なstrength値のみで試す
                    try:
                        print("PyAMG: strength値のみで試行中...")
                        self.ml = smoothed_aggregation_solver(A_csr, strength=0.0)
                    except Exception as e2:
                        print(f"strength値のみの設定失敗: {e2}")
                        
                        # 最後に、カスタム関数を試す
                        try:
                            print("PyAMG: カスタム設定を試行中...")
                            self._setup_with_custom_params(A_csr)
                        except Exception as e3:
                            print(f"カスタム設定失敗: {e3}")
                            # 全試行が失敗したので、対角前処理にフォールバック
                            self._setup_diagonal_preconditioner(A_csr)
                
                # MLが作成されていれば、それを使用
                if self.ml is not None:
                    try:
                        self.solver = self.ml.aspreconditioner(cycle=self.cycle_type)
                        self.M = self.solver
                        print("PyAMG前処理器を正常に作成しました")
                    except Exception as e:
                        print(f"前処理器作成エラー: {e}")
                        # 前処理器の作成失敗時は対角前処理にフォールバック
                        self._setup_diagonal_preconditioner(A_csr)
            
            except ImportError:
                print("AMG前処理にはpyamgパッケージが必要です。対角前処理にフォールバックします。")
                self._setup_diagonal_preconditioner(A_csr)
                
        except Exception as e:
            print(f"AMG前処理設定エラー: {e}")
            print("単位行列前処理にフォールバックします")
            self.M = None
        
        return self
    
    def _setup_with_custom_params(self, A):
        """カスタムパラメータでPyAMGを設定（異なるバージョンへの対応）"""
        from pyamg import smoothed_aggregation_solver
        
        # いくつかの異なる形式を順に試す
        methods = [
            # 方法1: 辞書パラメータ
            lambda: smoothed_aggregation_solver(A, max_levels=self.max_levels, strength={'method': 'symmetric', 'theta': 0.0}),
            
            # 方法2: 文字列強度指定
            lambda: smoothed_aggregation_solver(A, max_levels=self.max_levels, strength='symmetric'),
            
            # 方法3: タプル強度指定
            lambda: smoothed_aggregation_solver(A, max_levels=self.max_levels, strength=('symmetric', 0.0)),
            
            # 方法4: callable強度関数
            lambda: smoothed_aggregation_solver(A, max_levels=self.max_levels, 
                                              strength=lambda x: pyamg.strength.classical_strength_of_connection(x, 0.0)),
            
            # 方法5: 前処理なしの単純な設定
            lambda: smoothed_aggregation_solver(A, max_levels=self.max_levels, presmoother=None, postsmoother=None)
        ]
        
        # 順番に試す
        last_error = None
        for i, method in enumerate(methods):
            try:
                print(f"方法{i+1}を試行中...")
                self.ml = method()
                print(f"方法{i+1}が成功しました")
                return
            except Exception as e:
                last_error = e
                print(f"方法{i+1}が失敗: {e}")
        
        # 全ての方法が失敗
        raise Exception(f"すべての方法が失敗しました。最後のエラー: {last_error}")
    
    def _ensure_valid_matrix(self, A):
        """行列が有効かを確認し、必要に応じて修正"""
        # NaNやInfがないか確認
        if hasattr(A, 'data'):
            has_invalid = np.isnan(A.data).any() or np.isinf(A.data).any()
            if has_invalid:
                print("警告: 行列に無効な値があります。修正を行います。")
                A.data = np.nan_to_num(A.data, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # 対称性の確認（PyAMGは対称行列で最適）
        if (A - A.T).nnz > 0:
            print("警告: 行列が非対称です。PyAMGは対称行列での使用を推奨します。")
    
    def _setup_diagonal_preconditioner(self, A):
        """対角前処理を設定（フォールバック用）"""
        print("対角前処理にフォールバックします")
        
        if hasattr(A, 'diagonal'):
            diag = A.diagonal()
        else:
            diag = A.toarray().diagonal() if hasattr(A, 'toarray') else np.diag(A)
        
        # ゼロ要素を回避
        diag_safe = np.where(np.abs(diag) > 1e-10, diag, 1.0)
        inv_diag = 1.0 / diag_safe
        
        # 対角行列を作成
        if hasattr(A, 'shape'):
            n = A.shape[0]
            self.M = sp.diags(inv_diag, 0, shape=(n, n), format='csr')
        else:
            self.M = sp.diags(inv_diag)
    
    def __call__(self, b):
        """
        AMG前処理を適用
        
        Args:
            b: 右辺ベクトル
            
        Returns:
            M^-1 * b
        """
        if self.solver is not None:
            try:
                return self.solver(b)
            except Exception as e:
                print(f"前処理適用エラー: {e}")
        
        # ソルバーが設定されていないか、エラーが発生した場合
        if self.M is not None:
            try:
                if hasattr(self.M, 'dot'):
                    return self.M.dot(b)
                else:
                    return self.M @ b
            except Exception as e:
                print(f"前処理行列適用エラー: {e}")
        
        # すべて失敗した場合は入力をそのまま返す
        return b
    
    @property
    def name(self):
        """前処理名を返す"""
        return "AMGPreconditioner"
    
    @property
    def description(self):
        """前処理の説明を返す"""
        return f"代数的マルチグリッド前処理 (max_levels={self.max_levels}, cycle_type={self.cycle_type})"