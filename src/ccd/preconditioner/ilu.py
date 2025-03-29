"""
不完全LU分解前処理

このモジュールは、システム行列の不完全LU分解に基づく
効率的な前処理手法を提供します。
"""

from .base import BasePreconditioner

class ILUPreconditioner(BasePreconditioner):
    """不完全LU分解前処理"""
    
    def __init__(self, fill_factor=10, drop_tol=1e-4):
        """
        初期化
        
        Args:
            fill_factor: 充填因子（メモリ使用量制御）
            drop_tol: 切り捨て許容値（精度制御）
        """
        super().__init__()
        self.fill_factor = fill_factor
        self.drop_tol = drop_tol
        self.ilu = None
    
    def setup(self, A):
        """
        不完全LU分解前処理の設定
        
        Args:
            A: システム行列
            
        Returns:
            self: メソッドチェーン用
        """
        try:
            from scipy.sparse.linalg import spilu
            from scipy.sparse import csc_matrix
            from scipy.sparse.linalg import LinearOperator
            
            # CSC形式に変換（必要な場合）
            if hasattr(A, 'format'):
                if A.format != 'csc':
                    A_csc = csc_matrix(A)
                else:
                    A_csc = A
            else:
                # 密行列の場合
                A_csc = csc_matrix(A)
            
            # 不完全LU分解の計算
            self.ilu = spilu(A_csc, 
                           fill_factor=self.fill_factor,
                           drop_tol=self.drop_tol)
            
            # 線形演算子としてM^-1を作成
            def matvec(b):
                return self.ilu.solve(b)
            
            self.M = LinearOperator(A.shape, matvec)
        
        except Exception as e:
            print(f"ILU前処理設定エラー: {e}")
            # 単位行列にフォールバック
            self.M = None
        
        return self
    
    def __call__(self, b):
        """
        ILU前処理を適用
        
        Args:
            b: 右辺ベクトル
            
        Returns:
            M^-1 * b
        """
        if hasattr(self, 'ilu'):
            return self.ilu.solve(b)
        return b
    
    @property
    def name(self):
        """前処理名を返す"""
        return "ILUPreconditioner"
    
    @property
    def description(self):
        """前処理の説明を返す"""
        return f"不完全LU分解前処理 (fill_factor={self.fill_factor}, drop_tol={self.drop_tol})"