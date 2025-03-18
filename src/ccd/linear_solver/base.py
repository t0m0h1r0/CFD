"""
線形方程式系 Ax=b を解くためのソルバー基底クラス
"""

from abc import ABC, abstractmethod

class LinearSolver(ABC):
    """線形方程式系 Ax=b を解くための抽象基底クラス"""
    
    def __init__(self, A, enable_dirichlet=False, enable_neumann=False, scaling_method=None):
        """
        ソルバーを初期化
        
        Args:
            A: システム行列
            enable_dirichlet: ディリクレ境界条件を使用するか
            enable_neumann: ノイマン境界条件を使用するか
            scaling_method: スケーリング手法名（オプション）
        """
        self.original_A = A
        self.enable_dirichlet = enable_dirichlet
        self.enable_neumann = enable_neumann
        self.scaling_method = scaling_method
        self.last_iterations = None
        
        # スケーリング関連
        self.scaler = None
        self.scaling_info = None
        
        # 実装によるプロパティ初期化
        self._initialize()
    
    @abstractmethod
    def _initialize(self):
        """各実装による初期化処理"""
        pass
    
    @abstractmethod
    def solve(self, b, method="direct", options=None):
        """
        Ax=b を解く
        
        Args:
            b: 右辺ベクトル
            method: 解法メソッド名（direct, gmres, cg, etc.）
            options: 解法オプション
            
        Returns:
            解ベクトル x
        """
        pass