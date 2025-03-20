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
        
        # ソルバーメソッドとオプションの初期化
        self.solver_method = "direct"  # デフォルト解法
        self.solver_options = {}  # デフォルトオプション
        
        # スケーリング関連
        self.scaler = None
        self.scaling_info = None
        
        # 実装によるプロパティ初期化
        self._initialize()
    
    @abstractmethod
    def _initialize(self):
        """各実装による初期化処理"""
        pass
    
    def solve(self, b, method=None, options=None):
        """
        Ax=b を解く
        
        Args:
            b: 右辺ベクトル
            method: 解法メソッド名（設定済みのself.solver_methodを上書き）
            options: 解法オプション（設定済みのself.solver_optionsを上書き）
            
        Returns:
            解ベクトル x
        """
        # メソッドとオプションを決定（引数で上書き可能）
        actual_method = method if method is not None else self.solver_method
        actual_options = self.solver_options.copy()
        if options:
            actual_options.update(options)
        
        # 右辺ベクトルを適切な形式に変換
        b_processed = self._preprocess_vector(b)
        
        # スケーリングの適用
        b_scaled = b_processed
        if self.scaler and self.scaling_info:
            try:
                b_scaled = self.scaler.scale_b_only(b_processed, self.scaling_info)
            except Exception as e:
                print(f"Scaling error: {e}")
        
        # 解法メソッドの選択
        if actual_method not in self.solvers:
            print(f"Unsupported solver: {actual_method}, falling back to direct solver")
            actual_method = "direct"
        
        # 線形システムを解く
        try:
            solver_func = self.solvers[actual_method]
            x, iterations = solver_func(self.A, b_scaled, actual_options)
            
            # 結果のアンスケーリング
            if self.scaler and self.scaling_info:
                x = self.scaler.unscale(x, self.scaling_info)
                
            # 計算結果の記録
            self.last_iterations = iterations
                  
            return x
            
        except Exception as e:
            print(f"Solver error: {e}")
            # 直接解法にフォールバック
            try:
                x = self.solvers["direct"](self.A, b_scaled, {})[0]
                if self.scaler and self.scaling_info:
                    x = self.scaler.unscale(x, self.scaling_info)
                return x
            except Exception as fallback_error:
                print(f"Direct solver fallback error: {fallback_error}")
                raise
    
    def _preprocess_vector(self, b):
        """ベクトルを適切な形式に変換（オーバーライド可能）"""
        return b
        
    def get_available_solvers(self):
        """
        このソルバーでサポートされている解法の一覧を取得
        
        Returns:
            list: サポートされている解法名のリスト
        """
        # solversがなければ空のリストを返す
        return list(getattr(self, 'solvers', {}).keys())