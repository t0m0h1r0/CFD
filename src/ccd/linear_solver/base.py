"""
線形方程式系 Ax=b を解くためのソルバー基底クラス

このモジュールは、様々なバックエンド（CPU/SciPy、GPU/CuPy、JAX）を使用して
線形方程式系を効率的に解くための共通基底クラスを提供します。
"""

from abc import ABC, abstractmethod
import warnings


class LinearSolver(ABC):
    """線形方程式系 Ax=b を解くための抽象基底クラス"""
    
    def __init__(self, A, enable_dirichlet=False, enable_neumann=False, scaling_method=None, preconditioner=None):
        """
        ソルバーを初期化
        
        Args:
            A: システム行列
            enable_dirichlet: ディリクレ境界条件を使用するか
            enable_neumann: ノイマン境界条件を使用するか
            scaling_method: 無視される (以前はスケーリング手法名)
            preconditioner: 無視される (以前は前処理手法)
        """
        self.original_A = A
        self.A = None  # サブクラスで変換・初期化される
        self.enable_dirichlet = enable_dirichlet
        self.enable_neumann = enable_neumann
        self.last_iterations = None
        
        # ソルバーメソッドとオプションの初期化
        self.solver_method = "direct"
        self.solver_options = {}
        
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
        # メソッドとオプションを決定
        actual_method = method if method is not None else self.solver_method
        actual_options = self.solver_options.copy()
        if options:
            actual_options.update(options)
        
        # 右辺ベクトルを適切な形式に変換
        try:
            b_processed = self._preprocess_vector(b)
        except Exception as e:
            warnings.warn(f"ベクトル前処理エラー: {e}")
            b_processed = b
        
        # 解法メソッドの選択
        if actual_method not in self.solvers:
            warnings.warn(f"未サポートのソルバー: {actual_method}、直接解法にフォールバック")
            actual_method = "direct"
        
        # 線形システムを解く
        try:
            solver_func = self.solvers[actual_method]
            x, iterations = solver_func(self.A, b_processed, actual_options)
            
            # 計算結果の記録
            self.last_iterations = iterations
            return x
            
        except Exception as e:
            # エラー情報の表示
            warnings.warn(f"ソルバーエラー [{actual_method}]: {e}")
            
            # 直接解法にフォールバック（既に直接解法でない場合）
            if actual_method != "direct":
                try:
                    print("直接解法にフォールバックします...")
                    x, _ = self._direct_fallback(self.A, b_processed)
                    return x
                except Exception as fallback_error:
                    warnings.warn(f"直接解法フォールバックエラー: {fallback_error}")
            
            # すべてのフォールバックが失敗した場合
            raise
    
    def set_solver(self, method="direct", options=None, scaling_method=None, preconditioner=None):
        """
        ソルバーの設定
        
        Args:
            method: 解法メソッド
            options: ソルバーオプション辞書
            scaling_method: 無視される (以前はスケーリング手法名)
            preconditioner: 無視される (以前は前処理手法)
            
        Returns:
            self: メソッドチェーン用
        """
        self.solver_method = method
        self.solver_options = options or {}
        return self
    
    def _direct_fallback(self, A, b):
        """直接解法によるフォールバック（実装クラスでオーバーライド可能）"""
        if "direct" in self.solvers:
            return self.solvers["direct"](A, b, {})
        raise NotImplementedError("フォールバック用の直接解法がありません")
    
    def _preprocess_vector(self, b):
        """ベクトルを適切な形式に変換（オーバーライド可能）"""
        return b
    
    def get_available_solvers(self):
        """
        このソルバーでサポートされている解法の一覧を取得
        
        Returns:
            list: サポートされている解法名のリスト
        """
        return list(getattr(self, 'solvers', {}).keys())