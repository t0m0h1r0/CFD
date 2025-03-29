"""
線形方程式系 Ax=b を解くためのソルバー基底クラス
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
            preconditioner: 前処理器名または前処理器インスタンス
        """
        self.original_A = A
        self.A = None  # サブクラスで変換・初期化される
        self.enable_dirichlet = enable_dirichlet
        self.enable_neumann = enable_neumann
        self.last_iterations = None
        
        # ソルバーメソッドとオプションの初期化
        self.solver_method = "direct"
        self.solver_options = {}
        
        # 前処理器の初期化
        self.preconditioner = None
        self._initialize_preconditioner(preconditioner)
        
        # 実装によるプロパティ初期化
        self._initialize()
    
    def _initialize_preconditioner(self, preconditioner):
        """
        前処理器の初期化
        
        Args:
            preconditioner: 前処理器名または前処理器インスタンス
        """
        if preconditioner is None:
            return
            
        try:
            # 文字列名の場合はプラグインをロード
            if isinstance(preconditioner, str):
                try:
                    from preconditioner import get_preconditioner
                    self.preconditioner = get_preconditioner(preconditioner)
                except ImportError:
                    warnings.warn("preconditionerモジュールをインポートできません")
                except Exception as e:
                    warnings.warn(f"前処理器 '{preconditioner}' のロード中にエラー発生: {e}")
            else:
                # オブジェクトの場合はそのまま使用
                self.preconditioner = preconditioner
                
        except Exception as e:
            warnings.warn(f"前処理器の初期化に失敗しました: {e}")
    
    @abstractmethod
    def _initialize(self):
        """各実装による初期化処理"""
        pass
    
    def setup_preconditioner(self, A=None):
        """
        前処理器をセットアップ
        
        Args:
            A: システム行列 (Noneの場合は self.A を使用)
        """
        if self.preconditioner is None:
            return
            
        try:
            matrix = A if A is not None else self.A
            
            # ネイティブな行列を変換
            if hasattr(self, '_to_numpy_matrix'):
                matrix = self._to_numpy_matrix(matrix)
                
            # 前処理器をセットアップ
            self.preconditioner.setup(matrix)
            
        except Exception as e:
            warnings.warn(f"前処理器のセットアップに失敗しました: {e}")
    
    def solve(self, b, method=None, options=None):
        """
        Ax=b を解く
        
        Args:
            b: 右辺ベクトル
            method: 解法メソッド名
            options: 解法オプション
            
        Returns:
            解ベクトル x
        """
        # メソッドとオプションを決定
        actual_method = method if method is not None else self.solver_method
        actual_options = self.solver_options.copy()
        if options:
            actual_options.update(options)
        
        # 右辺ベクトルを変換
        try:
            b_processed = self._preprocess_vector(b)
        except Exception as e:
            warnings.warn(f"ベクトル前処理エラー: {e}")
            b_processed = b
        
        # 解法メソッドの選択
        if actual_method not in self.solvers:
            warnings.warn(f"未サポートのソルバー: {actual_method}、直接解法にフォールバック")
            actual_method = "direct"
        
        # 前処理器が必要なら設定（直接解法以外）
        if actual_method != "direct" and self.preconditioner is not None:
            self.setup_preconditioner()
        
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
            
            # 直接解法にフォールバック
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
            scaling_method: 無視される
            preconditioner: 前処理器名または前処理器インスタンス
            
        Returns:
            self: メソッドチェーン用
        """
        self.solver_method = method
        self.solver_options = options or {}
        
        # 前処理器を更新
        if preconditioner is not None:
            self._initialize_preconditioner(preconditioner)
            
        return self
    
    def _direct_fallback(self, A, b):
        """直接解法によるフォールバック"""
        if "direct" in self.solvers:
            return self.solvers["direct"](A, b, {})
        raise NotImplementedError("フォールバック用の直接解法がありません")
    
    def _preprocess_vector(self, b):
        """ベクトルを適切な形式に変換"""
        return b
    
    def _create_preconditioner_operator(self):
        """
        前処理演算子を作成
        
        Returns:
            前処理演算子またはNone
        """
        return self.preconditioner
    
    def get_available_solvers(self):
        """利用可能なソルバーの一覧を取得"""
        return list(getattr(self, 'solvers', {}).keys())