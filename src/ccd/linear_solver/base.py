"""
線形方程式系 Ax=b を解くためのソルバー基底クラス

このモジュールは、様々なバックエンド（CPU/SciPy、GPU/CuPy、JAX）を使用して
線形方程式系を効率的に解くための共通基底クラスを提供します。
"""

from abc import ABC, abstractmethod
import numpy as np
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
            scaling_method: スケーリング手法名（オプション）
            preconditioner: 前処理手法名またはインスタンス（オプション）
        """
        self.original_A = A
        self.A = None  # サブクラスで変換・初期化される
        self.enable_dirichlet = enable_dirichlet
        self.enable_neumann = enable_neumann
        self.scaling_method = scaling_method
        self.preconditioner_name = preconditioner if isinstance(preconditioner, str) else None
        self.preconditioner = preconditioner if not isinstance(preconditioner, str) else None
        self.last_iterations = None
        
        # ソルバーメソッドとオプションの初期化
        self.solver_method = "direct"  # デフォルト解法
        self.solver_options = {}  # デフォルトオプション
        
        # スケーリング関連
        self.scaler = None
        self.scaling_info = None
        
        # 前処理関連
        self._initialize_preconditioner()
        
        # 実装によるプロパティ初期化
        self._initialize()
    
    def _initialize_preconditioner(self):
        """前処理を初期化"""
        if self.preconditioner_name:
            try:
                from preconditioner import plugin_manager
                self.preconditioner = plugin_manager.get_plugin(self.preconditioner_name)
                print(f"前処理 '{self.preconditioner.name}' を初期化しました。")
            except ImportError:
                warnings.warn("preconditionerモジュールが見つかりません。")
                self.preconditioner = None
            except Exception as e:
                warnings.warn(f"前処理初期化エラー: {e}")
                self.preconditioner = None
    
    def setup_preconditioner(self, A=None):
        """
        前処理行列をセットアップ
        
        Args:
            A: 行列 (Noneの場合はself.Aを使用)
        """
        if not self.preconditioner or not hasattr(self.preconditioner, 'setup'):
            return
            
        try:
            # プラットフォーム固有の行列をNumPy形式に変換
            matrix = A if A is not None else self.A
            if hasattr(self, '_to_numpy_matrix'):
                matrix = self._to_numpy_matrix(matrix)
            
            # 前処理をセットアップ
            self.preconditioner.setup(matrix)
            print(f"前処理 '{self.preconditioner.name}' をセットアップしました。")
        except Exception as e:
            warnings.warn(f"前処理セットアップエラー: {e}")
    
    def _create_preconditioner_operator(self):
        """
        前処理演算子を作成
        
        Returns:
            前処理演算子またはNone
        """
        if not self.preconditioner:
            return None
            
        # 前処理器は既に設定されているはず
        if hasattr(self.preconditioner, 'M') and self.preconditioner.M is not None:
            return self.preconditioner.M
            
        # __call__メソッドを持つ場合
        if hasattr(self.preconditioner, '__call__'):
            return self.preconditioner
            
        return None
    
    def get_preconditioner(self):
        """
        前処理器オブジェクトを取得（視覚化用）
        
        Returns:
            前処理器オブジェクトまたはNone
        """
        return self.preconditioner
    
    @abstractmethod
    def _initialize(self):
        """各実装による初期化処理"""
        pass
    
    def solve(self, b, method=None, options=None):
        """
        Ax=b を解く（共通のエラー処理を実装）
        
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
        try:
            b_processed = self._preprocess_vector(b)
        except Exception as e:
            warnings.warn(f"Vector preprocessing error: {e}")
            b_processed = b
            
        # スケーリングの適用
        b_scaled = b_processed
        if self.scaler and self.scaling_info:
            try:
                b_scaled = self._apply_scaling_to_b(b_processed)
            except Exception as e:
                warnings.warn(f"Scaling error: {e}")
        
        # 解法メソッドの選択
        if actual_method not in self.solvers:
            warnings.warn(f"Unsupported solver: {actual_method}, falling back to direct solver")
            actual_method = "direct"
        
        # 前処理が必要なら設定
        if self.preconditioner and hasattr(self.preconditioner, 'setup') and \
           (not hasattr(self.preconditioner, 'M') or self.preconditioner.M is None):
            self.setup_preconditioner()
        
        # 線形システムを解く（エラー処理を一元化）
        try:
            solver_func = self.solvers[actual_method]
            x, iterations = solver_func(self.A, b_scaled, actual_options)
            
            # アンスケーリング処理
            if self.scaler and self.scaling_info:
                try:
                    x = self._apply_unscaling_to_x(x)
                except Exception as e:
                    warnings.warn(f"Unscaling error: {e}")
            
            # 計算結果の記録
            self.last_iterations = iterations
            return x
            
        except Exception as e:
            # エラー情報の表示
            warnings.warn(f"Solver error [{actual_method}]: {e}")
            
            # 直接解法にフォールバック（元のメソッドが直接解法でない場合）
            if actual_method != "direct":
                try:
                    print("Falling back to direct solver...")
                    x, _ = self._direct_fallback(self.A, b_scaled)
                    if self.scaler and self.scaling_info:
                        try:
                            x = self._apply_unscaling_to_x(x)
                        except Exception:
                            pass
                    return x
                except Exception as fallback_error:
                    warnings.warn(f"Direct solver fallback error: {fallback_error}")
            
            # すべてのフォールバックが失敗した場合
            raise
    
    def set_solver(self, method="direct", options=None, scaling_method=None, preconditioner=None):
        """
        ソルバーの設定
        
        Args:
            method: 解法メソッド
            options: ソルバーオプション辞書
            scaling_method: スケーリング手法名
            preconditioner: 前処理クラス名または前処理インスタンス
            
        Returns:
            self: メソッドチェーン用
        """
        self.solver_method = method
        self.solver_options = options or {}
        
        # スケーリング方法を変更した場合、初期化し直す
        if scaling_method is not None and scaling_method != self.scaling_method:
            self.scaling_method = scaling_method
            self._initialize_scaling()
            
        # 前処理器を変更した場合、初期化し直す
        if preconditioner is not None and (
               (isinstance(preconditioner, str) and preconditioner != self.preconditioner_name) or
               (not isinstance(preconditioner, str) and preconditioner != self.preconditioner)
           ):
            self.preconditioner_name = preconditioner if isinstance(preconditioner, str) else None
            self.preconditioner = preconditioner if not isinstance(preconditioner, str) else None
            self._initialize_preconditioner()
            if hasattr(self.preconditioner, 'setup'):
                self.setup_preconditioner()
                
        return self
    
    def _direct_fallback(self, A, b):
        """直接解法によるフォールバック（実装クラスでオーバーライド可能）"""
        if "direct" in self.solvers:
            return self.solvers["direct"](A, b, {})
        raise NotImplementedError("No direct solver available for fallback")
    
    def _apply_scaling_to_b(self, b):
        """右辺ベクトルにスケーリングを適用（実装クラスでオーバーライド可能）"""
        if self.scaler and self.scaling_info:
            row_scale = self.scaling_info.get('row_scale')
            if row_scale is not None:
                return b * row_scale
            else:
                D_sqrt_inv = self.scaling_info.get('D_sqrt_inv')
                if D_sqrt_inv is not None:
                    return b * D_sqrt_inv
        return b
    
    def _apply_unscaling_to_x(self, x):
        """解ベクトルにアンスケーリングを適用（実装クラスでオーバーライド可能）"""
        if self.scaler and self.scaling_info:
            col_scale = self.scaling_info.get('col_scale')
            if col_scale is not None:
                return x / col_scale
            else:
                D_sqrt_inv = self.scaling_info.get('D_sqrt_inv')
                if D_sqrt_inv is not None:
                    return x * D_sqrt_inv
        return x
    
    def _initialize_scaling(self):
        """スケーリング方法を初期化（サブクラスで実装）"""
        if not self.scaling_method:
            return
            
        try:
            from scaling import plugin_manager
            self.scaler = plugin_manager.get_plugin(self.scaling_method)
            self._prepare_scaling()
        except ImportError:
            warnings.warn("scaleモジュールが見つかりません。")
            self.scaler = None
        except Exception as e:
            warnings.warn(f"スケーリング初期化エラー: {e}")
            self.scaler = None
    
    def _prepare_scaling(self):
        """スケーリング前処理（サブクラスでオーバーライド可能）"""
        pass
    
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