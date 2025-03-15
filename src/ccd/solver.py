from abc import ABC, abstractmethod
import cupy as cp
import cupyx.scipy.sparse.linalg as splinalg
import matplotlib.pyplot as plt
import os
import time

from equation_system import EquationSystem
from scaling import plugin_manager


# ========== 戦略パターンによるソルバーインターフェース ==========
class SolverStrategy(ABC):
    """線形方程式系を解くための戦略インターフェース"""
    
    @abstractmethod
    def solve(self, A, b, options=None, callback=None):
        """線形方程式系を解くメソッド"""
        pass
    
    @abstractmethod
    def get_name(self):
        """戦略の名前を返す"""
        pass


class DirectSolverStrategy(SolverStrategy):
    """直接解法の戦略クラス"""
    
    def solve(self, A, b, options=None, callback=None):
        """直接解法を使用して線形方程式系を解く"""
        x = splinalg.spsolve(A, b)
        return x, None
    
    def get_name(self):
        return "Direct"


class GMRESSolverStrategy(SolverStrategy):
    """GMRES法の戦略クラス"""
    
    def solve(self, A, b, options=None, callback=None):
        """GMRES法を使用して線形方程式系を解く"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        restart = options.get("restart", 100)
        
        # 前処理子の作成
        precond = self._create_preconditioner(A, options)
        
        # GMRES法で解く
        try:
            x, info = splinalg.gmres(
                A, b, 
                tol=tol, 
                maxiter=maxiter, 
                M=precond,
                restart=restart,
                x0=cp.ones_like(b),
                callback=callback
            )
            
            # 反復回数を取得
            iterations = None
            if hasattr(info, 'iterations'):
                iterations = info.iterations
            elif isinstance(info, tuple) and len(info) > 0:
                iterations = info[0]
            else:
                iterations = maxiter if info != 0 else None
                
            return x, iterations
        except Exception as e:
            print(f"GMRES解法でエラーが発生しました: {e}")
            print("直接解法にフォールバックします")
            return splinalg.spsolve(A, b), None
    
    def _create_preconditioner(self, A, options):
        """前処理子を作成する"""
        if not options.get("use_preconditioner", True):
            return None
        
        # ヤコビ前処理子
        diag = A.diagonal()
        diag = cp.where(cp.abs(diag) < 1e-14, 1.0, diag)
        
        D_inv = splinalg.LinearOperator(
            A.shape, 
            matvec=lambda x: x / diag
        )
        
        return D_inv
    
    def get_name(self):
        return "GMRES"


class CGSolverStrategy(SolverStrategy):
    """共役勾配法の戦略クラス"""
    
    def solve(self, A, b, options=None, callback=None):
        """共役勾配法を使用して線形方程式系を解く"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        
        # 前処理子の作成
        precond = self._create_preconditioner(A, options)
        
        # CG法で解く
        try:
            x, info = splinalg.cg(
                A, b, 
                tol=tol, 
                maxiter=maxiter, 
                M=precond, 
                x0=cp.ones_like(b),
                callback=callback
            )
            
            # 反復回数を取得
            iterations = None
            if hasattr(info, 'iterations'):
                iterations = info.iterations
            else:
                iterations = maxiter if info != 0 else None
                
            return x, iterations
        except Exception as e:
            print(f"CG解法でエラーが発生しました: {e}")
            print("直接解法にフォールバックします")
            return splinalg.spsolve(A, b), None
    
    def _create_preconditioner(self, A, options):
        """前処理子を作成する"""
        if not options.get("use_preconditioner", True):
            return None
        
        # ヤコビ前処理子
        diag = A.diagonal()
        diag = cp.where(cp.abs(diag) < 1e-14, 1.0, diag)
        
        D_inv = splinalg.LinearOperator(
            A.shape, 
            matvec=lambda x: x / diag
        )
        
        return D_inv
    
    def get_name(self):
        return "CG"


class CGSSolverStrategy(SolverStrategy):
    """CGS法（共役勾配の平方）の戦略クラス"""
    
    def solve(self, A, b, options=None, callback=None):
        """CGS法を使用して線形方程式系を解く"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        
        # 前処理子を作成
        precond = self._create_preconditioner(A, options)
        
        # CGS法で解く
        try:
            x, info = splinalg.cgs(
                A, b, 
                tol=tol, 
                maxiter=maxiter, 
                M=precond,
                x0=cp.ones_like(b),
                callback=callback
            )
            
            # 反復回数を取得
            iterations = None
            if hasattr(info, 'iterations'):
                iterations = info.iterations
            else:
                iterations = maxiter if info != 0 else None
                
            return x, iterations
        except Exception as e:
            print(f"CGS解法でエラーが発生しました: {e}")
            print("直接解法にフォールバックします")
            return splinalg.spsolve(A, b), None
    
    def _create_preconditioner(self, A, options):
        """前処理子を作成する"""
        if not options.get("use_preconditioner", True):
            return None
        
        # ヤコビ前処理子
        diag = A.diagonal()
        diag = cp.where(cp.abs(diag) < 1e-14, 1.0, diag)
        
        D_inv = splinalg.LinearOperator(
            A.shape, 
            matvec=lambda x: x / diag
        )
        
        return D_inv
    
    def get_name(self):
        return "CGS"


class MINRESSolverStrategy(SolverStrategy):
    """MINRES法の戦略クラス"""
    
    def solve(self, A, b, options=None, callback=None):
        """MINRES法を使用して線形方程式系を解く"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        
        # 前処理子を作成
        precond = self._create_preconditioner(A, options)
        
        # MINRES法で解く
        try:
            x, info = splinalg.minres(
                A, b, 
                tol=tol, 
                maxiter=maxiter, 
                M=precond,
                x0=cp.ones_like(b),
                callback=callback
            )
            
            # 反復回数を取得
            iterations = None
            if hasattr(info, 'iterations'):
                iterations = info.iterations
            else:
                iterations = maxiter if info != 0 else None
                
            return x, iterations
        except Exception as e:
            print(f"MINRES解法でエラーが発生しました: {e}")
            print("直接解法にフォールバックします")
            return splinalg.spsolve(A, b), None
    
    def _create_preconditioner(self, A, options):
        """前処理子を作成する"""
        if not options.get("use_preconditioner", True):
            return None
        
        # ヤコビ前処理子
        diag = A.diagonal()
        diag = cp.where(cp.abs(diag) < 1e-14, 1.0, diag)
        
        D_inv = splinalg.LinearOperator(
            A.shape, 
            matvec=lambda x: x / diag
        )
        
        return D_inv
    
    def get_name(self):
        return "MINRES"


class LSQRSolverStrategy(SolverStrategy):
    """LSQR法の戦略クラス"""
    
    def solve(self, A, b, options=None, callback=None):
        """LSQR法を使用して線形方程式系を解く"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        atol = options.get("atol", 0.0)
        btol = options.get("btol", 0.0)
        
        # LSQR法で解く
        try:
            lsqr_callback = None
            if callback:
                def lsqr_callback_wrapper(x, itn, residual):
                    callback(x)
                lsqr_callback = lsqr_callback_wrapper
                
            x, info, itn, _, _, _, _, _, _, _ = splinalg.lsqr(
                A, b, 
            )
            
            return x, itn
        except Exception as e:
            print(f"LSQR解法でエラーが発生しました: {e}")
            print("直接解法にフォールバックします")
            return splinalg.spsolve(A, b), None
    
    def get_name(self):
        return "LSQR"


class LSMRSolverStrategy(SolverStrategy):
    """LSMR法の戦略クラス"""
    
    def solve(self, A, b, options=None, callback=None):
        """LSMR法を使用して線形方程式系を解く"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        atol = options.get("atol", 0.0)
        btol = options.get("btol", 0.0)
        
        # LSMR法で解く
        try:
            lsmr_callback = None
            if callback:
                def lsmr_callback_wrapper(x, itn, residual):
                    callback(x)
                lsmr_callback = lsmr_callback_wrapper
                
            x, info, itn, _, _, _, _, _ = splinalg.lsmr(
                A, b, 
                atol=atol, 
                btol=btol, 
                maxiter=maxiter,
                x0=cp.ones_like(b),
                #show=False,
                #callback=lsmr_callback
            )
            
            return x, itn
        except Exception as e:
            print(f"LSMR解法でエラーが発生しました: {e}")
            print("直接解法にフォールバックします")
            return splinalg.spsolve(A, b), None
    
    def get_name(self):
        return "LSMR"


# ========== ファクトリーパターンによるソルバー生成 ==========
class SolverFactory:
    """適切なソルバー戦略を作成するファクトリークラス"""
    
    @staticmethod
    def create_solver(method="direct"):
        """
        ソルバー戦略を作成
        
        Args:
            method: 解法メソッド名
            
        Returns:
            SolverStrategy インスタンス
        """
        method = method.lower()
        
        if method == "gmres":
            return GMRESSolverStrategy()
        elif method == "cg":
            return CGSolverStrategy()
        elif method == "cgs":
            return CGSSolverStrategy()
        elif method == "lsqr":
            return LSQRSolverStrategy()
        elif method == "lsmr":
            return LSMRSolverStrategy()
        elif method == "minres":
            return MINRESSolverStrategy()
        else:
            return DirectSolverStrategy()


# ========== 収束モニタリングの分離 ==========
class ConvMonitor:
    """反復ソルバーの収束状況をモニタリングするクラス"""
    
    def __init__(self, enable=False, display_interval=10, output_dir="results"):
        """
        モニタリング設定を初期化
        
        Args:
            enable: モニタリングを有効にするかどうか
            display_interval: 表示間隔（何反復ごとに表示するか）
            output_dir: 出力ディレクトリ
        """
        self.enable = enable
        self.display_interval = display_interval
        self.output_dir = output_dir
        self.residuals = []
        self.iterations = []
        self.start_time = None
        self.elapsed_times = []
        self.cb_A = None
        self.cb_b = None
        
    def set_system(self, A, b):
        """
        残差計算のためのシステム情報を設定
        
        Args:
            A: システム行列
            b: 右辺ベクトル
        """
        self.cb_A = A
        self.cb_b = b
        
    def start(self):
        """モニタリングを開始"""
        if self.enable:
            self.residuals = []
            self.iterations = []
            self.elapsed_times = []
            self.start_time = time.time()
            print("\n収束状況モニタリングを開始...")
    
    def update(self, iteration, residual):
        """
        収束状況を更新
        
        Args:
            iteration: 現在の反復回数
            residual: 現在の残差
        """
        if not self.enable:
            return
            
        self.iterations.append(iteration)
        self.residuals.append(residual)
        self.elapsed_times.append(time.time() - self.start_time)
        
        # 表示間隔ごとに出力
        if iteration % self.display_interval == 0:
            print(f"  反復 {iteration}: 残差 = {residual:.6e}, 経過時間 = {self.elapsed_times[-1]:.4f}秒")
    
    def get_callback(self):
        """反復ソルバー用のコールバック関数を返す"""
        def callback(xk):
            iteration = len(self.residuals)
            # xkから残差を計算
            if self.cb_A is not None and self.cb_b is not None:
                residual = cp.linalg.norm(self.cb_b - self.cb_A @ xk) / cp.linalg.norm(self.cb_b)
                self.update(iteration, float(residual))
        
        return callback if self.enable else None
    
    def finalize(self, total_iterations, method_name, prefix=""):
        """
        モニタリングを終了し、結果を可視化
        
        Args:
            total_iterations: 総反復回数
            method_name: ソルバー手法名
            prefix: 出力ファイル名の接頭辞
        """
        if not self.enable or not self.residuals:
            return
            
        print(f"収束状況モニタリングを終了: 総反復回数 = {total_iterations}")
        
        # 収束履歴を可視化
        self.visualize_convergence(method_name, prefix)
    
    def visualize_convergence(self, method_name, prefix=""):
        """
        収束履歴をグラフとして可視化
        
        Args:
            method_name: ソルバー手法名
            prefix: 出力ファイル名の接頭辞
        """
        if not self.enable or not self.residuals:
            return
            
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 残差の推移グラフ
        plt.figure(figsize=(10, 6))
        plt.semilogy(self.iterations, self.residuals, 'b-o')
        plt.grid(True, which="both", ls="--")
        plt.xlabel('反復回数')
        plt.ylabel('残差 (対数スケール)')
        plt.title(f'{method_name} ソルバーの収束履歴')
        
        # 保存
        filename = os.path.join(self.output_dir, f"{prefix}_convergence_{method_name.lower()}.png")
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"収束履歴グラフを保存しました: {filename}")
        
        # 経過時間グラフも作成
        plt.figure(figsize=(10, 6))
        plt.plot(self.iterations, self.elapsed_times, 'r-o')
        plt.grid(True)
        plt.xlabel('反復回数')
        plt.ylabel('経過時間 (秒)')
        plt.title(f'{method_name} ソルバーの計算時間履歴')
        
        # 保存
        filename = os.path.join(self.output_dir, f"{prefix}_timing_{method_name.lower()}.png")
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"計算時間履歴グラフを保存しました: {filename}")


# ========== 線形システムソルバー ==========
class LinearSystemSolver:
    """
    線形方程式系を解くためのクラス
    単一責任: 線形システムの解法とスケーリングの管理
    """
    
    def __init__(self, method="direct", options=None, scaling_method=None):
        """
        線形方程式系ソルバーを初期化
        
        Args:
            method: 解法メソッド
            options: ソルバーオプション
            scaling_method: スケーリング手法
        """
        self.solver_strategy = SolverFactory.create_solver(method)
        self.options = options or {}
        self.scaling_method = scaling_method
        self.conv_monitor = ConvMonitor(
            enable=self.options.get("monitor_convergence", False),
            display_interval=self.options.get("display_interval", 10),
            output_dir=self.options.get("output_dir", "results")
        )
        self.last_iterations = None
    
    def set_options(self, options):
        """ソルバーオプションを設定"""
        self.options = options or {}
        # 収束モニタリングを更新
        self.conv_monitor = ConvMonitor(
            enable=self.options.get("monitor_convergence", False),
            display_interval=self.options.get("display_interval", 10),
            output_dir=self.options.get("output_dir", "results")
        )
    
    def set_method(self, method):
        """解法メソッドを設定"""
        self.solver_strategy = SolverFactory.create_solver(method)
    
    def set_scaling_method(self, scaling_method):
        """スケーリング手法を設定"""
        self.scaling_method = scaling_method
    
    def solve(self, A, b):
        """
        線形方程式系を解く
        
        Args:
            A: システム行列
            b: 右辺ベクトル
            
        Returns:
            解ベクトル
        """
        # スケーリングを適用
        A_scaled, b_scaled, scaling_info, scaler = self._apply_scaling(A, b)
        
        # 収束モニタリングを準備
        prefix = self.options.get("prefix", "")
        self.conv_monitor.start()
        self.conv_monitor.set_system(A_scaled, b_scaled)
        
        # 線形方程式系を解く
        callback = self.conv_monitor.get_callback() if self.conv_monitor.enable else None
        x, iterations = self.solver_strategy.solve(A_scaled, b_scaled, self.options, callback)
        
        # 反復回数を保存
        self.last_iterations = iterations
        
        # 収束モニタリングを終了
        if self.conv_monitor.enable:
            self.conv_monitor.finalize(
                iterations or len(self.conv_monitor.residuals),
                self.solver_strategy.get_name(),
                prefix
            )
        
        # スケーリングを解除
        if scaling_info is not None and scaler is not None:
            x = scaler.unscale(x, scaling_info)
            
        return x
    
    def _apply_scaling(self, A, b):
        """
        行列と右辺ベクトルにスケーリングを適用
        
        Args:
            A: システム行列
            b: 右辺ベクトル
            
        Returns:
            tuple: (scaled_A, scaled_b, scaling_info, scaler)
        """
        scaling_info = None
        scaler = None
        
        if self.scaling_method is not None:
            scaler = plugin_manager.get_plugin(self.scaling_method)
            if scaler:
                print(f"スケーリング手法を適用: {scaler.name} - {scaler.description}")
                A_scaled, b_scaled, scaling_info = scaler.scale(A, b)
                return A_scaled, b_scaled, scaling_info, scaler
        
        return A, b, None, None


# ========== コンパクト差分法ソルバー基底クラス ==========
class BaseCCDSolver(ABC):
    """コンパクト差分法ソルバーの抽象基底クラス"""

    def __init__(self, equation_set, grid):
        """
        ソルバーを初期化
        
        Args:
            equation_set: 使用する方程式セット
            grid: グリッドオブジェクト
        """
        self.equation_set = equation_set
        self.grid = grid
        self.linear_solver = LinearSystemSolver()
        
        # システムを初期化し、行列Aを構築
        self.system = EquationSystem(grid)
        self.enable_dirichlet, self.enable_neumann = equation_set.setup_equations(self.system, grid)
        self.matrix_A = self.system.build_matrix_system()
        self.sparsity_info = None

    def set_solver(self, method="direct", options=None, scaling_method=None):
        """
        ソルバーメソッドとオプションを設定
        
        Args:
            method: 解法メソッド
            options: ソルバーオプション辞書
            scaling_method: スケーリング手法名
        """
        self.linear_solver.set_method(method)
        
        if options:
            self.linear_solver.set_options(options)
            
        if scaling_method is not None:
            self.linear_solver.set_scaling_method(scaling_method)
    
    @property
    def scaling_method(self):
        """スケーリング手法を取得"""
        return self.linear_solver.scaling_method
    
    @scaling_method.setter
    def scaling_method(self, value):
        """スケーリング手法を設定"""
        self.linear_solver.set_scaling_method(value)
    
    @property
    def last_iterations(self):
        """最後の反復回数を取得"""
        return self.linear_solver.last_iterations

    def analyze_system(self):
        """
        行列システムの疎性を分析
        
        Returns:
            疎性情報の辞書
        """
        sparsity_info = self.system.analyze_sparsity()
        self.sparsity_info = sparsity_info
        
        print("\n行列構造分析:")
        print(f"  行列サイズ: {sparsity_info['matrix_size']} x {sparsity_info['matrix_size']}")
        print(f"  非ゼロ要素数: {sparsity_info['non_zeros']}")
        print(f"  疎性率: {sparsity_info['sparsity']:.6f}")
        print(f"  メモリ使用量(密行列): {sparsity_info['memory_dense_MB']:.2f} MB")
        print(f"  メモリ使用量(疎行列): {sparsity_info['memory_sparse_MB']:.2f} MB")
        
        return sparsity_info

    def solve(self, analyze_before_solve=True, f_values=None, **boundary_values):
        """
        システムを解く
        
        Args:
            analyze_before_solve: 解く前に行列を分析するかどうか
            f_values: 支配方程式の右辺値
            **boundary_values: 境界値の辞書（ディメンションに依存）
            
        Returns:
            解コンポーネント
        """
        # 行列を分析（要求された場合）
        if analyze_before_solve:
            self.analyze_system()
            
        # 右辺ベクトルbを構築
        b = self._build_rhs_vector(f_values, **boundary_values)
        
        # 線形システムを解く
        sol = self.linear_solver.solve(self.matrix_A, b)

        # 解ベクトルから各要素を抽出
        return self._extract_solution(sol)

    @abstractmethod
    def _build_rhs_vector(self, f_values=None, **boundary_values):
        """右辺ベクトルを構築（次元による具体実装）"""
        pass
    
    @abstractmethod
    def _extract_solution(self, sol):
        """解ベクトルから各成分を抽出（次元による具体実装）"""
        pass


# ========== 1次元ソルバー ==========
class CCDSolver1D(BaseCCDSolver):
    """1次元コンパクト差分法ソルバー"""

    def __init__(self, equation_set, grid):
        """
        1Dソルバーを初期化
        
        Args:
            equation_set: 方程式セット
            grid: 1D グリッドオブジェクト
        """
        if grid.is_2d:
            raise ValueError("1Dソルバーは2Dグリッドでは使用できません")
            
        super().__init__(equation_set, grid)

    def _build_rhs_vector(self, f_values=None, left_dirichlet=None, right_dirichlet=None,
                        left_neumann=None, right_neumann=None, **kwargs):
        """
        1D右辺ベクトルを構築
        
        Args:
            f_values: ソース項の値
            left_dirichlet, right_dirichlet: ディリクレ境界値
            left_neumann, right_neumann: ノイマン境界値
            
        Returns:
            右辺ベクトル
        """
        n = self.grid.n_points
        b = cp.zeros(n*4)
        
        # 境界条件の設定状態を表示
        print(f"境界条件設定: ディリクレ = {'有効' if self.enable_dirichlet else '無効'}, "
              f"ノイマン = {'有効' if self.enable_neumann else '無効'}")
        
        # 各グリッド点に対して右辺値を設定
        for i in range(n):
            # 基本インデックス
            base_idx = i * 4
            
            # どのタイプの点か判定
            if i == 0:  # 左境界
                # 0: 支配方程式 (ψ)
                if f_values is not None:
                    b[base_idx] = f_values[i]
                
                # 1: ディリクレ境界条件 (ψ)
                if self.enable_dirichlet and left_dirichlet is not None:
                    b[base_idx + 1] = left_dirichlet
                
                # 2: ノイマン境界条件 (ψ')
                if self.enable_neumann and left_neumann is not None:
                    b[base_idx + 2] = left_neumann
                
                # 3: 補助方程式 (通常は0)
                # そのまま0のままでOK
                
            elif i == n - 1:  # 右境界
                # 0: 支配方程式 (ψ)
                if f_values is not None:
                    b[base_idx] = f_values[i]
                
                # 1: ディリクレ境界条件 (ψ)
                if self.enable_dirichlet and right_dirichlet is not None:
                    b[base_idx + 1] = right_dirichlet
                
                # 2: ノイマン境界条件 (ψ')
                if self.enable_neumann and right_neumann is not None:
                    b[base_idx + 2] = right_neumann
                
                # 3: 補助方程式 (通常は0)
                # そのまま0のままでOK
                
            else:  # 内部点
                # 0: 支配方程式 (ψ)
                if f_values is not None:
                    b[base_idx] = f_values[i]
                
                # 1-3: 内部点の補助方程式 (通常は0)
                # そのまま0のままでOK
        
        if self.enable_dirichlet:
            print(f"[1Dソルバー] ディリクレ境界条件: 左={left_dirichlet}, 右={right_dirichlet}")
        else:
            print("[1Dソルバー] ディリクレ境界条件は無効")
        
        if self.enable_neumann:
            print(f"[1Dソルバー] ノイマン境界条件: 左={left_neumann}, 右={right_neumann}")
        else:
            print("[1Dソルバー] ノイマン境界条件は無効")
        
        return b

    def _extract_solution(self, sol):
        """解ベクトルから各成分を抽出"""
        n = self.grid.n_points
        psi = sol[0::4][:n]
        psi_prime = sol[1::4][:n]
        psi_second = sol[2::4][:n]
        psi_third = sol[3::4][:n]

        return psi, psi_prime, psi_second, psi_third


# ========== 2次元ソルバー ==========
class CCDSolver2D(BaseCCDSolver):
    """2次元コンパクト差分法ソルバー"""

    def __init__(self, equation_set, grid):
        """
        2Dソルバーを初期化
        
        Args:
            equation_set: 方程式セット
            grid: 2D グリッドオブジェクト
        """
        if not grid.is_2d:
            raise ValueError("2Dソルバーは1Dグリッドでは使用できません")
            
        super().__init__(equation_set, grid)

    def _build_rhs_vector(self, f_values=None, left_dirichlet=None, right_dirichlet=None,
                    bottom_dirichlet=None, top_dirichlet=None, left_neumann=None, 
                    right_neumann=None, bottom_neumann=None, top_neumann=None, **kwargs):
        """
        2D右辺ベクトルを構築
        
        Args:
            f_values: ソース項の値 (nx×ny配列)
            left_dirichlet, right_dirichlet, bottom_dirichlet, top_dirichlet: 境界値
            left_neumann, right_neumann, bottom_neumann, top_neumann: 境界導関数
            
        Returns:
            右辺ベクトル
        """
        nx, ny = self.grid.nx_points, self.grid.ny_points
        n_unknowns = 7  # ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy
        b = cp.zeros(nx*ny*n_unknowns)
        
        # 境界条件の設定状態を表示
        boundary_status = []
        if self.enable_dirichlet:
            boundary_status.append("ディリクレ(有効)")
        else:
            boundary_status.append("ディリクレ(無効)")
        
        if self.enable_neumann:
            boundary_status.append("ノイマン(有効)")
        else:
            boundary_status.append("ノイマン(無効)")
        
        print(f"[2Dソルバー] 境界条件: {', '.join(boundary_status)}")
        
        # 各グリッド点に対して右辺値を設定
        for j in range(ny):
            for i in range(nx):
                # グリッド点の基本インデックス
                base_idx = (j * nx + i) * n_unknowns
                
                # 0: ψに対応する方程式の右辺値 (支配方程式)
                if f_values is not None:
                    b[base_idx] = f_values[i, j]
                
                # どのタイプの点か判定してから適切な右辺値を設定
                if i == 0:  # 左境界
                    if self.enable_dirichlet and left_dirichlet is not None:
                        # 1: ψ_x (左境界のディリクレ条件)
                        if isinstance(left_dirichlet, (list, cp.ndarray)) and j < len(left_dirichlet):
                            b[base_idx + 1] = left_dirichlet[j]
                        else:
                            b[base_idx + 1] = left_dirichlet
                    
                    if self.enable_neumann and left_neumann is not None:
                        # 2: ψ_xx (左境界のノイマン条件)
                        if isinstance(left_neumann, (list, cp.ndarray)) and j < len(left_neumann):
                            b[base_idx + 2] = left_neumann[j]
                        else:
                            b[base_idx + 2] = left_neumann
                
                elif i == nx - 1:  # 右境界
                    if self.enable_dirichlet and right_dirichlet is not None:
                        # 1: ψ_x (右境界のディリクレ条件)
                        if isinstance(right_dirichlet, (list, cp.ndarray)) and j < len(right_dirichlet):
                            b[base_idx + 1] = right_dirichlet[j]
                        else:
                            b[base_idx + 1] = right_dirichlet
                    
                    if self.enable_neumann and right_neumann is not None:
                        # 2: ψ_xx (右境界のノイマン条件)
                        if isinstance(right_neumann, (list, cp.ndarray)) and j < len(right_neumann):
                            b[base_idx + 2] = right_neumann[j]
                        else:
                            b[base_idx + 2] = right_neumann
                
                if j == 0:  # 下境界
                    if self.enable_dirichlet and bottom_dirichlet is not None:
                        # 4: ψ_y (下境界のディリクレ条件)
                        if isinstance(bottom_dirichlet, (list, cp.ndarray)) and i < len(bottom_dirichlet):
                            b[base_idx + 4] = bottom_dirichlet[i]
                        else:
                            b[base_idx + 4] = bottom_dirichlet
                    
                    if self.enable_neumann and bottom_neumann is not None:
                        # 5: ψ_yy (下境界のノイマン条件)
                        if isinstance(bottom_neumann, (list, cp.ndarray)) and i < len(bottom_neumann):
                            b[base_idx + 5] = bottom_neumann[i]
                        else:
                            b[base_idx + 5] = bottom_neumann
                
                elif j == ny - 1:  # 上境界
                    if self.enable_dirichlet and top_dirichlet is not None:
                        # 4: ψ_y (上境界のディリクレ条件)
                        if isinstance(top_dirichlet, (list, cp.ndarray)) and i < len(top_dirichlet):
                            b[base_idx + 4] = top_dirichlet[i]
                        else:
                            b[base_idx + 4] = top_dirichlet
                    
                    if self.enable_neumann and top_neumann is not None:
                        # 5: ψ_yy (上境界のノイマン条件)
                        if isinstance(top_neumann, (list, cp.ndarray)) and i < len(top_neumann):
                            b[base_idx + 5] = top_neumann[i]
                        else:
                            b[base_idx + 5] = top_neumann
                
                # 残りの補助方程式の右辺値はデフォルト0
        
        # 境界条件の設定を出力
        if self.enable_dirichlet:
            print(f"[2Dソルバー] ディリクレ境界条件: 左={left_dirichlet is not None}, 右={right_dirichlet is not None}, "
                 f"下={bottom_dirichlet is not None}, 上={top_dirichlet is not None}")
        else:
            print("[2Dソルバー] ディリクレ境界条件は無効")
        
        if self.enable_neumann:
            print(f"[2Dソルバー] ノイマン境界条件: 左={left_neumann is not None}, 右={right_neumann is not None}, "
                 f"下={bottom_neumann is not None}, 上={top_neumann is not None}")
        else:
            print("[2Dソルバー] ノイマン境界条件は無効")
        
        return b

    def _extract_solution(self, sol):
        """解ベクトルから各成分を抽出"""
        nx, ny = self.grid.nx_points, self.grid.ny_points
        n_unknowns = 7  # ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy
        
        # 解配列を初期化
        psi = cp.zeros((nx, ny))
        psi_x = cp.zeros((nx, ny))
        psi_xx = cp.zeros((nx, ny))
        psi_xxx = cp.zeros((nx, ny))
        psi_y = cp.zeros((nx, ny))
        psi_yy = cp.zeros((nx, ny))
        psi_yyy = cp.zeros((nx, ny))
        
        # 各グリッド点の値を抽出
        for j in range(ny):
            for i in range(nx):
                idx = (j * nx + i) * n_unknowns
                psi[i, j] = sol[idx]
                psi_x[i, j] = sol[idx + 1]
                psi_xx[i, j] = sol[idx + 2]
                psi_xxx[i, j] = sol[idx + 3]
                psi_y[i, j] = sol[idx + 4]
                psi_yy[i, j] = sol[idx + 5]
                psi_yyy[i, j] = sol[idx + 6]
        
        return psi, psi_x, psi_xx, psi_xxx, psi_y, psi_yy, psi_yyy