from abc import ABC, abstractmethod
import cupy as cp
import cupyx.scipy.sparse.linalg as splinalg
import matplotlib.pyplot as plt
import os
import time

from equation_system import EquationSystem
from scaling import plugin_manager

class ConvMonitor:
    """反復ソルバーの収束状況をモニタリングするクラス"""
    
    def __init__(self, enable=False, display_interval=10, output_dir="results"):
        self.enable = enable
        self.display_interval = display_interval
        self.output_dir = output_dir
        self.residuals = []
        self.iterations = []
        self.elapsed_times = []
        self.start_time = None
        
    def start(self):
        if not self.enable:
            return
        self.residuals = []
        self.iterations = []
        self.elapsed_times = []
        self.start_time = time.time()
        print("\n収束状況モニタリングを開始...")
    
    def update(self, iteration, residual):
        if not self.enable:
            return
        self.iterations.append(iteration)
        self.residuals.append(residual)
        self.elapsed_times.append(time.time() - self.start_time)
        
        if iteration % self.display_interval == 0:
            print(f"  反復 {iteration}: 残差 = {residual:.6e}, 経過時間 = {self.elapsed_times[-1]:.4f}秒")
    
    def finalize(self, total_iterations, method_name, prefix=""):
        if not self.enable or not self.residuals:
            return
            
        print(f"収束状況モニタリングを終了: 総反復回数 = {total_iterations}")
        
        # 収束履歴グラフの作成
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 残差グラフ
        plt.figure(figsize=(10, 6))
        plt.semilogy(self.iterations, self.residuals, 'b-o')
        plt.grid(True, which="both", ls="--")
        plt.xlabel('反復回数')
        plt.ylabel('残差 (対数スケール)')
        plt.title(f'{method_name} ソルバーの収束履歴')
        plt.savefig(f"{self.output_dir}/{prefix}_convergence_{method_name.lower()}.png", dpi=150)
        plt.close()
        
        # 時間グラフ
        plt.figure(figsize=(10, 6))
        plt.plot(self.iterations, self.elapsed_times, 'r-o')
        plt.grid(True)
        plt.xlabel('反復回数')
        plt.ylabel('経過時間 (秒)')
        plt.title(f'{method_name} ソルバーの計算時間履歴')
        plt.savefig(f"{self.output_dir}/{prefix}_timing_{method_name.lower()}.png", dpi=150)
        plt.close()
        
        print(f"収束履歴グラフを保存しました: {self.output_dir}/{prefix}_convergence_{method_name.lower()}.png")

class BaseCCDSolver(ABC):
    """コンパクト差分法ソルバーの抽象基底クラス"""

    def __init__(self, equation_set, grid):
        self.equation_set = equation_set
        self.grid = grid
        self.solver_method = "direct"
        self.solver_options = {}
        self.scaling_method = None
        self.last_iterations = None
        self.sparsity_info = None
        self.conv_monitor = ConvMonitor()
        
        # システムを初期化し、行列Aを構築
        self.system = EquationSystem(grid)
        self.enable_dirichlet, self.enable_neumann = equation_set.setup_equations(self.system, grid)
        self.matrix_A = self.system.build_matrix_system()

    def set_solver(self, method="direct", options=None, scaling_method=None):
        valid_methods = ["direct", "gmres", "cg", "cgs", "lsqr", "lsmr", "minres"]
        if method not in valid_methods:
            method = "direct"
            
        self.solver_method = method
        self.solver_options = options or {}
        self.scaling_method = scaling_method
        
        # 収束モニタリング設定
        self.conv_monitor = ConvMonitor(
            enable=self.solver_options.get("monitor_convergence", False),
            display_interval=self.solver_options.get("display_interval", 10),
            output_dir=self.solver_options.get("output_dir", "results")
        )

    def _create_preconditioner(self, A):
        if not self.solver_options.get("use_preconditioner", True):
            return None
        
        # ヤコビ前処理子
        diag = A.diagonal()
        diag = cp.where(cp.abs(diag) < 1e-14, 1.0, diag)
        return splinalg.LinearOperator(A.shape, matvec=lambda x: x / diag)

    def _callback(self, xk):
        """共通コールバック関数"""
        if not self.conv_monitor.enable:
            return
        iteration = len(self.conv_monitor.residuals)
        residual = cp.linalg.norm(self.cb_b - self.cb_A @ xk) / cp.linalg.norm(self.cb_b)
        self.conv_monitor.update(iteration, float(residual))

    def _solve_linear_system(self, A, b):
        """線形方程式系を解くヘルパーメソッド"""
        # 前処理子を作成
        precond = self._create_preconditioner(A)
        self.last_iterations = None
        
        # モニタリング用の設定
        prefix = self.solver_options.get("prefix", "")
        self.conv_monitor.start()
        
        if self.conv_monitor.enable:
            self.cb_A = A
            self.cb_b = b
            
        # 反復ソルバーの共通パラメータ
        tol = self.solver_options.get("tol", 1e-10)
        maxiter = self.solver_options.get("maxiter", 1000)
        
        try:
            # 各ソルバーの実行
            if self.solver_method == "direct":
                return splinalg.spsolve(A, b)
                
            elif self.solver_method == "gmres":
                restart = self.solver_options.get("restart", 100)
                x, info = splinalg.gmres(
                    A, b, tol=tol, maxiter=maxiter, M=precond, 
                    restart=restart, callback=self._callback if self.conv_monitor.enable else None
                )
                
                self.last_iterations = getattr(info, 'iterations', None)
                if self.last_iterations is None and isinstance(info, tuple) and len(info) > 0:
                    self.last_iterations = info[0]
                
            elif self.solver_method == "cg":
                x, info = splinalg.cg(
                    A, b, tol=tol, maxiter=maxiter, M=precond, 
                    callback=self._callback if self.conv_monitor.enable else None
                )
                self.last_iterations = getattr(info, 'iterations', None)
                
            elif self.solver_method == "cgs":
                x, info = splinalg.cgs(
                    A, b, tol=tol, maxiter=maxiter, M=precond, 
                    callback=self._callback if self.conv_monitor.enable else None
                )
                self.last_iterations = getattr(info, 'iterations', None)
                
            elif self.solver_method == "minres":
                x, info = splinalg.minres(
                    A, b, tol=tol, maxiter=maxiter, M=precond, 
                    callback=self._callback if self.conv_monitor.enable else None
                )
                self.last_iterations = getattr(info, 'iterations', None)
                
            elif self.solver_method == "lsqr":
                atol = self.solver_options.get("atol", 0.0)
                btol = self.solver_options.get("btol", 0.0)
                
                # モニタリング用コールバック
                def lsqr_callback(x, itn, residual):
                    if self.conv_monitor.enable:
                        self.conv_monitor.update(itn, float(residual))
                
                x, _, itn, *_ = splinalg.lsqr(
                    A, b, atol=atol, btol=btol, iter_lim=maxiter,
                    show=False, callback=lsqr_callback if self.conv_monitor.enable else None
                )
                self.last_iterations = itn
                
            elif self.solver_method == "lsmr":
                atol = self.solver_options.get("atol", 0.0)
                btol = self.solver_options.get("btol", 0.0)
                
                # モニタリング用コールバック
                def lsmr_callback(x, itn, residual):
                    if self.conv_monitor.enable:
                        self.conv_monitor.update(itn, float(residual))
                
                x, _, itn, *_ = splinalg.lsmr(
                    A, b, atol=atol, btol=btol, maxiter=maxiter,
                    show=False, callback=lsmr_callback if self.conv_monitor.enable else None
                )
                self.last_iterations = itn
            
            # 収束モニタリングを終了
            if self.conv_monitor.enable:
                iterations = self.last_iterations or len(self.conv_monitor.residuals)
                self.conv_monitor.finalize(iterations, self.solver_method.upper(), prefix)
            
            return x
                
        except Exception as e:
            print(f"反復解法でエラーが発生しました: {e}")
            print("直接解法にフォールバックします")
            return splinalg.spsolve(A, b)

    def _apply_scaling(self, A, b):
        """行列と右辺ベクトルにスケーリングを適用"""
        if self.scaling_method is None:
            return A, b, None, None
            
        scaler = plugin_manager.get_plugin(self.scaling_method)
        if scaler:
            print(f"スケーリング手法を適用: {scaler.name} - {scaler.description}")
            A_scaled, b_scaled, scaling_info = scaler.scale(A, b)
            return A_scaled, b_scaled, scaling_info, scaler
        
        return A, b, None, None

    def analyze_system(self):
        """行列システムの疎性を分析"""
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
        """システムを解く"""
        # 行列を分析（要求された場合）
        if analyze_before_solve:
            self.analyze_system()
            
        # 右辺ベクトルbを構築
        b = self._build_rhs_vector(f_values, **boundary_values)
        
        # スケーリングを適用
        A_scaled, b_scaled, scaling_info, scaler = self._apply_scaling(self.matrix_A, b)

        # 線形システムを解く
        sol = self._solve_linear_system(A_scaled, b_scaled)
            
        # スケーリングが適用された場合は解をアンスケール
        if scaling_info is not None and scaler is not None:
            sol = scaler.unscale(sol, scaling_info)

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


class CCDSolver1D(BaseCCDSolver):
    """1次元コンパクト差分法ソルバー"""

    def __init__(self, equation_set, grid):
        if grid.is_2d:
            raise ValueError("1Dソルバーは2Dグリッドでは使用できません")
        super().__init__(equation_set, grid)

    def _build_rhs_vector(self, f_values=None, left_dirichlet=None, right_dirichlet=None,
                        left_neumann=None, right_neumann=None, **kwargs):
        n = self.grid.n_points
        b = cp.zeros(n*4)
        
        # ポアソン方程式/ソース項の値を設定
        if f_values is not None:
            for i in range(n):
                b[i * 4] = f_values[i]
        
        # 境界条件の設定状態を表示
        print(f"境界条件設定: ディリクレ = {'有効' if self.enable_dirichlet else '無効'}, "
              f"ノイマン = {'有効' if self.enable_neumann else '無効'}")
        
        # ディリクレ境界条件
        if self.enable_dirichlet:
            if left_dirichlet is not None:
                b[1] = left_dirichlet  # 左境界ディリクレ (ψ)
            if right_dirichlet is not None:
                b[(n-1) * 4 + 1] = right_dirichlet  # 右境界ディリクレ (ψ)
                
            print(f"[1Dソルバー] ディリクレ境界条件が有効: 左={left_dirichlet}, 右={right_dirichlet}")
        else:
            print("[1Dソルバー] ディリクレ境界条件が無効")
        
        # ノイマン境界条件
        if self.enable_neumann:
            if left_neumann is not None:
                b[2] = left_neumann  # 左境界ノイマン (ψ')
            if right_neumann is not None:
                b[(n-1) * 4 + 2] = right_neumann  # 右境界ノイマン (ψ')
                
            print(f"[1Dソルバー] ノイマン境界条件が有効: 左={left_neumann}, 右={right_neumann}")
        else:
            print("[1Dソルバー] ノイマン境界条件が無効")
        
        return b

    def _extract_solution(self, sol):
        """解ベクトルから各成分を抽出"""
        n = self.grid.n_points
        psi = sol[0::4][:n]
        psi_prime = sol[1::4][:n]
        psi_second = sol[2::4][:n]
        psi_third = sol[3::4][:n]

        return psi, psi_prime, psi_second, psi_third


class CCDSolver2D(BaseCCDSolver):
    """2次元コンパクト差分法ソルバー"""

    def __init__(self, equation_set, grid):
        if not grid.is_2d:
            raise ValueError("2Dソルバーは1Dグリッドでは使用できません")
        super().__init__(equation_set, grid)

    def _build_rhs_vector(self, f_values=None, left_dirichlet=None, right_dirichlet=None, 
                      bottom_dirichlet=None, top_dirichlet=None, left_neumann=None, 
                      right_neumann=None, bottom_neumann=None, top_neumann=None, **kwargs):
        nx, ny = self.grid.nx_points, self.grid.ny_points
        n_unknowns = 7  # ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy
        b = cp.zeros(nx*ny*n_unknowns)
        
        # ポアソン方程式/ソース項の値を設定
        if f_values is not None:
            for j in range(ny):
                for i in range(nx):
                    b[(j * nx + i) * n_unknowns] = f_values[i][j]
        
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
        
        # ディリクレ境界条件
        if self.enable_dirichlet:
            # x方向境界（左右）
            for j in range(ny):
                # 左境界ディリクレ(i=0)
                if left_dirichlet is not None:
                    idx = (j * nx + 0) * n_unknowns + 1  # ψ_x
                    b[idx] = left_dirichlet[j] if isinstance(left_dirichlet, (list, cp.ndarray)) and j < len(left_dirichlet) else left_dirichlet
                
                # 右境界ディリクレ(i=nx-1)
                if right_dirichlet is not None:
                    idx = (j * nx + (nx-1)) * n_unknowns + 1  # ψ_x
                    b[idx] = right_dirichlet[j] if isinstance(right_dirichlet, (list, cp.ndarray)) and j < len(right_dirichlet) else right_dirichlet
            
            # y方向境界（下上）
            for i in range(nx):
                # 下境界ディリクレ(j=0)
                if bottom_dirichlet is not None:
                    idx = (0 * nx + i) * n_unknowns + 4  # ψ_y
                    b[idx] = bottom_dirichlet[i] if isinstance(bottom_dirichlet, (list, cp.ndarray)) and i < len(bottom_dirichlet) else bottom_dirichlet
                
                # 上境界ディリクレ(j=ny-1)
                if top_dirichlet is not None:
                    idx = ((ny-1) * nx + i) * n_unknowns + 4  # ψ_y
                    b[idx] = top_dirichlet[i] if isinstance(top_dirichlet, (list, cp.ndarray)) and i < len(top_dirichlet) else top_dirichlet
        
        # ノイマン境界条件
        if self.enable_neumann:
            # x方向境界（左右）
            for j in range(ny):
                # 左境界ノイマン(i=0)
                if left_neumann is not None:
                    idx = (j * nx + 0) * n_unknowns + 2  # ψ_xx
                    b[idx] = left_neumann[j] if isinstance(left_neumann, (list, cp.ndarray)) and j < len(left_neumann) else left_neumann
                
                # 右境界ノイマン(i=nx-1)
                if right_neumann is not None:
                    idx = (j * nx + (nx-1)) * n_unknowns + 2  # ψ_xx
                    b[idx] = right_neumann[j] if isinstance(right_neumann, (list, cp.ndarray)) and j < len(right_neumann) else right_neumann
            
            # y方向境界（下上）
            for i in range(nx):
                # 下境界ノイマン(j=0)
                if bottom_neumann is not None:
                    idx = (0 * nx + i) * n_unknowns + 5  # ψ_yy
                    b[idx] = bottom_neumann[i] if isinstance(bottom_neumann, (list, cp.ndarray)) and i < len(bottom_neumann) else bottom_neumann
                
                # 上境界ノイマン(j=ny-1)
                if top_neumann is not None:
                    idx = ((ny-1) * nx + i) * n_unknowns + 5  # ψ_yy
                    b[idx] = top_neumann[i] if isinstance(top_neumann, (list, cp.ndarray)) and i < len(top_neumann) else top_neumann
                    
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