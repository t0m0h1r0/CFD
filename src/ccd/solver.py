import cupy as cp
import cupyx.scipy.sparse.linalg as splinalg
import matplotlib.pyplot as plt
import os
import time
from abc import ABC, abstractmethod
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
        """モニタリング開始"""
        if not self.enable:
            return
            
        self.residuals = []
        self.iterations = []
        self.elapsed_times = []
        self.start_time = time.time()
        print("\n収束状況モニタリング開始...")
    
    def update(self, iteration, residual):
        """収束状況更新"""
        if not self.enable:
            return
            
        self.iterations.append(iteration)
        self.residuals.append(residual)
        self.elapsed_times.append(time.time() - self.start_time)
        
        if iteration % self.display_interval == 0:
            print(f"  反復 {iteration}: 残差 = {residual:.6e}, 経過時間 = {self.elapsed_times[-1]:.4f}秒")
    
    def finalize(self, total_iterations, method_name, prefix=""):
        """モニタリング終了・結果可視化"""
        if not self.enable or not self.residuals:
            return
            
        print(f"収束状況モニタリング終了: 総反復回数 = {total_iterations}")
        self._visualize_convergence(method_name, prefix)
    
    def _visualize_convergence(self, method_name, prefix=""):
        """収束履歴をグラフ化"""
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
        
        filename = os.path.join(self.output_dir, f"{prefix}_convergence_{method_name.lower()}.png")
        plt.savefig(filename, dpi=150)
        plt.close()
        
        # 経過時間グラフ
        plt.figure(figsize=(10, 6))
        plt.plot(self.iterations, self.elapsed_times, 'r-o')
        plt.grid(True)
        plt.xlabel('反復回数')
        plt.ylabel('経過時間 (秒)')
        plt.title(f'{method_name} ソルバーの計算時間履歴')
        
        filename = os.path.join(self.output_dir, f"{prefix}_timing_{method_name.lower()}.png")
        plt.savefig(filename, dpi=150)
        plt.close()


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
        
        # システム初期化・行列構築
        self.system = EquationSystem(grid)
        self.enable_dirichlet, self.enable_neumann = equation_set.setup_equations(self.system, grid)
        self.matrix_A = self.system.build_matrix_system()

    def set_solver(self, method="direct", options=None, scaling_method=None):
        """ソルバー設定"""
        valid_methods = ["direct", "gmres", "cg", "cgs", "lsqr", "lsmr", "minres"]
        self.solver_method = method if method in valid_methods else "direct"
        self.solver_options = options or {}
        self.scaling_method = scaling_method
        
        # 収束モニタリング設定
        self.conv_monitor = ConvMonitor(
            enable=self.solver_options.get("monitor_convergence", False),
            display_interval=self.solver_options.get("display_interval", 10),
            output_dir=self.solver_options.get("output_dir", "results")
        )

    def _create_preconditioner(self, A):
        """前処理子作成"""
        if not self.solver_options.get("use_preconditioner", True):
            return None
        
        # ヤコビ前処理子
        diag = A.diagonal()
        diag = cp.where(cp.abs(diag) < 1e-14, 1.0, diag)
        return splinalg.LinearOperator(A.shape, matvec=lambda x: x / diag)

    def _solve_linear_system(self, A, b):
        """線形方程式系を解く"""
        # コールバック用パラメータ設定
        self.cb_A = A
        self.cb_b = b
        prefix = self.solver_options.get("prefix", "")
        self.conv_monitor.start()
        
        # 前処理子を作成
        precond = self._create_preconditioner(A)
        self.last_iterations = None
        
        # ソルバーオプション
        tol = self.solver_options.get("tol", 1e-10)
        maxiter = self.solver_options.get("maxiter", 1000)
        restart = self.solver_options.get("restart", 100)
        
        # コールバック関数
        callback = None
        if self.conv_monitor.enable:
            if self.solver_method in ["gmres", "cg", "cgs", "minres"]:
                callback = self._callback_residual
            elif self.solver_method in ["lsqr", "lsmr"]:
                callback = self._callback_lsqr
        
        try:
            # ソルバー選択
            if self.solver_method == "gmres":
                x, info = splinalg.gmres(A, b, tol=tol, maxiter=maxiter, 
                                        M=precond, restart=restart, callback=callback)
                
                self.last_iterations = getattr(info, 'iterations', None)
                if self.last_iterations is None and isinstance(info, tuple) and len(info) > 0:
                    self.last_iterations = info[0]
                
                if self.conv_monitor.enable:
                    self.conv_monitor.finalize(
                        self.last_iterations or len(self.conv_monitor.residuals),
                        "GMRES", prefix
                    )
                
                if info == 0:
                    return x
                    
            elif self.solver_method == "cg":
                x, info = splinalg.cg(A, b, tol=tol, maxiter=maxiter, 
                                    M=precond, callback=callback)
                
                self.last_iterations = getattr(info, 'iterations', None)
                
                if self.conv_monitor.enable:
                    self.conv_monitor.finalize(
                        self.last_iterations or len(self.conv_monitor.residuals),
                        "CG", prefix
                    )
                    
                if info == 0:
                    return x
                    
            elif self.solver_method == "cgs":
                x, info = splinalg.cgs(A, b, tol=tol, maxiter=maxiter, 
                                    M=precond, callback=callback)
                
                self.last_iterations = getattr(info, 'iterations', None)
                
                if self.conv_monitor.enable:
                    self.conv_monitor.finalize(
                        self.last_iterations or len(self.conv_monitor.residuals),
                        "CGS", prefix
                    )
                    
                if info == 0:
                    return x
                    
            elif self.solver_method in ["lsqr", "lsmr"]:
                solver_func = splinalg.lsqr if self.solver_method == "lsqr" else splinalg.lsmr
                
                if self.solver_method == "lsqr":
                    result = solver_func(A, b, atol=0.0, btol=0.0, 
                                       iter_lim=maxiter, show=False, callback=callback)
                    x, info, itn = result[0], result[1], result[2]
                else:
                    result = solver_func(A, b, atol=0.0, btol=0.0, 
                                       maxiter=maxiter, show=False, callback=callback)
                    x, info, itn = result[0], result[1], result[2]
                
                self.last_iterations = itn
                
                if self.conv_monitor.enable:
                    self.conv_monitor.finalize(itn, self.solver_method.upper(), prefix)
                
                if info > 0 and info < 7:
                    return x
                    
            elif self.solver_method == "minres":
                x, info = splinalg.minres(A, b, tol=tol, maxiter=maxiter, 
                                        M=precond, callback=callback)
                
                self.last_iterations = getattr(info, 'iterations', None)
                
                if self.conv_monitor.enable:
                    self.conv_monitor.finalize(
                        self.last_iterations or len(self.conv_monitor.residuals),
                        "MINRES", prefix
                    )
                    
                if info == 0:
                    return x
            
            # デフォルトは直接解法
            return splinalg.spsolve(A, b)
            
        except Exception as e:
            print(f"反復解法エラー: {e}")
            print("直接解法にフォールバック")
            return splinalg.spsolve(A, b)

    def _callback_residual(self, xk):
        """反復ソルバー用コールバック関数"""
        iteration = len(self.conv_monitor.residuals)
        residual = cp.linalg.norm(self.cb_b - self.cb_A @ xk) / cp.linalg.norm(self.cb_b)
        self.conv_monitor.update(iteration, float(residual))
    
    def _callback_lsqr(self, xk, itn, residual):
        """LSQR/LSMR用コールバック関数"""
        self.conv_monitor.update(itn, float(residual))

    def _apply_scaling(self, A, b):
        """スケーリング適用"""
        if self.scaling_method is None:
            return A, b, None, None
            
        scaler = plugin_manager.get_plugin(self.scaling_method)
        if scaler:
            print(f"スケーリング適用: {scaler.name}")
            A_scaled, b_scaled, scaling_info = scaler.scale(A, b)
            return A_scaled, b_scaled, scaling_info, scaler
            
        return A, b, None, None

    def analyze_system(self):
        """行列システム分析"""
        sparsity_info = self.system.analyze_sparsity()
        self.sparsity_info = sparsity_info
        
        print("\n行列構造分析:")
        print(f"  サイズ: {sparsity_info['matrix_size']} × {sparsity_info['matrix_size']}")
        print(f"  非ゼロ要素: {sparsity_info['non_zeros']}")
        print(f"  疎性率: {sparsity_info['sparsity']:.6f}")
        print(f"  メモリ(密): {sparsity_info['memory_dense_MB']:.2f} MB")
        print(f"  メモリ(疎): {sparsity_info['memory_sparse_MB']:.2f} MB")
        
        return sparsity_info

    def solve(self, analyze_before_solve=True, f_values=None, **boundary_values):
        """システムを解く"""
        # 前処理
        if analyze_before_solve:
            self.analyze_system()
            
        # 右辺ベクトル構築
        b = self._build_rhs_vector(f_values, **boundary_values)
        
        # スケーリング適用
        A_scaled, b_scaled, scaling_info, scaler = self._apply_scaling(self.matrix_A, b)

        # 線形システムを解く
        sol = self._solve_linear_system(A_scaled, b_scaled)
            
        # 必要に応じてアンスケール
        if scaling_info is not None and scaler is not None:
            sol = scaler.unscale(sol, scaling_info)

        # 解ベクトルから各要素を抽出
        return self._extract_solution(sol)

    @abstractmethod
    def _build_rhs_vector(self, f_values=None, **boundary_values):
        """右辺ベクトル構築（次元依存）"""
        pass
    
    @abstractmethod
    def _extract_solution(self, sol):
        """解ベクトルから各成分を抽出（次元依存）"""
        pass


class CCDSolver1D(BaseCCDSolver):
    """1次元コンパクト差分法ソルバー"""

    def __init__(self, equation_set, grid):
        if grid.is_2d:
            raise ValueError("1Dソルバーは2Dグリッドでは使用できません")
        super().__init__(equation_set, grid)

    def _build_rhs_vector(self, f_values=None, left_dirichlet=None, right_dirichlet=None,
                        left_neumann=None, right_neumann=None, **kwargs):
        """1D右辺ベクトル構築"""
        n = self.grid.n_points
        b = cp.zeros(n*4)
        
        # ソース項設定
        if f_values is not None:
            b[::4] = f_values
        
        # 境界条件状態表示
        print(f"境界条件: ディリクレ={'有効' if self.enable_dirichlet else '無効'}, "
              f"ノイマン={'有効' if self.enable_neumann else '無効'}")
        
        # ディリクレ境界条件
        if self.enable_dirichlet:
            if left_dirichlet is not None:
                b[1] = left_dirichlet  # 左境界ψ
            if right_dirichlet is not None:
                b[(n-1) * 4 + 1] = right_dirichlet  # 右境界ψ
                
            print(f"[1D] ディリクレ境界条件: 左={left_dirichlet}, 右={right_dirichlet}")
        
        # ノイマン境界条件
        if self.enable_neumann:
            if left_neumann is not None:
                b[2] = left_neumann  # 左境界ψ'
            if right_neumann is not None:
                b[(n-1) * 4 + 2] = right_neumann  # 右境界ψ'
                
            print(f"[1D] ノイマン境界条件: 左={left_neumann}, 右={right_neumann}")
        
        return b

    def _extract_solution(self, sol):
        """1D解ベクトル抽出"""
        n = self.grid.n_points
        return (
            sol[0::4][:n],    # psi
            sol[1::4][:n],    # psi_prime
            sol[2::4][:n],    # psi_second
            sol[3::4][:n]     # psi_third
        )


class CCDSolver2D(BaseCCDSolver):
    """2次元コンパクト差分法ソルバー"""

    def __init__(self, equation_set, grid):
        if not grid.is_2d:
            raise ValueError("2Dソルバーは1Dグリッドでは使用できません")
        super().__init__(equation_set, grid)

    def _build_rhs_vector(self, f_values=None, left_dirichlet=None, right_dirichlet=None, 
                      bottom_dirichlet=None, top_dirichlet=None, left_neumann=None, 
                      right_neumann=None, bottom_neumann=None, top_neumann=None, **kwargs):
        """2D右辺ベクトル構築"""
        nx, ny = self.grid.nx_points, self.grid.ny_points
        n_unknowns = 7  # ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy
        b = cp.zeros(nx*ny*n_unknowns)
        
        # ソース項設定
        if f_values is not None:
            for j in range(ny):
                for i in range(nx):
                    idx = (j * nx + i) * n_unknowns
                    b[idx] = f_values[i][j]
        
        # 境界条件状態表示
        boundary_status = [
            f"ディリクレ({'有効' if self.enable_dirichlet else '無効'})",
            f"ノイマン({'有効' if self.enable_neumann else '無効'})"
        ]
        print(f"[2D] 境界条件: {', '.join(boundary_status)}")
        
        # ディリクレ境界条件
        if self.enable_dirichlet:
            self._set_boundary_values(b, nx, ny, n_unknowns, 
                                    left_dirichlet, right_dirichlet, 
                                    bottom_dirichlet, top_dirichlet,
                                    is_dirichlet=True)
        
        # ノイマン境界条件
        if self.enable_neumann:
            self._set_boundary_values(b, nx, ny, n_unknowns, 
                                    left_neumann, right_neumann, 
                                    bottom_neumann, top_neumann,
                                    is_dirichlet=False)
            
        return b
    
    def _set_boundary_values(self, b, nx, ny, n_unknowns, left, right, bottom, top, is_dirichlet=True):
        """境界値を設定するヘルパーメソッド"""
        # インデックスオフセット (ディリクレル/ノイマン)
        x_offset = 1 if is_dirichlet else 2  # ψ_x or ψ_xx
        y_offset = 4 if is_dirichlet else 5  # ψ_y or ψ_yy
        
        # 左右境界
        for j in range(ny):
            # 左境界 (i=0)
            if left is not None:
                idx = (j * nx + 0) * n_unknowns + x_offset
                b[idx] = left[j] if isinstance(left, (list, cp.ndarray)) and j < len(left) else left
            
            # 右境界 (i=nx-1)
            if right is not None:
                idx = (j * nx + (nx-1)) * n_unknowns + x_offset
                b[idx] = right[j] if isinstance(right, (list, cp.ndarray)) and j < len(right) else right
        
        # 上下境界
        for i in range(nx):
            # 下境界 (j=0)
            if bottom is not None:
                idx = (0 * nx + i) * n_unknowns + y_offset
                b[idx] = bottom[i] if isinstance(bottom, (list, cp.ndarray)) and i < len(bottom) else bottom
            
            # 上境界 (j=ny-1)
            if top is not None:
                idx = ((ny-1) * nx + i) * n_unknowns + y_offset
                b[idx] = top[i] if isinstance(top, (list, cp.ndarray)) and i < len(top) else top

    def _extract_solution(self, sol):
        """2D解ベクトル抽出"""
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