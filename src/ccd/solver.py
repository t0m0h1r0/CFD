from abc import ABC, abstractmethod
import cupy as cp
import cupyx.scipy.sparse.linalg as splinalg

# ソルバー戦略のインターフェース
class SolverStrategy(ABC):
    """ソルバー戦略のインターフェース"""

    @abstractmethod
    def solve(self, A, b, options=None):
        """
        線形方程式系 Ax = b を解く
        
        Args:
            A: システム行列
            b: 右辺ベクトル
            options: ソルバーオプション
            
        Returns:
            tuple: (解ベクトル, 情報)
        """
        pass


# 具体的なソルバー実装
class DirectSolver(SolverStrategy):
    """疎行列LU分解を使用する直接解法"""
    
    def solve(self, A, b, options=None):
        """直接法で解く"""
        x = splinalg.spsolve(A, b)
        return x, None


class GMRESSolver(SolverStrategy):
    """GMRES反復解法"""
    
    def solve(self, A, b, options=None):
        """GMRES法で解く"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        restart = options.get("restart", 100)
        precond = options.get("preconditioner")
        
        try:
            x, info = splinalg.gmres(
                A, b, 
                tol=tol, 
                maxiter=maxiter, 
                M=precond,
                restart=restart
            )
            
            # 反復情報の抽出
            iterations = None
            if hasattr(info, 'iterations'):
                iterations = info.iterations
            elif isinstance(info, tuple) and len(info) > 0:
                iterations = info[0]
            else:
                iterations = maxiter if info != 0 else None
                
            return x, iterations
        except Exception as e:
            print(f"GMRESソルバーでエラーが発生しました: {e}")
            x = splinalg.spsolve(A, b)
            return x, None


class CGSolver(SolverStrategy):
    """共役勾配法ソルバー"""
    
    def solve(self, A, b, options=None):
        """CG法で解く"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        precond = options.get("preconditioner")
        
        try:
            x, info = splinalg.cg(A, b, tol=tol, maxiter=maxiter, M=precond)
            
            # 反復情報の抽出
            iterations = None
            if hasattr(info, 'iterations'):
                iterations = info.iterations
            else:
                iterations = maxiter if info != 0 else None
                
            return x, iterations
        except Exception as e:
            print(f"CG解法でエラーが発生しました: {e}")
            x = splinalg.spsolve(A, b)
            return x, None


class CGSSolver(SolverStrategy):
    """共役勾配二乗法ソルバー"""
    
    def solve(self, A, b, options=None):
        """CGS法で解く"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        maxiter = options.get("maxiter", 1000)
        precond = options.get("preconditioner")
        
        try:
            x, info = splinalg.cgs(A, b, tol=tol, maxiter=maxiter, M=precond)
            
            # 反復情報の抽出
            iterations = None
            if hasattr(info, 'iterations'):
                iterations = info.iterations
            else:
                iterations = maxiter if info != 0 else None
                
            return x, iterations
        except Exception as e:
            print(f"CGS解法でエラーが発生しました: {e}")
            x = splinalg.spsolve(A, b)
            return x, None


# 前処理子インターフェース
class Preconditioner(ABC):
    """前処理子インターフェース"""
    
    @abstractmethod
    def create(self, A):
        """
        行列Aに対する前処理子を作成
        
        Args:
            A: システム行列
            
        Returns:
            前処理子オペレータ
        """
        pass


# 具体的な前処理子実装
class JacobiPreconditioner(Preconditioner):
    """ヤコビ（対角）前処理子"""
    
    def create(self, A):
        """ヤコビ前処理子を作成"""
        diag = A.diagonal()
        diag = cp.where(cp.abs(diag) < 1e-14, 1.0, diag)
        
        D_inv = splinalg.LinearOperator(
            A.shape, 
            matvec=lambda x: x / diag
        )
        
        return D_inv


# ファクトリークラス
class SolverFactory:
    """ソルバー戦略を作成するファクトリー"""
    
    @staticmethod
    def create(method="direct"):
        """名前からソルバー戦略を作成"""
        if method == "gmres":
            return GMRESSolver()
        elif method == "cg":
            return CGSolver()
        elif method == "cgs":
            return CGSSolver()
        else:
            return DirectSolver()


class PreconditionerFactory:
    """前処理子を作成するファクトリー"""
    
    @staticmethod
    def create(type="jacobi"):
        """タイプから前処理子を作成"""
        if type == "jacobi":
            return JacobiPreconditioner()
        else:
            return JacobiPreconditioner()  # デフォルト


# ベースCCDソルバー
class BaseCCDSolver(ABC):
    """コンパクト差分法ソルバーの抽象基底クラス"""

    def __init__(self, system, grid):
        """
        ソルバーを初期化
        
        Args:
            system: 方程式システム
            grid: グリッドオブジェクト
        """
        self.system = system
        self.grid = grid
        self.solver_method = "direct"
        self.solver_options = {}
        self.scaling_method = None
        self.last_iterations = None
        self.sparsity_info = None

    def set_solver(self, method="direct", options=None, scaling_method=None):
        """
        ソルバーメソッドとオプションを設定
        
        Args:
            method: 解法メソッド ("direct", "gmres", "cg", "cgs")
            options: ソルバーオプション辞書
            scaling_method: スケーリング手法名（Noneの場合はスケーリングなし）
        """
        valid_methods = ["direct", "gmres", "cg", "cgs"]
        if method not in valid_methods:
            method = "direct"
            
        self.solver_method = method
        self.solver_options = options or {}
        self.scaling_method = scaling_method

    def _create_preconditioner(self, A):
        """
        前処理子を作成
        
        Args:
            A: システム行列
            
        Returns:
            前処理子オペレータ
        """
        if not self.solver_options.get("use_preconditioner", True):
            return None
        
        preconditioner_factory = PreconditionerFactory()
        preconditioner = preconditioner_factory.create("jacobi")
        return preconditioner.create(A)

    def _solve_with_strategy(self, A, b):
        """
        選択されたソルバー戦略を使用してシステムを解く
        
        Args:
            A: システム行列
            b: 右辺ベクトル
            
        Returns:
            tuple: (解, 反復回数)
        """
        # ソルバー戦略を作成
        solver_factory = SolverFactory()
        solver = solver_factory.create(self.solver_method)
        
        # 必要に応じて前処理子を作成
        precond = self._create_preconditioner(A)
        
        # オプションに前処理子を追加
        options = dict(self.solver_options)
        options["preconditioner"] = precond
        
        # システムを解く
        sol, iterations = solver.solve(A, b, options)
        
        # 分析用に反復回数を保存
        self.last_iterations = iterations
        
        return sol

    def _apply_scaling(self, A, b):
        """
        システム行列と右辺ベクトルにスケーリングを適用
        
        Args:
            A: システム行列
            b: 右辺ベクトル
            
        Returns:
            tuple: (scaled_A, scaled_b, scaling_info, scaler)
        """
        # スケーリングが要求されている場合に適用
        scaling_info = None
        scaler = None
        
        if self.scaling_method is not None:
            from scaling import plugin_manager
            scaler = plugin_manager.get_plugin(self.scaling_method)
            if scaler:
                print(f"スケーリング手法を適用: {scaler.name} - {scaler.description}")
                A_scaled, b_scaled, scaling_info = scaler.scale(A, b)
                return A_scaled, b_scaled, scaling_info, scaler
        
        return A, b, None, None

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
        print(f"  非ゼロ要素: {sparsity_info['non_zeros']}")
        print(f"  疎性率: {sparsity_info['sparsity']:.6f}")
        print(f"  メモリ使用量(密行列): {sparsity_info['memory_dense_MB']:.2f} MB")
        print(f"  メモリ使用量(疎行列): {sparsity_info['memory_sparse_MB']:.2f} MB")
        print(f"  メモリ削減率: {sparsity_info['memory_dense_MB'] / sparsity_info['memory_sparse_MB']:.1f}倍")
        
        return sparsity_info

    @abstractmethod
    def solve(self, analyze_before_solve=True):
        """
        システムを解く
        
        Args:
            analyze_before_solve: 解く前に行列を分析するかどうか
            
        Returns:
            解コンポーネント
        """
        pass


# 1Dソルバー実装
class CCDSolver1D(BaseCCDSolver):
    """1次元コンパクト差分法ソルバー"""

    def __init__(self, system, grid):
        """
        1Dソルバーを初期化
        
        Args:
            system: 方程式システム
            grid: 1D グリッドオブジェクト
        """
        super().__init__(system, grid)
        if grid.is_2d:
            raise ValueError("1Dソルバーは2Dグリッドでは使用できません")

    def _update_rhs(self, b, f_values=None, 
                    left_dirichlet=None, right_dirichlet=None,
                    left_neumann=None, right_neumann=None,
                    enable_dirichlet=True, enable_neumann=True):
        """
        境界値で右辺を更新
        
        Args:
            b: 右辺ベクトル
            f_values: ソース項の値
            left_dirichlet, right_dirichlet: ディリクレ値
            left_neumann, right_neumann: ノイマン値
            enable_dirichlet, enable_neumann: 有効フラグ
            
        Returns:
            更新された右辺ベクトル
        """
        n = self.grid.n_points
        
        # ソース項の値を設定
        if f_values is not None:
            for i in range(n):
                b[i * 4] = f_values[i] if i < len(f_values) else 0.0
        
        # 有効フラグに基づいて境界条件値を設定
        if enable_dirichlet:
            if left_dirichlet is not None:
                b[1] = left_dirichlet  # 左境界ディリクレ
            if right_dirichlet is not None:
                b[(n-1) * 4 + 1] = right_dirichlet  # 右境界ディリクレ
        
        if enable_neumann:
            if left_neumann is not None:
                b[2] = left_neumann  # 左境界ノイマン
            if right_neumann is not None:
                b[(n-1) * 4 + 2] = right_neumann  # 右境界ノイマン
                
        return b

    def solve(self, analyze_before_solve=True, f_values=None, 
              left_dirichlet=None, right_dirichlet=None,
              left_neumann=None, right_neumann=None,
              enable_dirichlet=True, enable_neumann=True):
        """
        1Dシステムを解く
        
        Args:
            analyze_before_solve: 解く前に行列を分析するかどうか
            f_values: 支配方程式の右辺値 (形状: (n,))
            left_dirichlet: 左境界ディリクレ値
            right_dirichlet: 右境界ディリクレ値
            left_neumann: 左境界ノイマン値
            right_neumann: 右境界ノイマン値
            enable_dirichlet: ディリクレ境界条件を有効にするかどうか
            enable_neumann: ノイマン境界条件を有効にするかどうか
            
        Returns:
            (psi, psi_prime, psi_second, psi_third) タプル
        """
        A, b = self.system.build_matrix_system()

        if analyze_before_solve:
            self.analyze_system()
            
        # 右辺値を更新
        b = self._update_rhs(
            b, f_values, 
            left_dirichlet, right_dirichlet,
            left_neumann, right_neumann,
            enable_dirichlet, enable_neumann
        )
            
        # 要求されている場合はスケーリングを適用
        A_scaled, b_scaled, scaling_info, scaler = self._apply_scaling(A, b)

        # システムを解く
        sol = self._solve_with_strategy(A_scaled, b_scaled)
            
        # スケーリングが適用された場合は解をアンスケール
        if scaling_info is not None and scaler is not None:
            sol = scaler.unscale(sol, scaling_info)

        # 1D解を処理
        return self._process_solution(sol)

    def _process_solution(self, sol):
        """1D解ベクトルを処理"""
        n = self.grid.n_points
        psi = sol[0::4][:n]
        psi_prime = sol[1::4][:n]
        psi_second = sol[2::4][:n]
        psi_third = sol[3::4][:n]

        return psi, psi_prime, psi_second, psi_third


# 2Dソルバー実装
class CCDSolver2D(BaseCCDSolver):
    """2次元コンパクト差分法ソルバー"""

    def __init__(self, system, grid):
        """
        2Dソルバーを初期化
        
        Args:
            system: 方程式システム
            grid: 2D グリッドオブジェクト
        """
        super().__init__(system, grid)
        if not grid.is_2d:
            raise ValueError("2Dソルバーは1Dグリッドでは使用できません")

    def _update_rhs(self, b, f_values=None,
                   left_dirichlet=None, right_dirichlet=None, 
                   bottom_dirichlet=None, top_dirichlet=None,
                   left_neumann=None, right_neumann=None,
                   bottom_neumann=None, top_neumann=None,
                   enable_dirichlet=True, enable_neumann=True):
        """
        2Dシステムの境界値で右辺を更新
        
        Args:
            b: 右辺ベクトル
            f_values: ソース項の値
            left_dirichlet, right_dirichlet, bottom_dirichlet, top_dirichlet: ディリクレ値
            left_neumann, right_neumann, bottom_neumann, top_neumann: ノイマン値
            enable_dirichlet, enable_neumann: 有効フラグ
            
        Returns:
            更新された右辺ベクトル
        """
        nx, ny = self.grid.nx_points, self.grid.ny_points
        n_unknowns = 7  # ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy
        
        # ソース項の値を設定
        if f_values is not None:
            for j in range(ny):
                for i in range(nx):
                    idx = (j * nx + i) * n_unknowns
                    if isinstance(f_values, (list, cp.ndarray)) and i < len(f_values) and j < len(f_values[i]):
                        b[idx] = f_values[i][j]
                    elif isinstance(f_values, (int, float)):
                        b[idx] = f_values
        
        # 有効フラグに基づいて境界条件値を設定
        if enable_dirichlet:
            # ディリクレ境界条件
            self._set_boundary_values(b, left_dirichlet, nx, ny, 0, 1, lambda j: (j * nx + 0) * n_unknowns + 1)
            self._set_boundary_values(b, right_dirichlet, nx, ny, 0, 1, lambda j: (j * nx + (nx-1)) * n_unknowns + 1)
            self._set_boundary_values(b, bottom_dirichlet, nx, ny, 1, 0, lambda i: (0 * nx + i) * n_unknowns + 4)
            self._set_boundary_values(b, top_dirichlet, nx, ny, 1, 0, lambda i: ((ny-1) * nx + i) * n_unknowns + 4)
        
        if enable_neumann:
            # ノイマン境界条件
            self._set_boundary_values(b, left_neumann, nx, ny, 0, 1, lambda j: (j * nx + 0) * n_unknowns + 2)
            self._set_boundary_values(b, right_neumann, nx, ny, 0, 1, lambda j: (j * nx + (nx-1)) * n_unknowns + 2)
            self._set_boundary_values(b, bottom_neumann, nx, ny, 1, 0, lambda i: (0 * nx + i) * n_unknowns + 5)
            self._set_boundary_values(b, top_neumann, nx, ny, 1, 0, lambda i: ((ny-1) * nx + i) * n_unknowns + 5)
                
        return b

    def _set_boundary_values(self, b, values, nx, ny, dim_i, dim_j, idx_func):
        """
        特定の境界に沿って境界値を設定
        
        Args:
            b: 右辺ベクトル
            values: 境界値
            nx, ny: グリッド次元
            dim_i, dim_j: 次元インデックス（0または1）
            idx_func: インデックス計算関数
            
        Returns:
            None (bをその場で変更)
        """
        if values is None:
            return
            
        max_i = nx if dim_i == 1 else 1
        max_j = ny if dim_j == 1 else 1
        
        for idx in range(max_i * dim_i + max_j * dim_j):
            index = idx_func(idx)
            
            if isinstance(values, (list, cp.ndarray)) and idx < len(values):
                b[index] = values[idx]
            elif isinstance(values, (int, float)):
                b[index] = values

    def solve(self, analyze_before_solve=True, f_values=None,
              left_dirichlet=None, right_dirichlet=None, 
              bottom_dirichlet=None, top_dirichlet=None,
              left_neumann=None, right_neumann=None,
              bottom_neumann=None, top_neumann=None,
              enable_dirichlet=True, enable_neumann=True):
        """
        2Dシステムを解く
        
        Args:
            analyze_before_solve: 解く前に行列を分析するかどうか
            f_values: 支配方程式の右辺値 (形状: (nx, ny))
            left_dirichlet, right_dirichlet, bottom_dirichlet, top_dirichlet: 境界値
            left_neumann, right_neumann, bottom_neumann, top_neumann: 境界導関数
            enable_dirichlet, enable_neumann: 境界条件を有効にするかどうか
            
        Returns:
            (psi, psi_x, psi_xx, psi_xxx, psi_y, psi_yy, psi_yyy) タプル
        """
        A, b = self.system.build_matrix_system()

        if analyze_before_solve:
            self.analyze_system()
            
        # 右辺値を更新
        b = self._update_rhs(
            b, f_values,
            left_dirichlet, right_dirichlet, bottom_dirichlet, top_dirichlet,
            left_neumann, right_neumann, bottom_neumann, top_neumann,
            enable_dirichlet, enable_neumann
        )
                
        # 要求されている場合はスケーリングを適用
        A_scaled, b_scaled, scaling_info, scaler = self._apply_scaling(A, b)

        # システムを解く
        sol = self._solve_with_strategy(A_scaled, b_scaled)
            
        # スケーリングが適用された場合は解をアンスケール
        if scaling_info is not None and scaler is not None:
            sol = scaler.unscale(sol, scaling_info)

        # 2D解を処理
        return self._process_solution(sol)

    def _process_solution(self, sol):
        """2D解ベクトルを処理"""
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