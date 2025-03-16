"""
高精度コンパクト差分法 (CCD) のための線形方程式系ソルバーモジュール

このモジュールは1次元・2次元のポアソン方程式および高階微分方程式を解くための
ソルバーを提供します。equation_system.pyと連携して方程式系を構築し解きます。
"""

import os
import time
import numpy as np
import scipy.sparse.linalg as splinalg_cpu
from abc import ABC, abstractmethod

from equation_system import EquationSystem
from scaling import plugin_manager

class LinearSystemSolver:
    """線形方程式系 Ax=b を解くためのクラス"""
    
    def __init__(self, method="direct", options=None, scaling_method=None):
        """
        線形方程式系ソルバーを初期化
        
        Args:
            method: 解法メソッド ("direct", "gmres", "cg", "cgs", "minres", "lsqr")
            options: ソルバーオプション辞書
            scaling_method: スケーリング手法名
        """
        self.method = method
        self.options = options or {}
        self.scaling_method = scaling_method
        self.last_iterations = None
        self.monitor_convergence = self.options.get("monitor_convergence", False)
        
        # GPU上の行列とスケーリング情報を保持する変数
        self.A_gpu = None
        self.A_scaled_gpu = None
        self.scaling_info = None
        self.scaler = None
        self.original_A = None  # 比較用に元の行列への参照を保持
    
    def __del__(self):
        """デストラクタ：インスタンス削除時にGPUメモリを解放"""
        self.clear_gpu_memory()
    
    def clear_gpu_memory(self):
        """GPU上の行列メモリを解放"""
        self.A_gpu = None
        self.A_scaled_gpu = None
        self.scaling_info = None
        self.scaler = None
        if hasattr(self, 'original_A'):
            self.original_A = None
    
    def _is_new_matrix(self, A):
        """現在のGPU行列と異なる行列かどうかを判定"""
        # GPU行列がまだないか、形状が違う場合は新しい行列
        if self.A_gpu is None or self.original_A is None or self.A_gpu.shape != A.shape:
            return True
        
        # 形状が同じ場合は、保存している参照と比較
        if self.original_A is A:
            return False
        
        # オブジェクトが異なれば新しい行列と判断
        return True
    
    def _solve_with_gpu(self, A, b, callback=None):
        """GPU (CuPy) を使用して線形方程式系を解く"""
        import cupy as cp
        import cupyx.scipy.sparse.linalg as splinalg
        
        if self.method == "direct":
            x = splinalg.spsolve(A, b)
            iterations = None
        elif self.method == "gmres":
            tol = self.options.get("tol", 1e-10)
            maxiter = self.options.get("maxiter", 1000)
            restart = self.options.get("restart", 100)
            x0 = cp.ones_like(b)
            x, iterations = splinalg.gmres(A, b, x0=x0, tol=tol, maxiter=maxiter, 
                                    restart=restart, callback=callback)
        elif self.method in ["cg", "cgs", "minres"]:
            tol = self.options.get("tol", 1e-10)
            maxiter = self.options.get("maxiter", 1000)
            solver_func = getattr(splinalg, self.method)
            x0 = cp.ones_like(b)
            x, iterations = solver_func(A, b, x0=x0, tol=tol, maxiter=maxiter, callback=callback)
        elif self.method in ["lsqr", "lsmr"]:
            solver_func = getattr(splinalg, self.method)
            maxiter = self.options.get("maxiter", 1000)
            if self.method == "lsmr":
                x0 = cp.ones_like(b)
                x = solver_func(A, b, x0=x0, maxiter=maxiter)[0]
            else:
                x = solver_func(A, b)[0]
            iterations = None
        else:
            print(f"未知の解法 {self.method}。直接解法を使用します。")
            x = splinalg.spsolve(A, b)
            iterations = None
        
        return x, iterations

    def _solve_with_cpu(self, A, b):
        """CPU (SciPy) を使用して線形方程式系を解く"""
        # 既にCPU (NumPy/SciPy) の形式と想定
        if self.method == "direct" or self.method not in ["gmres", "cg", "cgs", "minres", "lsqr", "lsmr"]:
            x = splinalg_cpu.spsolve(A, b)
            iterations = None
        elif self.method == "gmres":
            tol = self.options.get("tol", 1e-10)
            maxiter = self.options.get("maxiter", 1000)
            restart = self.options.get("restart", 100)
            x0 = np.ones_like(b)
            x, iterations = splinalg_cpu.gmres(A, b, x0=x0, tol=tol, maxiter=maxiter, restart=restart)
        elif self.method in ["cg", "cgs", "minres"]:
            tol = self.options.get("tol", 1e-10)
            maxiter = self.options.get("maxiter", 1000)
            solver_func = getattr(splinalg_cpu, self.method)
            x0 = np.ones_like(b)
            x, iterations = solver_func(A, b, x0=x0, tol=tol, maxiter=maxiter)
        elif self.method in ["lsqr", "lsmr"]:
            maxiter = self.options.get("maxiter", 1000)
            solver_func = getattr(splinalg_cpu, self.method)
            x = solver_func(A, b)[0]
            iterations = None
        
        return x, iterations

    def solve(self, A, b):
        """
        線形方程式系を解く
        
        Args:
            A: システム行列 (CPU/SciPy形式)
            b: 右辺ベクトル (CPU/NumPy形式)
            
        Returns:
            解ベクトル x (CPU/NumPy形式)
        """
        # スケーリング実行前のタイミング計測
        start_time = time.time()
        residuals = []
        
        # force_cpuオプションが設定されているかチェック
        force_cpu = self.options.get("force_cpu", False)
        
        if force_cpu:
            # CPU処理を強制的に実行
            print("CPU処理を強制的に実行します (スケーリング機能は無効化されます)...")
            
            # GPUメモリを解放
            self.clear_gpu_memory()
            
            # CPU専用パス
            try:
                x, iterations = self._solve_with_cpu(A, b)
            except Exception as e:
                print(f"CPU解法でエラー: {e}")
                raise
        else:
            # GPU優先の処理
            try:
                import cupy as cp
                import cupyx.scipy.sparse as sp
                
                # 行列が変更されたかチェック
                is_new_matrix = self._is_new_matrix(A)
                
                # 新しい行列の場合はGPUに転送
                if is_new_matrix:
                    print("行列を GPU に転送しています...")
                    self.clear_gpu_memory()
                    self.A_gpu = sp.csr_matrix(A)
                    self.original_A = A  # 元の行列への参照を保存
                    
                    # 新しい行列なのでスケーリング情報もリセット
                    self.A_scaled_gpu = None
                    self.scaling_info = None
                    self.scaler = None
                else:
                    print("GPU上の既存行列を再利用します")
                
                # bをGPUに転送
                b_gpu = cp.array(b)
                
                # スケーリングを適用または再利用
                if is_new_matrix or self.A_scaled_gpu is None:
                    # 新しい行列または初めての呼び出し時
                    self.A_scaled_gpu, b_scaled, self.scaling_info, self.scaler = self._apply_scaling(self.A_gpu, b_gpu)
                else:
                    # 既存のスケーリング済み行列を再利用して、bだけを新たにスケーリング
                    if self.scaling_method is not None and self.scaler is not None:
                        # スケーリング情報からbをスケーリング
                        if hasattr(self.scaler, 'scale_b_only'):
                            # scale_b_onlyメソッドがあればそれを使用
                            b_scaled = self.scaler.scale_b_only(b_gpu, self.scaling_info)
                        else:
                            # なければ完全なスケーリングを実行しA_scaled_gpuは無視
                            _, b_scaled, _, _ = self.scaler.scale(self.A_gpu, b_gpu)
                    else:
                        b_scaled = b_gpu
                
                # モニタリングコールバック
                if self.monitor_convergence:
                    def callback(xk):
                        residual = cp.linalg.norm(b_scaled - self.A_scaled_gpu @ xk) / cp.linalg.norm(b_scaled)
                        residuals.append(float(residual))
                        if len(residuals) % 10 == 0:
                            print(f"  反復 {len(residuals)}: 残差 = {residual:.6e}")
                else:
                    callback = None
                    
                # GPU で計算を実行
                x_gpu, iterations = self._solve_with_gpu(self.A_scaled_gpu, b_scaled, callback)
                
                # スケーリングを戻す
                if self.scaling_info is not None and self.scaler is not None:
                    x_gpu = self.scaler.unscale(x_gpu, self.scaling_info)
                    
                # 結果をCPUに転送
                x = x_gpu.get()
                
            except Exception as e:
                print(f"GPU処理でエラー: {e}")
                print("CPU (SciPy) に切り替えて計算を実行します...")
                # エラー時はGPUメモリを解放して再試行を容易にする
                self.clear_gpu_memory()
                x, iterations = self._solve_with_cpu(A, b)
        
        elapsed = time.time() - start_time
        self.last_iterations = iterations
        
        print(f"解法実行: {self.method}, 経過時間: {elapsed:.4f}秒")
        if iterations:
            print(f"反復回数: {iterations}")
            
        if self.monitor_convergence and residuals:
            self._visualize_convergence(residuals)
        
        return x
    
    def _apply_scaling(self, A, b):
        """
        行列と右辺ベクトルにスケーリングを適用
        
        Args:
            A: システム行列 (GPU)
            b: 右辺ベクトル (GPU)
            
        Returns:
            tuple: (scaled_A, scaled_b, scaling_info, scaler)
        """
        if self.scaling_method is not None:
            scaler = plugin_manager.get_plugin(self.scaling_method)
            if scaler:
                print(f"スケーリング: {scaler.name}")
                A_scaled, b_scaled, scaling_info = scaler.scale(A, b)
                return A_scaled, b_scaled, scaling_info, scaler
        
        return A, b, None, None
    
    def _visualize_convergence(self, residuals):
        """収束履歴をグラフ化（必要な場合）"""
        if not residuals:
            return
            
        # 出力ディレクトリを確保
        output_dir = self.options.get("output_dir", "results")
        os.makedirs(output_dir, exist_ok=True)
        
        # 残差の推移グラフ
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.semilogy(range(1, len(residuals)+1), residuals, 'b-')
        plt.grid(True, which="both", ls="--")
        plt.xlabel('反復回数')
        plt.ylabel('残差 (対数スケール)')
        plt.title(f'{self.method.upper()} ソルバーの収束履歴')
        
        # 保存
        prefix = self.options.get("prefix", "")
        filename = os.path.join(output_dir, f"{prefix}_convergence_{self.method}.png")
        plt.savefig(filename, dpi=150)
        plt.close()


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
        
        # システムを初期化し、行列Aを構築 (CPU処理)
        self.system = EquationSystem(grid)
        self.enable_dirichlet, self.enable_neumann = equation_set.setup_equations(self.system, grid)
        self.matrix_A = self.system.build_matrix_system()
    
    def set_solver(self, method="direct", options=None, scaling_method=None):
        """
        ソルバーの設定
        
        Args:
            method: 解法メソッド
            options: ソルバーオプション辞書
            scaling_method: スケーリング手法名
        """
        self.linear_solver = LinearSystemSolver(method, options, scaling_method)
    
    @property
    def scaling_method(self):
        """スケーリング手法を取得"""
        return self.linear_solver.scaling_method
    
    @scaling_method.setter
    def scaling_method(self, value):
        """スケーリング手法を設定"""
        self.linear_solver.scaling_method = value
    
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
        A = self.matrix_A
        total_size = A.shape[0]
        nnz = A.nnz
        sparsity = 1.0 - (nnz / (total_size * total_size))
        memory_dense_MB = (total_size * total_size * 8) / (1024 * 1024)
        memory_sparse_MB = (nnz * 12) / (1024 * 1024)
        
        print("\n行列構造分析:")
        print(f"  行列サイズ: {total_size} x {total_size}")
        print(f"  非ゼロ要素数: {nnz}")
        print(f"  疎性率: {sparsity:.6f}")
        print(f"  メモリ使用量(密行列): {memory_dense_MB:.2f} MB")
        print(f"  メモリ使用量(疎行列): {memory_sparse_MB:.2f} MB")
        
        return {
            "matrix_size": total_size,
            "non_zeros": nnz,
            "sparsity": sparsity,
            "memory_dense_MB": memory_dense_MB,
            "memory_sparse_MB": memory_sparse_MB
        }

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
            
        # 右辺ベクトルbを構築 (CPU処理)
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
    
    def _to_numpy(self, arr):
        """CuPy配列をNumPy配列に変換する (必要な場合のみ)"""
        if hasattr(arr, 'get'):
            return arr.get()
        return arr


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
        1D右辺ベクトルを構築 (CPU処理)
        
        Args:
            f_values: ソース項の値（全格子点の配列）
            left_dirichlet, right_dirichlet: ディリクレ境界値
            left_neumann, right_neumann: ノイマン境界値
            
        Returns:
            右辺ベクトル (NumPy配列)
        """
        n = self.grid.n_points
        var_per_point = 4  # [ψ, ψ', ψ'', ψ''']
        b = np.zeros(n * var_per_point)
        
        # 入力値を NumPy に変換（必要な場合）
        if f_values is not None:
            f_values = self._to_numpy(f_values)
        
        # 境界条件に関する情報を出力
        boundary_info = []
        if self.enable_dirichlet:
            boundary_info.append(f"ディリクレ境界条件: 左={left_dirichlet}, 右={right_dirichlet}")
        if self.enable_neumann:
            boundary_info.append(f"ノイマン境界条件: 左={left_neumann}, 右={right_neumann}")
        if boundary_info:
            print("[1Dソルバー] " + "; ".join(boundary_info))
        
        # 各格子点に対して処理
        for i in range(n):
            base_idx = i * var_per_point
            location = self.system._get_point_location(i)
            location_equations = self.system.equations[location]
            
            # 方程式を種類別に分類
            eq_by_type = {"governing": None, "dirichlet": None, "neumann": None, "auxiliary": []}
            for eq in location_equations:
                eq_type = self.system._identify_equation_type(eq, i)
                if eq_type == "auxiliary":
                    eq_by_type["auxiliary"].append(eq)
                elif eq_type:  # Noneでない場合
                    eq_by_type[eq_type] = eq
            
            # ソース項の処理（支配方程式に対応する行）
            if eq_by_type["governing"] and f_values is not None:
                b[base_idx] = f_values[i]
            
            # 境界条件の処理
            if location == 'left':
                if self.enable_dirichlet and left_dirichlet is not None:
                    dirichlet_row = self._find_dirichlet_row(base_idx, location_equations, i)
                    if dirichlet_row is not None:
                        b[dirichlet_row] = left_dirichlet
                        
                if self.enable_neumann and left_neumann is not None:
                    neumann_row = self._find_neumann_row(base_idx, location_equations, i)
                    if neumann_row is not None:
                        b[neumann_row] = left_neumann
                        
            elif location == 'right':
                if self.enable_dirichlet and right_dirichlet is not None:
                    dirichlet_row = self._find_dirichlet_row(base_idx, location_equations, i)
                    if dirichlet_row is not None:
                        b[dirichlet_row] = right_dirichlet
                        
                if self.enable_neumann and right_neumann is not None:
                    neumann_row = self._find_neumann_row(base_idx, location_equations, i)
                    if neumann_row is not None:
                        b[neumann_row] = right_neumann
        
        return b
    
    def _find_dirichlet_row(self, base_idx, equations, i):
        """ディリクレ境界条件に対応する行インデックスを見つける"""
        for row_offset, eq in enumerate(equations):
            eq_type = self.system._identify_equation_type(eq, i)
            if eq_type == "dirichlet":
                return base_idx + row_offset
        return None
    
    def _find_neumann_row(self, base_idx, equations, i):
        """ノイマン境界条件に対応する行インデックスを見つける"""
        for row_offset, eq in enumerate(equations):
            eq_type = self.system._identify_equation_type(eq, i)
            if eq_type == "neumann":
                return base_idx + row_offset
        return None

    def _extract_solution(self, sol):
        """解ベクトルから各成分を抽出 (CPU処理)"""
        n = self.grid.n_points
        psi = sol[0::4][:n]
        psi_prime = sol[1::4][:n]
        psi_second = sol[2::4][:n]
        psi_third = sol[3::4][:n]

        return psi, psi_prime, psi_second, psi_third


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
        2D右辺ベクトルを構築 (CPU処理)
        
        Args:
            f_values: ソース項の値 (nx×ny配列)
            left_dirichlet, right_dirichlet, bottom_dirichlet, top_dirichlet: 境界値
            left_neumann, right_neumann, bottom_neumann, top_neumann: 境界導関数
            
        Returns:
            右辺ベクトル (NumPy配列)
        """
        nx, ny = self.grid.nx_points, self.grid.ny_points
        var_per_point = 7  # [ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy]
        b = np.zeros(nx * ny * var_per_point)
        
        # 入力値を NumPy に変換（必要な場合）
        if f_values is not None:
            f_values = self._to_numpy(f_values)
            
        # その他の境界値も同様に変換（必要な場合）
        boundary_values = {
            'left_dirichlet': self._to_numpy(left_dirichlet) if left_dirichlet is not None else None,
            'right_dirichlet': self._to_numpy(right_dirichlet) if right_dirichlet is not None else None,
            'bottom_dirichlet': self._to_numpy(bottom_dirichlet) if bottom_dirichlet is not None else None,
            'top_dirichlet': self._to_numpy(top_dirichlet) if top_dirichlet is not None else None,
            'left_neumann': self._to_numpy(left_neumann) if left_neumann is not None else None,
            'right_neumann': self._to_numpy(right_neumann) if right_neumann is not None else None,
            'bottom_neumann': self._to_numpy(bottom_neumann) if bottom_neumann is not None else None,
            'top_neumann': self._to_numpy(top_neumann) if top_neumann is not None else None
        }
        
        # 境界条件の状態を出力
        self._print_boundary_info(**boundary_values)
        
        # 各格子点に対して処理
        for j in range(ny):
            for i in range(nx):
                base_idx = (j * nx + i) * var_per_point
                location = self.system._get_point_location(i, j)
                location_equations = self.system.equations[location]
                
                # 方程式を種類別に分類
                eq_by_type = {
                    "governing": None, 
                    "dirichlet": None, 
                    "neumann_x": None, 
                    "neumann_y": None, 
                    "auxiliary": []
                }
                
                for eq in location_equations:
                    eq_type = self.system._identify_equation_type(eq, i, j)
                    if eq_type == "auxiliary":
                        eq_by_type["auxiliary"].append(eq)
                    elif eq_type:  # Noneでない場合
                        eq_by_type[eq_type] = eq
                
                # 各行に割り当てる方程式を決定
                assignments = self.system._assign_equations_2d(eq_by_type, i, j)
                
                # ソース項の処理（支配方程式に対応する行）
                governing_row = self._find_governing_row(assignments)
                if governing_row is not None and f_values is not None:
                    b[base_idx + governing_row] = f_values[i, j]
                
                # 境界条件の処理
                self._apply_boundary_values(
                    b, base_idx, location, assignments,
                    **boundary_values,
                    i=i, j=j
                )
        
        return b
    
    def _print_boundary_info(self, left_dirichlet=None, right_dirichlet=None, 
                           bottom_dirichlet=None, top_dirichlet=None,
                           left_neumann=None, right_neumann=None, 
                           bottom_neumann=None, top_neumann=None):
        """境界条件の情報を出力"""
        if self.enable_dirichlet:
            print(f"[2Dソルバー] ディリクレ境界条件: "
                  f"左={left_dirichlet is not None}, 右={right_dirichlet is not None}, "
                  f"下={bottom_dirichlet is not None}, 上={top_dirichlet is not None}")
        if self.enable_neumann:
            print(f"[2Dソルバー] ノイマン境界条件: "
                  f"左={left_neumann is not None}, 右={right_neumann is not None}, "
                  f"下={bottom_neumann is not None}, 上={top_neumann is not None}")
    
    def _find_governing_row(self, assignments):
        """支配方程式が割り当てられた行を見つける"""
        from equation.poisson import PoissonEquation, PoissonEquation2D
        from equation.original import OriginalEquation, OriginalEquation2D
        
        for row, eq in enumerate(assignments):
            if isinstance(eq, (PoissonEquation, PoissonEquation2D, OriginalEquation, OriginalEquation2D)):
                return row
        return None
    
    def _apply_boundary_values(self, b, base_idx, location, assignments,
                            left_dirichlet=None, right_dirichlet=None, 
                            bottom_dirichlet=None, top_dirichlet=None,
                            left_neumann=None, right_neumann=None, 
                            bottom_neumann=None, top_neumann=None,
                            i=None, j=None):
        """適切な場所に境界値を設定"""
        # インポートは関数内で行い、依存関係をローカルに限定
        from equation.boundary import (
            DirichletBoundaryEquation2D, NeumannXBoundaryEquation2D, NeumannYBoundaryEquation2D
        )
        
        # 境界値の取得
        def get_boundary_value(value, idx):
            if isinstance(value, (list, np.ndarray)) and idx < len(value):
                return value[idx]
            return value
        
        # 各行に割り当てられた方程式をチェック
        for row, eq in enumerate(assignments):
            # ディリクレ条件
            if isinstance(eq, DirichletBoundaryEquation2D) and self.enable_dirichlet:
                if 'left' in location and left_dirichlet is not None:
                    b[base_idx + row] = get_boundary_value(left_dirichlet, j)
                elif 'right' in location and right_dirichlet is not None:
                    b[base_idx + row] = get_boundary_value(right_dirichlet, j)
                elif 'bottom' in location and bottom_dirichlet is not None:
                    b[base_idx + row] = get_boundary_value(bottom_dirichlet, i)
                elif 'top' in location and top_dirichlet is not None:
                    b[base_idx + row] = get_boundary_value(top_dirichlet, i)
            
            # X方向ノイマン条件
            elif isinstance(eq, NeumannXBoundaryEquation2D) and self.enable_neumann:
                if 'left' in location and left_neumann is not None:
                    b[base_idx + row] = get_boundary_value(left_neumann, j)
                elif 'right' in location and right_neumann is not None:
                    b[base_idx + row] = get_boundary_value(right_neumann, j)
            
            # Y方向ノイマン条件
            elif isinstance(eq, NeumannYBoundaryEquation2D) and self.enable_neumann:
                if 'bottom' in location and bottom_neumann is not None:
                    b[base_idx + row] = get_boundary_value(bottom_neumann, i)
                elif 'top' in location and top_neumann is not None:
                    b[base_idx + row] = get_boundary_value(top_neumann, i)

    def _extract_solution(self, sol):
        """解ベクトルから各成分を抽出 (CPU処理)"""
        nx, ny = self.grid.nx_points, self.grid.ny_points
        n_unknowns = 7  # ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy
        
        # 解配列を初期化 (NumPy配列)
        psi = np.zeros((nx, ny))
        psi_x = np.zeros((nx, ny))
        psi_xx = np.zeros((nx, ny))
        psi_xxx = np.zeros((nx, ny))
        psi_y = np.zeros((nx, ny))
        psi_yy = np.zeros((nx, ny))
        psi_yyy = np.zeros((nx, ny))
        
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