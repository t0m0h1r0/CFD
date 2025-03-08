# solver.py
import cupy as cp
import cupyx.scipy.sparse as sp
import cupyx.scipy.sparse.linalg as splinalg
from typing import Tuple, Optional, Dict, Union
from equation_system import EquationSystem
from grid import Grid
from matrix_scaling import MatrixRehuScaling


class CCDSolver:
    """CCDソルバークラス (スパース行列最適化版)"""

    def __init__(self, system: EquationSystem, grid: Grid):
        """
        初期化

        Args:
            system: 方程式システム
            grid: 計算格子
        """
        self.system = system
        self.grid = grid
        self.scaler = None  # スケーリングオブジェクト
        self.solver_method = "direct"  # デフォルトはdirect (スパース直接法)
        self.solver_options = {}
        self.sparsity_info = None  # 疎性分析結果
        
        # CuPyで利用可能なソルバー
        self.available_solvers = {
            "direct": self._solve_direct,
            "gmres": self._solve_gmres,
            "cg": self._solve_cg,
            "cgs": self._solve_cgs
        }

    def set_rehu_scaling(
        self,
        rehu_number: float = 1.0,
        characteristic_velocity: float = 1.0,
        reference_length: float = 1.0,
    ):
        """
        Reynolds-Hugoniotスケーリングを設定

        Args:
            rehu_number: Reynolds-Hugoniot数
            characteristic_velocity: 特性速度
            reference_length: 代表長さ
        """
        self.scaler = MatrixRehuScaling(
            characteristic_velocity=characteristic_velocity,
            reference_length=reference_length,
            rehu_number=rehu_number,
        )
        print(
            f"Matrix-based Reynolds-Hugoniot scaling set with Rehu number: {rehu_number}"
        )

    def set_solver(self, method: str = "direct", options: Dict = None):
        """
        使用するソルバーを設定

        Args:
            method: ソルバー手法 ('direct', 'gmres', 'cg', 'cgs')
            options: ソルバー固有のオプション
        """
        # CuPyで利用可能なソルバーに対応
        if method not in self.available_solvers:
            # 利用不可能なソルバーが指定された場合
            fallback_solver = "gmres"
            print(f"警告: ソルバー '{method}' はCuPyでは利用できません。")
            print(f"代わりに '{fallback_solver}' を使用します。")
            print(f"CuPyで利用可能なソルバー: {list(self.available_solvers.keys())}")
            method = fallback_solver
            
        self.solver_method = method
        self.solver_options = options or {}
        print(f"Solver set to: {method}")

    def analyze_system(self) -> Dict:
        """
        行列システムの疎性を分析

        Returns:
            Dict: 疎性の統計情報
        """
        self.sparsity_info = self.system.analyze_sparsity()
        print("\n行列システム分析:")
        print(f"サイズ: {self.sparsity_info['size']}×{self.sparsity_info['size']}")
        print(f"非ゼロ要素数: {self.sparsity_info['nonzeros']}")
        print(f"密度: {self.sparsity_info['density']:.6f}")
        print(f"行あたり平均非ゼロ要素数: {self.sparsity_info['avg_nonzeros_per_row']:.2f}")
        print(f"推定バンド幅: {self.sparsity_info['estimated_bandwidth']}")
        print(f"密行列必要メモリ: {self.sparsity_info['memory_dense_MB']:.2f} MB")
        print(f"疎行列必要メモリ: {self.sparsity_info['memory_sparse_MB']:.2f} MB")
        print(f"メモリ削減率: {(1 - self.sparsity_info['memory_sparse_MB'] / self.sparsity_info['memory_dense_MB']) * 100:.2f}%")
        
        return self.sparsity_info

    def _scale_sparse_matrix(
        self, A: sp.csr_matrix, b: cp.ndarray, n_points: int
    ) -> Tuple[sp.csr_matrix, cp.ndarray]:
        """
        スパース行列システムにスケーリングを適用

        Args:
            A: 元のスパース係数行列
            b: 元の右辺ベクトル
            n_points: グリッド点の数

        Returns:
            スケーリングされた行列システム (A_scaled, b_scaled)
        """
        # スケーリング行列の作成
        S = self.scaler._create_scaling_matrix(n_points)
        S_inv = cp.linalg.inv(S)  # 逆行列

        # スパース行列のスケーリング: S⁻¹ A S
        # まず S⁻¹ A を計算
        S_inv_A = sp.dia_matrix((S_inv.diagonal(), [0]), shape=S_inv.shape) @ A
        # 次に (S⁻¹ A) S を計算
        A_scaled = S_inv_A @ sp.dia_matrix((S.diagonal(), [0]), shape=S.shape)
        
        # 右辺ベクトルのスケーリング: S⁻¹ b
        b_scaled = S_inv @ b
        
        return A_scaled, b_scaled

    def _create_preconditioner(self, A: sp.csr_matrix):
        """
        前処理器を作成

        Args:
            A: 係数行列

        Returns:
            前処理器（または None）
        """
        if not self.solver_options.get("use_preconditioner", True):
            print("  前処理は使用しません")
            return None
            
        try:
            print("  前処理器を作成中...")
            # 対角成分を使った簡単な前処理
            diag = A.diagonal()
            
            # 対角成分が0に近い場合は小さな値で置き換え
            diag = cp.where(cp.abs(diag) < 1e-14, 1.0, diag)
            
            # 対角スケーリング前処理
            D_inv = sp.diags(1.0 / diag)
            
            # 前処理器として使用するLinearOperator
            precond = splinalg.LinearOperator(
                A.shape, 
                matvec=lambda x: D_inv @ x
            )
            print("  対角スケーリング前処理を適用")
            return precond
        except Exception as e:
            print(f"  前処理器の作成に失敗: {e}")
            print("  前処理なしで続行します")
            return None

    def _solve_direct(self, A: sp.csr_matrix, b: cp.ndarray) -> cp.ndarray:
        """直接法を使用してスパース行列システムを解く"""
        print("直接法でスパース行列システムを解いています...")
        return splinalg.spsolve(A, b)
    
    def _solve_gmres(self, A: sp.csr_matrix, b: cp.ndarray) -> cp.ndarray:
        """GMRESを使用してスパース行列システムを解く"""
        tol = self.solver_options.get("tol", 1e-10)
        maxiter = self.solver_options.get("maxiter", 1000)
        restart = self.solver_options.get("restart", 100)
        
        print(f"GMRESでスパース行列システムを解いています...")
        print(f"  収束許容誤差: {tol}, 最大反復回数: {maxiter}, リスタート値: {restart}")
        
        # 前処理器の作成
        precond = self._create_preconditioner(A)
        
        try:
            x, info = splinalg.gmres(
                A, b, 
                tol=tol, 
                maxiter=maxiter, 
                M=precond,
                restart=restart
            )
            
            if info == 0:
                print(f"  GMRESが収束しました")
                return x
            else:
                print(f"  警告: GMRESが収束しませんでした (info={info})")
                print("  直接法にフォールバックします...")
                return splinalg.spsolve(A, b)
                
        except Exception as e:
            print(f"  GMRESでエラーが発生しました: {e}")
            print("  直接法にフォールバックします...")
            return splinalg.spsolve(A, b)
    
    def _solve_cg(self, A: sp.csr_matrix, b: cp.ndarray) -> cp.ndarray:
        """CGを使用してスパース行列システムを解く（対称正定値行列用）"""
        tol = self.solver_options.get("tol", 1e-10)
        maxiter = self.solver_options.get("maxiter", 1000)
        
        print(f"CGでスパース行列システムを解いています...")
        print(f"  収束許容誤差: {tol}, 最大反復回数: {maxiter}")
        print("  注意: CG法は対称正定値行列専用です")
        
        # 前処理器の作成
        precond = self._create_preconditioner(A)
        
        try:
            x, info = splinalg.cg(
                A, b, 
                tol=tol, 
                maxiter=maxiter, 
                M=precond
            )
            
            if info == 0:
                print(f"  CGが収束しました")
                return x
            else:
                print(f"  警告: CGが収束しませんでした (info={info})")
                print("  直接法にフォールバックします...")
                return splinalg.spsolve(A, b)
                
        except Exception as e:
            print(f"  CGでエラーが発生しました: {e}")
            print("  行列が対称正定値でない可能性があります")
            print("  直接法にフォールバックします...")
            return splinalg.spsolve(A, b)
            
    def _solve_cgs(self, A: sp.csr_matrix, b: cp.ndarray) -> cp.ndarray:
        """CGSを使用してスパース行列システムを解く"""
        tol = self.solver_options.get("tol", 1e-10)
        maxiter = self.solver_options.get("maxiter", 1000)
        
        print(f"CGSでスパース行列システムを解いています...")
        print(f"  収束許容誤差: {tol}, 最大反復回数: {maxiter}")
        
        # 前処理器の作成
        precond = self._create_preconditioner(A)
        
        try:
            x, info = splinalg.cgs(
                A, b, 
                tol=tol, 
                maxiter=maxiter, 
                M=precond
            )
            
            if info == 0:
                print(f"  CGSが収束しました")
                return x
            else:
                print(f"  警告: CGSが収束しませんでした (info={info})")
                print("  直接法にフォールバックします...")
                return splinalg.spsolve(A, b)
                
        except Exception as e:
            print(f"  CGSでエラーが発生しました: {e}")
            print("  直接法にフォールバックします...")
            return splinalg.spsolve(A, b)

    def solve(
        self, analyze_before_solve: bool = True
    ) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
        """
        方程式を解く

        Args:
            analyze_before_solve: 解く前に行列システムを分析するかどうか

        Returns:
            (psi, psi', psi'', psi''')の組
        """
        # 行列システムを構築
        print("スパース行列システムを構築中...")
        A, b = self.system.build_matrix_system()

        if analyze_before_solve:
            self.analyze_system()

        # スケーリングが設定されていれば適用
        if self.scaler is not None:
            print("行列スケーリングを適用中...")
            A, b = self._scale_sparse_matrix(A, b, self.grid.n_points)

        # システムを解く
        solver_func = self.available_solvers.get(self.solver_method, self._solve_direct)
        sol = solver_func(A, b)

        # スケーリングが適用されていた場合、結果を元のスケールに戻す
        if self.scaler is not None:
            print("解のスケーリングを元に戻しています...")
            sol = self.scaler.unscale_solution(sol, self.grid.n_points)

        # 解から各成分を抽出
        n = self.grid.n_points
        psi = sol[0::4][:n]
        psi_prime = sol[1::4][:n]
        psi_second = sol[2::4][:n]
        psi_third = sol[3::4][:n]

        return psi, psi_prime, psi_second, psi_third