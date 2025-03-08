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
            method: ソルバー手法 ('direct', 'iterative', 'gmres', 'bicgstab', 'cg')
            options: ソルバー固有のオプション
        """
        valid_methods = ["direct", "iterative", "gmres", "bicgstab", "cg"]
        if method not in valid_methods:
            raise ValueError(f"ソルバー手法 '{method}' は無効です。有効な値: {valid_methods}")
            
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

    def _solve_direct(self, A: sp.csr_matrix, b: cp.ndarray) -> cp.ndarray:
        """直接法を使用してスパース行列システムを解く"""
        print("直接法でスパース行列システムを解いています...")
        return splinalg.spsolve(A, b)
    
    def _solve_iterative(self, A: sp.csr_matrix, b: cp.ndarray) -> cp.ndarray:
        """反復法を使用してスパース行列システムを解く"""
        method = self.solver_method if self.solver_method != "iterative" else "gmres"
        
        # ソルバーとオプションの設定
        tol = self.solver_options.get("tol", 1e-10)
        maxiter = self.solver_options.get("maxiter", 1000)
        
        print(f"反復法（{method}）でスパース行列システムを解いています...")
        print(f"  収束許容誤差: {tol}, 最大反復回数: {maxiter}")
        
        # 前処理の設定
        precond = None
        if self.solver_options.get("use_preconditioner", True):
            # ILU前処理は大きな行列に有効
            try:
                precond = splinalg.spilu(A)
                precond = splinalg.LinearOperator(A.shape, lambda x: precond.solve(x))
                print("  ILU前処理を適用")
            except Exception as e:
                print(f"  前処理器の作成に失敗: {e}")
        
        # 適切なソルバーの選択と実行
        if method == "gmres":
            x, info = splinalg.gmres(
                A, b, 
                tol=tol, 
                maxiter=maxiter, 
                M=precond,
                restart=self.solver_options.get("restart", 100)
            )
        elif method == "bicgstab":
            x, info = splinalg.bicgstab(
                A, b, 
                tol=tol, 
                maxiter=maxiter, 
                M=precond
            )
        elif method == "cg":  # 対称正定値行列用
            x, info = splinalg.cg(
                A, b, 
                tol=tol, 
                maxiter=maxiter, 
                M=precond
            )
        else:
            raise ValueError(f"未知の反復法: {method}")
        
        if info == 0:
            print(f"  反復法が収束しました")
        else:
            print(f"  警告: 反復法が収束しませんでした (info={info})")
        
        return x

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
        if self.solver_method == "direct":
            sol = self._solve_direct(A, b)
        else:  # iterative, gmres, bicgstab, cg
            sol = self._solve_iterative(A, b)

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