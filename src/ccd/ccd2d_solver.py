"""
2次元CuPy対応結合コンパクト差分法ソルバーモジュール

2次元結合コンパクト差分法による微分計算クラスを提供します。
"""

import cupy as cp
import cupyx.scipy.sparse.linalg as cpx_spla
import cupyx.scipy.sparse as cpx_sparse
from typing import Dict, Any, Optional, Tuple

from grid2d_config import Grid2DConfig
from system2d_builder import CCD2DSystemBuilder
from matrix2d_builder import CCD2DLeftHandBuilder
from vector2d_builder import CCD2DRightHandBuilder
from result2d_extractor import CCD2DResultExtractor


class CCD2DSolver:
    """2次元結合コンパクト差分法による微分計算クラス（CuPy対応）"""

    def __init__(
        self,
        grid_config: Grid2DConfig,
        use_iterative: bool = True,
        solver_type: str = "gmres",
        solver_kwargs: Optional[Dict[str, Any]] = None,
        system_builder: Optional[CCD2DSystemBuilder] = None,
        coeffs: Optional[Dict[str, float]] = None,
    ):
        """
        2次元CCDソルバーの初期化

        Args:
            grid_config: 2次元グリッド設定
            use_iterative: 反復法を使用するかどうか
            solver_type: 反復法のタイプ ('gmres', 'cg', 'bicgstab', 'lsqr')
            solver_kwargs: 線形ソルバーのパラメータ
            system_builder: CCD2DSystemBuilderのインスタンス
            coeffs: 方程式の係数
        """
        # CuPyデバイスの明示的選択
        cp.cuda.Device(0).use()

        self.grid_config = grid_config

        # 係数の設定
        if coeffs is not None:
            self.grid_config.coeffs = coeffs

        # 係数への参照
        self.coeffs = self.grid_config.coeffs

        # システムビルダーの初期化
        if system_builder is None:
            matrix_builder = CCD2DLeftHandBuilder()
            vector_builder = CCD2DRightHandBuilder()
            result_extractor = CCD2DResultExtractor()
            system_builder = CCD2DSystemBuilder(matrix_builder, vector_builder, result_extractor)
        
        self.system_builder = system_builder

        # デフォルトの零ベクトルもCuPyに変換
        nx, ny = grid_config.nx, grid_config.ny
        zero_values = cp.zeros((nx, ny))

        # システムビルダーを使用して行列を作成
        self.L, _ = self.system_builder.build_system(grid_config, zero_values)

        # 線形ソルバーの設定
        self.use_iterative = use_iterative
        self.solver_type = solver_type.lower()
        self.solver_kwargs = solver_kwargs or {}
        
        # プリコンディショナの設定（サポートされている場合）
        self.preconditioner = None
        if self.use_iterative and "precond" in self.solver_kwargs:
            precond_type = self.solver_kwargs.pop("precond")
            self._setup_preconditioner(precond_type)

    def _setup_preconditioner(self, precond_type: str = "ilu"):
        """
        プリコンディショナを設定
        
        Args:
            precond_type: プリコンディショナのタイプ ('ilu', 'jacobi', etc.)
        """
        if precond_type == "ilu":
            from cupyx.scipy.sparse.linalg import spilu
            # ILU分解をプリコンディショナとして使用
            try:
                ilu = spilu(self.L.tocsc())
                
                # LinearOperatorを作成してプリコンディショナとして使用
                from cupyx.scipy.sparse.linalg import LinearOperator
                
                def mv(v):
                    return ilu.solve(v)
                
                n = self.L.shape[0]
                self.preconditioner = LinearOperator((n, n), matvec=mv)
            except Exception as e:
                print(f"ILUプリコンディショナの設定に失敗しました: {e}")
                self.preconditioner = None
        
        elif precond_type == "jacobi":
            # ヤコビプリコンディショナ（対角成分の逆数）
            diag = self.L.diagonal()
            diag_inv = 1.0 / cp.maximum(cp.abs(diag), 1e-10)
            
            # LinearOperatorを作成
            from cupyx.scipy.sparse.linalg import LinearOperator
            
            def mv(v):
                return diag_inv * v
            
            n = self.L.shape[0]
            self.preconditioner = LinearOperator((n, n), matvec=mv)
            
        else:
            print(f"未対応のプリコンディショナタイプ: {precond_type}")
            self.preconditioner = None

    def solve(
        self, f_values: cp.ndarray
    ) -> Dict[str, cp.ndarray]:
        """
        2次元関数値fに対するCCD方程式系を解き、関数とその各階微分を返す

        Args:
            f_values: グリッド点での関数値 (shape: (nx, ny))

        Returns:
            各成分を含む辞書
            {
                "f": 関数値(nx, ny),
                "f_x": x方向1階微分(nx, ny),
                "f_y": y方向1階微分(nx, ny),
                "f_xx": x方向2階微分(nx, ny),
                ...
            }
        """
        # 入力をCuPy配列に変換
        f_values_cupy = cp.asarray(f_values)

        # 右辺ベクトルを構築
        _, b = self.system_builder.build_system(self.grid_config, f_values_cupy)

        # 線形方程式を解く
        if self.use_iterative:
            solution = self._solve_iterative(self.L, b)
        else:
            solution = self._solve_direct(self.L, b)

        # 解から関数値と各階微分を抽出
        result = self.system_builder.extract_results(self.grid_config, solution)

        # メモリ管理: 不要になった大きな配列を明示的に解放
        del f_values_cupy, b, solution
        cp.get_default_memory_pool().free_all_blocks()

        return result

    def _solve_direct(self, L: cpx_sparse.spmatrix, b: cp.ndarray) -> cp.ndarray:
        """
        直接法で線形方程式を解く

        Args:
            L: 左辺行列
            b: 右辺ベクトル

        Returns:
            解ベクトル
        """
        # スパース行列の場合はCSRに変換
        if not isinstance(L, cpx_sparse.csr_matrix):
            L = cpx_sparse.csr_matrix(L)

        # 直接法で解く
        try:
            return cpx_spla.spsolve(L, b)
        except Exception as e:
            print(f"直接法での解法に失敗しました: {e}")
            print("反復法を試みます...")
            return self._solve_iterative(L, b)

    def _solve_iterative(self, L: cpx_sparse.spmatrix, b: cp.ndarray) -> cp.ndarray:
        """
        反復法で線形方程式を解く

        Args:
            L: 左辺行列
            b: 右辺ベクトル

        Returns:
            解ベクトル
        """
        # スパース行列の場合はCSRに変換
        if not isinstance(L, cpx_sparse.csr_matrix):
            L = cpx_sparse.csr_matrix(L)

        # ソルバーのパラメータの取得
        tol = self.solver_kwargs.get("tol", 1e-10)
        atol = self.solver_kwargs.get("atol", 1e-12)
        maxiter = self.solver_kwargs.get("maxiter", 1000)
        restart = self.solver_kwargs.get("restart", 20)

        # 選択されたソルバーで解く
        if self.solver_type == "gmres":
            x, info = cpx_spla.gmres(
                L, b, tol=tol, atol=atol, restart=restart, maxiter=maxiter,
                M=self.preconditioner
            )
        elif self.solver_type == "cg":
            x, info = cpx_spla.cg(
                L, b, tol=tol, atol=atol, maxiter=maxiter,
                M=self.preconditioner
            )
        elif self.solver_type == "bicgstab":
            x, info = cpx_spla.bicgstab(
                L, b, tol=tol, atol=atol, maxiter=maxiter,
                M=self.preconditioner
            )
        elif self.solver_type == "lsqr":
            # LSQRにはプリコンディショナを直接渡せない
            x, info = cpx_spla.lsqr(
                L, b, atol=atol, btol=tol, iter_lim=maxiter
            )
        else:
            raise ValueError(f"未対応のソルバータイプ: {self.solver_type}")

        # 収束状態の確認
        if info != 0:
            print(f"警告: 反復法が収束しませんでした (info={info})")

        return x

    def get_system_info(self) -> Dict[str, Any]:
        """
        システムの情報を取得
        
        Returns:
            システム情報を含む辞書
        """
        # 行列のサイズと疎密度
        nnz = self.L.nnz
        size = self.L.shape[0]
        density = nnz / (size * size)
        
        return {
            "matrix_size": size,
            "nonzeros": nnz,
            "density": density,
            "grid_nx": self.grid_config.nx,
            "grid_ny": self.grid_config.ny,
            "x_deriv_order": self.grid_config.x_deriv_order,
            "y_deriv_order": self.grid_config.y_deriv_order,
            "coeffs": self.coeffs,
            "solver_type": "direct" if not self.use_iterative else self.solver_type,
        }
