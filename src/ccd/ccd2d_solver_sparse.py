"""
スパース行列を使用した2次元CCD法ソルバーモジュール

スパース行列を活用して、メモリ効率の良い2次元CCDソルバーを提供します。
"""

import jax
import jax.numpy as jnp
import scipy.sparse as sp
import numpy as np
from functools import partial
from typing import Dict, List, Optional, Tuple, Callable, Any

from grid_2d_config import Grid2DConfig
from ccd.ccd_core import CCDLeftHandBuilder, CCDRightHandBuilder, CCDResultExtractor


class SparseCCD2DSolver:
    """
    スパース行列を使用した2次元CCD法ソルバー
    
    大規模な問題に対応するため、スパース行列と適切な解法を使用します。
    """
    
    def __init__(
        self, 
        grid_config: Grid2DConfig, 
        use_direct_solver: bool = True,
        solver_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        初期化
        
        Args:
            grid_config: 2次元グリッド設定
            use_direct_solver: 直接法を使用するかどうか
            solver_kwargs: ソルバーのパラメータ
        """
        self.grid_config = grid_config
        self.left_builder = CCDLeftHandBuilder()
        self.right_builder = CCDRightHandBuilder()
        self.result_extractor = CCDResultExtractor()
        
        # ソルバーパラメータ
        self.use_direct_solver = use_direct_solver
        self.solver_kwargs = solver_kwargs or {}
        
        # グリッドサイズとパラメータ
        self.nx, self.ny = grid_config.nx, grid_config.ny
        self.hx, self.hy = grid_config.hx, grid_config.hy
        self.depth = 4  # 各点での未知数の数
        
        print(f"初期化: グリッド {self.nx}x{self.ny}, 深さ {self.depth}")
        print(f"推定行列サイズ: {self.nx * self.ny * self.depth}x{self.nx * self.ny * self.depth}")
        
        # 係数に応じて2次元演算子行列を構築（スパース形式）
        self.L_2d_sparse = self._build_2d_operator_matrix_sparse()
        
        # 行列サイズと非ゼロ要素数を確認
        nnz = self.L_2d_sparse.nnz
        shape = self.L_2d_sparse.shape
        print(f"スパース行列サイズ: {shape}, 非ゼロ要素数: {nnz}, 密度: {nnz / (shape[0] * shape[1]):.2e}")
    
    def _build_1d_operator_matrices(self) -> Tuple[sp.spmatrix, sp.spmatrix]:
        """
        1次元の演算子行列を構築（スパース形式）
        
        Returns:
            (Lx, Ly): x方向とy方向の1次元スパース演算子行列
        """
        coeffs = self.grid_config.coeffs
        
        # 係数の解釈（一例）
        # [a, b, c, d, e, f] : a*ψ + b*ψx + c*ψxx + d*ψy + e*ψyy + f*ψxy
        
        # 係数が標準的な長さでなければ補正
        if len(coeffs) < 6:
            coeffs = coeffs + [0.0] * (6 - len(coeffs))
        
        # x方向の係数 [a, b, c, 0]
        x_coeffs = [coeffs[0], coeffs[1], coeffs[2], 0.0]
        
        # y方向の係数 [a, d, e, 0]
        y_coeffs = [coeffs[0], coeffs[3], coeffs[4], 0.0]
        
        # 1次元演算子行列を構築（JAXで構築後NumPyに変換）
        x_config = self._create_1d_grid_config(
            n_points=self.nx, 
            h=self.grid_config.hx, 
            coeffs=x_coeffs
        )
        
        y_config = self._create_1d_grid_config(
            n_points=self.ny, 
            h=self.grid_config.hy, 
            coeffs=y_coeffs
        )
        
        Lx_dense = self.left_builder.build_matrix(x_config)
        Ly_dense = self.left_builder.build_matrix(y_config)
        
        # NumPyに変換
        Lx_np = np.array(Lx_dense)
        Ly_np = np.array(Ly_dense)
        
        # スパース行列に変換
        Lx_sparse = sp.csr_matrix(Lx_np)
        Ly_sparse = sp.csr_matrix(Ly_np)
        
        return Lx_sparse, Ly_sparse
    
    def _create_1d_grid_config(self, n_points: int, h: float, coeffs: List[float]):
        """1次元GridConfigを作成（CCDLeftHandBuilderのために）"""
        from ccd.grid_config import GridConfig
        return GridConfig(n_points=n_points, h=h, coeffs=coeffs)
    
    def _build_2d_operator_matrix_sparse(self) -> sp.spmatrix:
        """
        2次元演算子行列を構築（スパース形式）
        
        Returns:
            L_2d_sparse: 完全な2次元スパース演算子行列
        """
        # 1次元演算子行列を取得（スパース形式）
        Lx_sparse, Ly_sparse = self._build_1d_operator_matrices()
        
        # 単位行列（スパース形式）
        Ix_sparse = sp.eye(Lx_sparse.shape[0])
        Iy_sparse = sp.eye(Ly_sparse.shape[0])
        
        # スパース行列のクロネッカー積
        # 注: sp.kron()はSciPyのスパースクロネッカー積
        print("x方向演算子のクロネッカー積を計算中...")
        L_x_2d_sparse = sp.kron(Iy_sparse, Lx_sparse)
        
        print("y方向演算子のクロネッカー積を計算中...")
        L_y_2d_sparse = sp.kron(Ly_sparse, Ix_sparse)
        
        # 交差微分項の処理
        if len(self.grid_config.coeffs) >= 6 and self.grid_config.coeffs[5] != 0:
            # 交差微分項を含む場合（より複雑な実装が必要）
            # 現在のバージョンでは未対応
            print("警告: スパース実装では交差微分項はサポートされていません")
        
        # 合計演算子
        print("演算子の合計を計算中...")
        L_2d_sparse = L_x_2d_sparse + L_y_2d_sparse
        
        # CSR形式に変換して返す（計算効率のため）
        return L_2d_sparse.tocsr()
    
    def solve(self, f: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        2次元偏微分方程式を解く
        
        Args:
            f: 右辺関数値の2次元配列 (ny, nx)
            
        Returns:
            solutions: 解と各種微分を含む辞書
        """
        # 右辺ベクトルを構築
        f_flat = np.array(f).reshape(-1)  # 平坦化とNumPy変換
        total_points = self.nx * self.ny
        rhs = np.zeros(total_points * self.depth)
        
        # 関数値の位置に右辺値を設定
        indices = np.arange(0, total_points * self.depth, self.depth)
        rhs[indices] = f_flat
        
        # 境界条件を適用
        L_sparse, rhs = self._apply_boundary_conditions(self.L_2d_sparse.copy(), rhs.copy())
        
        # 次元を確認して一致させる
        matrix_size = L_sparse.shape[0]
        if rhs.shape[0] != matrix_size:
            print(f"警告: 行列次元({matrix_size})と右辺ベクトル次元({rhs.shape[0]})が一致しません")
            print("右辺ベクトルを調整します")
            
            if rhs.shape[0] < matrix_size:
                # 右辺ベクトルが小さい場合は拡張
                rhs = np.pad(rhs, (0, matrix_size - rhs.shape[0]))
            else:
                # 右辺ベクトルが大きい場合は切り詰め
                rhs = rhs[:matrix_size]
        
        # 連立方程式を解く
        print("連立方程式を解いています...")
        if self.use_direct_solver:
            # 直接法（スパース用）
            from scipy.sparse.linalg import spsolve
            solution = spsolve(L_sparse, rhs)
        else:
            # 反復法（例：BiCGSTAB）
            from scipy.sparse.linalg import bicgstab
            solution, info = bicgstab(L_sparse, rhs, **self.solver_kwargs)
            if info != 0:
                print(f"警告: 反復法が収束しませんでした (info={info})、直接法を試みます")
                # 反復法が失敗した場合は直接法を試す
                solution = spsolve(L_sparse, rhs)
        
        # 結果を抽出して構造化
        return self._extract_solution_components(solution)
    
    def _apply_boundary_conditions(self, L_sparse: sp.spmatrix, rhs: np.ndarray) -> Tuple[sp.spmatrix, np.ndarray]:
        """
        境界条件を行列と右辺ベクトルに適用（スパース行列向け）
        
        Args:
            L_sparse: スパース左辺行列
            rhs: 右辺ベクトル
            
        Returns:
            (L_sparse_modified, rhs_modified): 境界条件を適用した行列とベクトル
        """
        # 行列をCSR形式に変換（行単位の操作に効率的）
        L_sparse = L_sparse.tocsr()
        
        # 境界インデックスを取得
        boundary_indices = self.grid_config.get_boundary_indices()
        
        # ディリクレ境界条件を適用
        for boundary, values in self.grid_config.dirichlet_values.items():
            if boundary in boundary_indices:
                idx_array = boundary_indices[boundary]
                
                for i, idx in enumerate(idx_array):
                    # 関数値に対応する行のインデックス
                    row_idx = idx * self.depth
                    
                    # 行を0にリセット
                    L_sparse[row_idx, :] = 0
                    
                    # 対角要素を1に設定
                    L_sparse[row_idx, row_idx] = 1
                    
                    # 右辺に境界値を設定
                    rhs[row_idx] = values[i]
        
        # ノイマン境界条件の適用（微分値の設定）
        for boundary, values in self.grid_config.neumann_values.items():
            if boundary in boundary_indices:
                idx_array = boundary_indices[boundary]
                
                for i, idx in enumerate(idx_array):
                    # 一階微分に対応する行のインデックス
                    if boundary in ['left', 'right']:
                        # x方向の微分
                        row_idx = idx * self.depth + 1
                    else:  # 'top', 'bottom'
                        # y方向の微分
                        row_idx = idx * self.depth + 3  # 注意: y方向微分は通常3番目
                    
                    # 行を0にリセット
                    L_sparse[row_idx, :] = 0
                    
                    # 対角要素を1に設定
                    L_sparse[row_idx, row_idx] = 1
                    
                    # 右辺に境界値を設定
                    rhs[row_idx] = values[i]
        
        return L_sparse, rhs
    
    def _extract_solution_components(self, solution: np.ndarray) -> Dict[str, jnp.ndarray]:
        """
        解ベクトルから各成分を抽出して辞書形式で返す
        
        Args:
            solution: 解ベクトル（NumPy配列）
            
        Returns:
            components: 各成分を含む辞書（JAX配列）
        """
        total_points = self.nx * self.ny
        
        # 解の長さを確認
        solution_len = solution.shape[0]
        expected_len = total_points * self.depth
        
        if solution_len != expected_len:
            print(f"警告: 解ベクトルの長さ({solution_len})が期待値({expected_len})と異なります")
            # 長さの調整
            if solution_len < expected_len:
                # 解が短い場合は拡張
                solution = np.pad(solution, (0, expected_len - solution_len))
            else:
                # 解が長い場合は切り詰め
                solution = solution[:expected_len]
        
        # 各成分の抽出
        u_values = solution[0::self.depth]
        
        # 解のサイズを確認
        if u_values.shape[0] != total_points:
            print(f"警告: 抽出された関数値の数({u_values.shape[0]})がグリッド点数({total_points})と一致しません")
            # サイズの調整
            if u_values.shape[0] < total_points:
                # 値が足りない場合は0で拡張
                pad_len = total_points - u_values.shape[0]
                u_values = np.pad(u_values, (0, pad_len))
            else:
                # 値が多い場合は切り詰め
                u_values = u_values[:total_points]
        
        # 2次元形状に再構成
        u = u_values.reshape(self.ny, self.nx)
        
        # 基本的な結果辞書
        results = {
            'u': jnp.array(u),  # NumPy配列をJAX配列に変換
        }
        
        # 他の微分も同様にサイズを確認して抽出
        try:
            ux_values = solution[1::self.depth]
            if ux_values.shape[0] >= total_points:
                ux = ux_values[:total_points].reshape(self.ny, self.nx)
                results['ux'] = jnp.array(ux)
            
            uy_values = solution[2::self.depth]
            if uy_values.shape[0] >= total_points:
                uy = uy_values[:total_points].reshape(self.ny, self.nx)
                results['uy'] = jnp.array(uy)
            
            uxx_values = solution[3::self.depth]
            if uxx_values.shape[0] >= total_points:
                uxx = uxx_values[:total_points].reshape(self.ny, self.nx)
                results['uxx'] = jnp.array(uxx)
        except Exception as e:
            print(f"微分値の抽出中にエラーが発生しました: {e}")
        
        return results
