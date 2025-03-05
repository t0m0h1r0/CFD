"""
高速スパース2次元CCDソルバー

事前分解された逆行列を用いて、線形時間で複数の右辺に対する解を計算します。
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu, spsolve
import time
from typing import Dict, List, Optional, Tuple, Callable, Any

from grid_2d_config import Grid2DConfig
from ccd.ccd_core import CCDLeftHandBuilder, CCDRightHandBuilder


class FastSparseCCD2DSolver:
    """
    事前計算された逆行列（LU分解）を使用した高速2次元CCDソルバー
    
    同じ格子・境界条件に対して複数回のソルブを高速化するために、
    行列分解を事前計算して再利用します。
    """
    
    def __init__(
        self, 
        grid_config: Grid2DConfig, 
        precompute: bool = True,
        solver_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        初期化
        
        Args:
            grid_config: 2次元グリッド設定
            precompute: 初期化時に行列分解を事前計算するかどうか
            solver_kwargs: ソルバーのパラメータ
        """
        self.grid_config = grid_config
        self.left_builder = CCDLeftHandBuilder()
        self.right_builder = CCDRightHandBuilder()
        
        # ソルバーパラメータ
        self.solver_kwargs = solver_kwargs or {}
        
        # グリッドサイズとパラメータ
        self.nx, self.ny = grid_config.nx, grid_config.ny
        self.hx, self.hy = grid_config.hx, grid_config.hy
        self.depth = 4  # 各点での未知数の数
        
        # システム行列とLU分解の初期化
        self.L_sparse = None  # システム行列（スパース形式）
        self.LU = None        # LU分解
        
        print(f"初期化: グリッド {self.nx}x{self.ny}, 深さ {self.depth}")
        
        # 事前計算フラグが有効なら、初期化時に行列とLU分解を準備
        if precompute:
            self.setup_system_matrix()
    
    def setup_system_matrix(self):
        """
        システム行列とそのLU分解を構築・保存
        
        境界条件が変わった場合などに再実行する必要があります。
        """
        start_time = time.time()
        print("システム行列を構築中...")
        
        # スパース形式で行列を構築
        self.L_sparse = self._build_2d_operator_matrix_sparse()
        
        # 境界条件を適用
        self.L_sparse = self._apply_boundary_conditions(self.L_sparse)
        
        # 行列サイズと非ゼロ要素数を確認
        matrix_size = self.L_sparse.shape[0]
        nnz = self.L_sparse.nnz
        print(f"スパース行列サイズ: {matrix_size}x{matrix_size}, 非ゼロ要素数: {nnz}, 密度: {nnz / (matrix_size**2):.2e}")
        
        # LU分解を事前計算
        print("LU分解を計算中...")
        self.LU = splu(self.L_sparse.tocsc())  # CSC形式が直接法に最適
        
        matrix_time = time.time() - start_time
        print(f"システム行列とLU分解の準備完了 ({matrix_time:.2f}秒)")
    
    def _build_1d_operator_matrices(self) -> Tuple[sp.spmatrix, sp.spmatrix]:
        """
        1次元の演算子行列を構築（スパース形式）
        
        Returns:
            (Lx, Ly): x方向とy方向の1次元スパース演算子行列
        """
        coeffs = self.grid_config.coeffs
        
        # 係数の解釈
        # [a, b, c, d, e, f] : a*ψ + b*ψx + c*ψxx + d*ψy + e*ψyy + f*ψxy
        
        # 係数が標準的な長さでなければ補正
        if len(coeffs) < 6:
            coeffs = coeffs + [0.0] * (6 - len(coeffs))
        
        # x方向の係数 [a, b, c, 0]
        x_coeffs = [coeffs[0], coeffs[1], coeffs[2], 0.0]
        
        # y方向の係数 [a, d, e, 0]
        y_coeffs = [coeffs[0], coeffs[3], coeffs[4], 0.0]
        
        # 1次元演算子行列を構築
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
        # 1次元演算子行列を取得
        Lx_sparse, Ly_sparse = self._build_1d_operator_matrices()
        
        # 単位行列（スパース形式）
        Ix_sparse = sp.eye(Lx_sparse.shape[0])
        Iy_sparse = sp.eye(Ly_sparse.shape[0])
        
        # スパース行列のクロネッカー積
        print("x方向演算子のクロネッカー積を計算中...")
        L_x_2d_sparse = sp.kron(Iy_sparse, Lx_sparse)
        
        print("y方向演算子のクロネッカー積を計算中...")
        L_y_2d_sparse = sp.kron(Ly_sparse, Ix_sparse)
        
        # 交差微分項を含む場合
        if len(self.grid_config.coeffs) >= 6 and self.grid_config.coeffs[5] != 0:
            print("警告: スパース実装では交差微分項はサポートされていません")
        
        # 合計演算子
        print("演算子の合計を計算中...")
        L_2d_sparse = L_x_2d_sparse + L_y_2d_sparse
        
        # CSR形式に変換して返す
        return L_2d_sparse.tocsr()
    
    def _apply_boundary_conditions(self, L_sparse: sp.spmatrix) -> sp.spmatrix:
        """
        境界条件を行列に適用
        
        Args:
            L_sparse: スパース左辺行列
            
        Returns:
            L_sparse_modified: 境界条件を適用した行列
        """
        # CSR形式に変換（行単位の操作に効率的）
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
        
        # ノイマン境界条件の適用
        for boundary, values in self.grid_config.neumann_values.items():
            if boundary in boundary_indices:
                idx_array = boundary_indices[boundary]
                
                for i, idx in enumerate(idx_array):
                    # 一階微分に対応する行のインデックス
                    if boundary in ['left', 'right']:
                        row_idx = idx * self.depth + 1
                    else:  # 'top', 'bottom'
                        row_idx = idx * self.depth + 3
                    
                    # 行を0にリセット
                    L_sparse[row_idx, :] = 0
                    
                    # 対角要素を1に設定
                    L_sparse[row_idx, row_idx] = 1
        
        return L_sparse
    
    def build_rhs(self, f: np.ndarray) -> np.ndarray:
        """
        右辺ベクトルを構築
        
        Args:
            f: 右辺関数値の2次元配列 (ny, nx)
            
        Returns:
            rhs: 右辺ベクトル（境界条件適用済み）
        """
        # 右辺ベクトルを構築
        f_flat = np.array(f).reshape(-1)  # 平坦化
        total_points = self.nx * self.ny
        rhs = np.zeros(total_points * self.depth)
        
        # 関数値の位置に右辺値を設定
        indices = np.arange(0, total_points * self.depth, self.depth)
        rhs[indices] = f_flat
        
        # 境界条件を適用
        # 境界インデックスを取得
        boundary_indices = self.grid_config.get_boundary_indices()
        
        # ディリクレ境界条件を適用
        for boundary, values in self.grid_config.dirichlet_values.items():
            if boundary in boundary_indices:
                idx_array = boundary_indices[boundary]
                
                for i, idx in enumerate(idx_array):
                    # 関数値に対応する行のインデックス
                    row_idx = idx * self.depth
                    
                    # 右辺に境界値を設定
                    rhs[row_idx] = values[i]
        
        # ノイマン境界条件の適用
        for boundary, values in self.grid_config.neumann_values.items():
            if boundary in boundary_indices:
                idx_array = boundary_indices[boundary]
                
                for i, idx in enumerate(idx_array):
                    # 一階微分に対応する行のインデックス
                    if boundary in ['left', 'right']:
                        row_idx = idx * self.depth + 1
                    else:  # 'top', 'bottom'
                        row_idx = idx * self.depth + 3
                    
                    # 右辺に境界値を設定
                    rhs[row_idx] = values[i]
        
        return rhs
    
    def solve(self, f: np.ndarray) -> Dict[str, np.ndarray]:
        """
        2次元偏微分方程式を解く
        
        事前計算されたLU分解を使用して高速に解きます。
        
        Args:
            f: 右辺関数値の2次元配列 (ny, nx)
            
        Returns:
            solutions: 解と各種微分を含む辞書
        """
        start_time = time.time()
        
        # システム行列が未設定の場合は構築
        if self.L_sparse is None:
            self.setup_system_matrix()
        
        # 右辺ベクトルを構築
        rhs = self.build_rhs(f)
        
        # 連立方程式を解く
        print("方程式を解いています...")
        if self.LU is not None:
            # 事前計算されたLU分解を使用（高速）
            solution = self.LU.solve(rhs)
        else:
            # 直接法を使用（より遅い）
            solution = spsolve(self.L_sparse, rhs)
        
        solve_time = time.time() - start_time
        print(f"方程式を解きました ({solve_time:.2f}秒)")
        
        # 結果を抽出して構造化
        return self._extract_solution_components(solution)
    
    def solve_multiple(self, f_list: List[np.ndarray]) -> List[Dict[str, np.ndarray]]:
        """
        複数の右辺に対して解く
        
        Args:
            f_list: 右辺関数値の2次元配列のリスト [(ny, nx), ...]
            
        Returns:
            solutions_list: 解と各種微分を含む辞書のリスト
        """
        # システム行列が未設定の場合は構築
        if self.L_sparse is None:
            self.setup_system_matrix()
        
        start_time = time.time()
        solutions = []
        
        print(f"{len(f_list)}個の右辺に対して解いています...")
        for i, f in enumerate(f_list):
            # 右辺ベクトルを構築
            rhs = self.build_rhs(f)
            
            # 連立方程式を解く
            if self.LU is not None:
                # 事前計算されたLU分解を使用（高速）
                solution = self.LU.solve(rhs)
            else:
                # 直接法を使用（より遅い）
                solution = spsolve(self.L_sparse, rhs)
            
            # 結果を抽出して構造化し、リストに追加
            solutions.append(self._extract_solution_components(solution))
            
            if (i + 1) % 10 == 0 or i == len(f_list) - 1:
                print(f"  {i + 1}/{len(f_list)} 完了")
        
        total_time = time.time() - start_time
        avg_time = total_time / len(f_list)
        print(f"全ての方程式を解きました (合計: {total_time:.2f}秒, 平均: {avg_time:.2f}秒/問題)")
        
        return solutions
    
    def _extract_solution_components(self, solution: np.ndarray) -> Dict[str, np.ndarray]:
        """
        解ベクトルから各成分を抽出して辞書形式で返す
        
        Args:
            solution: 解ベクトル（NumPy配列）
            
        Returns:
            components: 各成分を含む辞書
        """
        total_points = self.nx * self.ny
        
        # 解の長さを確認
        solution_len = solution.shape[0]
        expected_len = total_points * self.depth
        
        if solution_len != expected_len:
            print(f"警告: 解ベクトルの長さ({solution_len})が期待値({expected_len})と異なります")
            # 長さの調整
            if solution_len < expected_len:
                solution = np.pad(solution, (0, expected_len - solution_len))
            else:
                solution = solution[:expected_len]
        
        # 各成分の抽出
        u_values = solution[0::self.depth]
        
        # 2次元形状に再構成
        u = u_values.reshape(self.ny, self.nx)
        
        # 基本的な結果辞書
        results = {
            'u': u,
        }
        
        # 他の微分も同様にサイズを確認して抽出
        try:
            ux_values = solution[1::self.depth]
            if ux_values.shape[0] >= total_points:
                ux = ux_values[:total_points].reshape(self.ny, self.nx)
                results['ux'] = ux
            
            uy_values = solution[2::self.depth]
            if uy_values.shape[0] >= total_points:
                uy = uy_values[:total_points].reshape(self.ny, self.nx)
                results['uy'] = uy
            
            uxx_values = solution[3::self.depth]
            if uxx_values.shape[0] >= total_points:
                uxx = uxx_values[:total_points].reshape(self.ny, self.nx)
                results['uxx'] = uxx
        except Exception as e:
            print(f"微分値の抽出中にエラーが発生しました: {e}")
        
        return results
