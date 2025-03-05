"""
2次元CCD法ソルバーモジュール

汎用的な2次元結合コンパクト差分法（CCD）による微分計算クラスを提供します。
係数設定に応じて様々な偏微分方程式を解くことができます。
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, Dict, List, Optional, Callable, Any

from grid_2d_config import Grid2DConfig
from ccd.ccd_core import CCDLeftHandBuilder, CCDRightHandBuilder, CCDResultExtractor
from ccd.grid_config import GridConfig


class CCD2DSolver:
    """
    汎用2次元CCD法ソルバー
    
    係数に応じて様々な偏微分方程式を解くことができます：
    - ポアソン方程式 (∇²u = f) 
    - 移流拡散方程式 (a∇²u + b∇u = f)
    - 波動方程式 (∂²u/∂t² = c²∇²u + f)
    - 熱伝導方程式 (∂u/∂t = α∇²u + f)
    などの静的問題や定常問題
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
        
        # 係数に応じて2次元演算子行列を構築
        self.L_2d = self._build_2d_operator_matrix()
        
        # 行列サイズを確認
        print(f"実際の行列サイズ: {self.L_2d.shape}")
        
        # 条件数を推定（小さいグリッドの場合のみ）
        if self.nx * self.ny <= 100:  # 小さなグリッドの場合のみ
            try:
                s = jnp.linalg.svd(self.L_2d, compute_uv=False)
                cond = s[0] / s[-1]
                print(f"行列の条件数（推定）: {cond:.2e}")
            except Exception as e:
                print(f"条件数の計算ができませんでした: {e}")
    
    def _build_1d_operator_matrices(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        1次元の演算子行列を構築
        
        Returns:
            (Lx, Ly): x方向とy方向の1次元演算子行列
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
        
        # 1次元演算子行列を構築
        x_config = GridConfig(
            n_points=self.nx, 
            h=self.hx, 
            coeffs=x_coeffs
        )
        
        y_config = GridConfig(
            n_points=self.ny, 
            h=self.hy, 
            coeffs=y_coeffs
        )
        
        Lx = self.left_builder.build_matrix(x_config)
        Ly = self.left_builder.build_matrix(y_config)
        
        return Lx, Ly
    
    def _build_2d_operator_matrix(self) -> jnp.ndarray:
        """
        2次元演算子行列を構築
        
        Returns:
            L_2d: 完全な2次元演算子行列
        """
        # 1次元演算子行列を取得
        Lx, Ly = self._build_1d_operator_matrices()
        
        # 行列のサイズを確認
        nx_depth = Lx.shape[0]  # nx * depth
        ny_depth = Ly.shape[0]  # ny * depth
        
        print(f"1次元行列サイズ: Lx({Lx.shape}), Ly({Ly.shape})")
        
        # 単位行列
        Ix = jnp.eye(nx_depth)
        Iy = jnp.eye(ny_depth)
        
        # クロネッカー積による2次元演算子
        L_x_2d = jnp.kron(Iy, Lx)  # x方向の演算子
        L_y_2d = jnp.kron(Ly, Ix)  # y方向の演算子
        
        print(f"2次元演算子サイズ: L_x_2d({L_x_2d.shape}), L_y_2d({L_y_2d.shape})")
        
        # 交差微分項（例：ψxy）の処理
        # 係数が十分にある場合のみ
        if len(self.grid_config.coeffs) >= 6 and self.grid_config.coeffs[5] != 0:
            # 交差微分項の係数
            cross_coeff = self.grid_config.coeffs[5]
            
            # 1階微分演算子を抽出
            # これはより複雑な実装が必要かもしれません
            # 簡易的な実装として示しています
            Dx = self._extract_derivative_operator(Lx, derivative=1)
            Dy = self._extract_derivative_operator(Ly, derivative=1)
            
            # 交差微分演算子 ∂²/∂x∂y
            Dxy_2d = cross_coeff * jnp.kron(Dy, Dx)
            
            # 全演算子に追加
            L_2d = L_x_2d + L_y_2d + Dxy_2d
        else:
            # 交差項なし
            # サイズを確認して、必要であれば調整
            if L_x_2d.shape == L_y_2d.shape:
                L_2d = L_x_2d + L_y_2d
            else:
                print(f"警告: 演算子の次元が一致しません。x方向: {L_x_2d.shape}, y方向: {L_y_2d.shape}")
                # 最小の次元に合わせる
                min_dim = min(L_x_2d.shape[0], L_y_2d.shape[0])
                L_2d = L_x_2d[:min_dim, :min_dim] + L_y_2d[:min_dim, :min_dim]
        
        print(f"最終的な演算子サイズ: L_2d({L_2d.shape})")
        return L_2d
    
    def _extract_derivative_operator(self, L: jnp.ndarray, derivative: int) -> jnp.ndarray:
        """
        特定階数の微分演算子を抽出
        
        Args:
            L: 完全な演算子行列
            derivative: 抽出したい微分の階数（1=一階微分、2=二階微分など）
            
        Returns:
            D: 抽出された微分演算子
        """
        # 注：これは簡易的な実装です
        # 実際には、CCDの仕組みに合わせた適切な抽出が必要です
        
        n = L.shape[0] // self.depth
        
        # 特定階数の微分に対応する行を抽出
        rows = []
        for i in range(n):
            row_idx = i * self.depth + derivative  # 0=関数値, 1=一階微分, 2=二階微分...
            rows.append(L[row_idx])
        
        return jnp.vstack(rows)
    
    @partial(jax.jit, static_argnums=(0,))
    def solve(self, f: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        2次元偏微分方程式を解く
        
        Args:
            f: 右辺関数値の2次元配列 (ny, nx)
            
        Returns:
            solutions: 解と各種微分を含む辞書
                {
                    'u': 関数値の2次元配列 (ny, nx),
                    'ux': x方向一階微分 (ny, nx),
                    'uy': y方向一階微分 (ny, nx),
                    'uxx': x方向二階微分 (ny, nx),
                    'uyy': y方向二階微分 (ny, nx) - 必要に応じて追加計算
                }
        """
        # 右辺ベクトルを構築
        f_flat = f.reshape(-1)  # 平坦化
        total_points = self.nx * self.ny
        rhs = jnp.zeros(total_points * self.depth)
        
        # 関数値の位置に右辺値を設定
        indices = jnp.arange(0, total_points * self.depth, self.depth)
        rhs = rhs.at[indices].set(f_flat)
        
        # 境界条件を適用
        L_2d, rhs = self._apply_boundary_conditions(self.L_2d.copy(), rhs.copy())
        
        # 次元を確認して一致させる
        matrix_size = L_2d.shape[0]
        if rhs.shape[0] != matrix_size:
            # 次元が不一致の場合、右辺ベクトルを調整
            print(f"警告: 行列次元({matrix_size})と右辺ベクトル次元({rhs.shape[0]})が一致しません")
            print("右辺ベクトルを調整します")
            
            if rhs.shape[0] < matrix_size:
                # 右辺ベクトルが小さい場合は拡張
                rhs = jnp.pad(rhs, (0, matrix_size - rhs.shape[0]))
            else:
                # 右辺ベクトルが大きい場合は切り詰め
                rhs = rhs[:matrix_size]
        
        # 連立方程式を解く
        if self.use_direct_solver:
            solution = jnp.linalg.solve(L_2d, rhs)
        else:
            # 反復法（例：CG法）
            from jax.scipy.sparse.linalg import cg
            solution, _ = cg(L_2d, rhs, **self.solver_kwargs)
        
        # 結果を抽出して構造化
        return self._extract_solution_components(solution)
    
    def _apply_boundary_conditions(self, L: jnp.ndarray, rhs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        境界条件を行列と右辺ベクトルに適用
        
        Args:
            L: 左辺行列
            rhs: 右辺ベクトル
            
        Returns:
            (L_modified, rhs_modified): 境界条件を適用した行列とベクトル
        """
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
                    L = L.at[row_idx, :].set(0.0)
                    
                    # 対角要素を1に設定
                    L = L.at[row_idx, row_idx].set(1.0)
                    
                    # 右辺に境界値を設定
                    rhs = rhs.at[row_idx].set(values[i])
        
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
                    L = L.at[row_idx, :].set(0.0)
                    
                    # 対角要素を1に設定
                    L = L.at[row_idx, row_idx].set(1.0)
                    
                    # 右辺に境界値を設定
                    rhs = rhs.at[row_idx].set(values[i])
        
        return L, rhs
    
    def _extract_solution_components(self, solution: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        解ベクトルから各成分を抽出して辞書形式で返す
        
        Args:
            solution: 解ベクトル
            
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
                # 解が短い場合は拡張
                solution = jnp.pad(solution, (0, expected_len - solution_len))
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
                u_values = jnp.pad(u_values, (0, pad_len))
            else:
                # 値が多い場合は切り詰め
                u_values = u_values[:total_points]
        
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
    
    def get_poisson_solver(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """
        ポアソン方程式専用のソルバー関数を返す
        
        Returns:
            poisson_solver: ポアソン方程式のソルバー関数
        """
        @partial(jax.jit, static_argnums=(0,))
        def poisson_solver(f: jnp.ndarray) -> jnp.ndarray:
            """
            ポアソン方程式 ∇²u = f を解く
            
            Args:
                f: 右辺関数値の2次元配列 (ny, nx)
                
            Returns:
                u: 解の2次元配列 (ny, nx)
            """
            # 汎用ソルバーを使用
            solution = self.solve(f)
            
            # 関数値のみを返す
            return solution['u']
        
        return poisson_solver
    
    def get_laplacian(self, u: jnp.ndarray) -> jnp.ndarray:
        """
        関数のラプラシアン ∇²u = uxx + uyy を計算
        
        Args:
            u: 入力関数の2次元配列 (ny, nx)
            
        Returns:
            laplacian: ラプラシアンの2次元配列 (ny, nx)
        """
        # 現在の係数を保存
        original_coeffs = self.grid_config.coeffs.copy()
        
        # ラプラシアン用の係数に設定（uxx + uyy）
        laplacian_coeffs = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        self.grid_config.coeffs = laplacian_coeffs
        
        # ラプラシアン用の行列を構築
        laplacian_matrix = self._build_2d_operator_matrix()
        
        # u値から右辺ベクトルを構築
        u_flat = u.reshape(-1)
        rhs = jnp.zeros(self.nx * self.ny * self.depth)
        indices = jnp.arange(0, self.nx * self.ny * self.depth, self.depth)
        rhs = rhs.at[indices].set(u_flat)
        
        # 境界条件を適用
        laplacian_matrix, rhs = self._apply_boundary_conditions(laplacian_matrix, rhs)
        
        # ラプラシアンを計算
        laplacian_values = laplacian_matrix @ rhs
        
        # 関数値の位置からラプラシアン値を抽出
        laplacian = laplacian_values[0::self.depth].reshape(self.ny, self.nx)
        
        # 元の係数に戻す
        self.grid_config.coeffs = original_coeffs
        
        return laplacian