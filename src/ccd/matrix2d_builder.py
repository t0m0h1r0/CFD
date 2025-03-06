"""
2次元CCD行列ビルダーモジュール（メモリ最適化版）

2次元CCD法の左辺行列を生成するクラスを提供します。
疎行列構造を活用してメモリ使用量を最小化します。
"""

import cupy as cp
import cupyx.scipy.sparse as cpx_sparse
from typing import List, Optional, Tuple, Dict, Union
import gc
import time

from grid2d_config import Grid2DConfig
from matrix_builder import CCDLeftHandBuilder  # 1次元CCDの行列ビルダー


class CCD2DLeftHandBuilder:
    """2次元CCD左辺行列を生成するクラス（CuPy疎行列対応、メモリ最適化版）"""

    def __init__(self):
        """初期化"""
        # 1次元CCD行列ビルダーをインスタンス化
        self.builder1d = CCDLeftHandBuilder()
        # メモリ使用量の統計情報
        self.memory_stats = {}

    def _log_memory_usage(self, stage_name: str) -> None:
        """
        現在のメモリ使用量を記録

        Args:
            stage_name: 処理ステージの名前
        """
        # CuPyメモリプールの使用量を取得
        mem_pool = cp.get_default_memory_pool()
        used_bytes = mem_pool.used_bytes()
        total_bytes = mem_pool.total_bytes()
        
        self.memory_stats[stage_name] = {
            "used_bytes": used_bytes,
            "total_bytes": total_bytes,
            "used_mb": used_bytes / (1024 * 1024),
            "total_mb": total_bytes / (1024 * 1024)
        }
        
        print(f"メモリ使用量 [{stage_name}]: 使用={used_bytes/(1024*1024):.2f}MB, 合計={total_bytes/(1024*1024):.2f}MB")

    def build_matrix(
        self, grid_config: Grid2DConfig, coeffs: Optional[Dict[str, float]] = None
    ) -> cpx_sparse.spmatrix:
        """
        2次元CCD全体の左辺行列を生成（メモリ効率版）

        COO形式を使用して行列要素を一度に構築し、メモリ使用量を削減します。

        Args:
            grid_config: 2次元グリッド設定
            coeffs: 係数（省略時はgrid_configから取得）

        Returns:
            2次元CCD左辺行列（CuPy疎行列）
        """
        start_time = time.time()
        self._log_memory_usage("開始")
        
        if coeffs is None:
            coeffs = grid_config.coeffs

        # グリッドサイズとパラメータ
        nx, ny = grid_config.nx, grid_config.ny
        hx, hy = grid_config.hx, grid_config.hy
        
        # 未知数の数（各点でf, f_x, f_y, f_xx）
        depth = 4
        total_unknowns = nx * ny * depth
        
        # メモリ使用量予測と出力
        estimated_nnz = total_unknowns * 9  # 各行につき平均で9つの非ゼロ要素を仮定
        estimated_memory_mb = (estimated_nnz * (8 + 4 + 4)) / (1024 * 1024)  # 値(8バイト) + 行(4バイト) + 列(4バイト)
        
        print(f"行列構築開始 - グリッドサイズ: {nx}x{ny}, 未知数: {total_unknowns}")
        print(f"推定非ゼロ要素数: {estimated_nnz}, 推定メモリ使用量: {estimated_memory_mb:.2f} MB")
        
        # Python listを使用して疎行列データを格納（CuPyのメモリ使用量を削減）
        rows = []
        cols = []
        data = []
        
        # 行列データ構築の分割処理（メモリ使用量を抑制）
        # 分割数を設定
        chunk_size = min(50, ny)  # y方向の分割サイズ
        
        # 分割処理
        for j_start in range(0, ny, chunk_size):
            j_end = min(j_start + chunk_size, ny)
            print(f"行列チャンク構築中: y={j_start}～{j_end-1} ({j_end-j_start}/{ny}行)")
            
            # 各チャンクで構築
            self._build_matrix_chunk(rows, cols, data, grid_config, coeffs, j_start, j_end)
            
            # 必要に応じてガベージコレクションを強制実行
            if j_start % (chunk_size * 5) == 0:
                gc.collect()
                self._log_memory_usage(f"チャンク処理後 y={j_start}～{j_end-1}")
        
        self._log_memory_usage("データ構築後")
        
        # COO形式でCuPy疎行列を作成
        print(f"疎行列作成中 - 非ゼロ要素数: {len(data)}")
        try:
            # CuPyの配列にデータを変換
            coo_data = cp.array(data, dtype=cp.float64)
            coo_rows = cp.array(rows, dtype=cp.int32)
            coo_cols = cp.array(cols, dtype=cp.int32)
            
            self._log_memory_usage("CuPy配列変換後")
            
            # COO形式の疎行列を作成
            L_coo = cpx_sparse.coo_matrix(
                (coo_data, (coo_rows, coo_cols)), 
                shape=(total_unknowns, total_unknowns),
                dtype=cp.float64
            )
            
            self._log_memory_usage("COO行列作成後")
            
            # メモリ使用量削減のため、元のリストを削除
            del rows, cols, data
            gc.collect()
            
            # CSR形式に変換（計算効率化のため）
            L_csr = L_coo.tocsr()
            
            self._log_memory_usage("CSR変換後")
            
            # COO形式の行列を削除してメモリ解放
            del L_coo, coo_data, coo_rows, coo_cols
            gc.collect()
            
        except Exception as e:
            print(f"行列作成エラー: {e}")
            
            # メモリが不足している可能性がある場合、代替アプローチを試行
            print("代替アプローチを試行中...")
            
            # より少ないメモリで構築する方法を試行
            try:
                # データを分割してチャンク単位でCSR行列を作成
                chunk_size = len(data) // 4 + 1
                L_chunks = []
                
                for i in range(0, len(data), chunk_size):
                    end = min(i + chunk_size, len(data))
                    chunk_data = cp.array(data[i:end], dtype=cp.float64)
                    chunk_rows = cp.array(rows[i:end], dtype=cp.int32)
                    chunk_cols = cp.array(cols[i:end], dtype=cp.int32)
                    
                    chunk_coo = cpx_sparse.coo_matrix(
                        (chunk_data, (chunk_rows, chunk_cols)),
                        shape=(total_unknowns, total_unknowns)
                    )
                    L_chunks.append(chunk_coo.tocsr())
                    
                    del chunk_data, chunk_rows, chunk_cols, chunk_coo
                    gc.collect()
                
                # 全てのチャンクを合成
                L_csr = sum(L_chunks)
                del L_chunks
                gc.collect()
                
            except Exception as e2:
                print(f"代替アプローチも失敗: {e2}")
                raise RuntimeError("行列構築に失敗しました。グリッドサイズの削減を検討してください。")
        
        # メモリ使用状況の報告
        nnz = L_csr.nnz
        density = nnz / (total_unknowns * total_unknowns)
        memory_mb = (nnz * (8 + 4 + 4)) / (1024 * 1024)  # 値(8バイト) + 行(4バイト) + 列(4バイト)
        
        elapsed_time = time.time() - start_time
        print(f"行列構築完了 - 経過時間: {elapsed_time:.2f}秒")
        print(f"サイズ: {L_csr.shape}, 非ゼロ要素: {nnz}, 密度: {density:.2e}")
        print(f"推定メモリ使用量: {memory_mb:.2f} MB")
        
        self._log_memory_usage("完了")
        
        return L_csr

    def _build_matrix_chunk(
        self, 
        rows: List[int],
        cols: List[int],
        data: List[float],
        grid_config: Grid2DConfig, 
        coeffs: Dict[str, float],
        j_start: int,
        j_end: int
    ) -> None:
        """
        行列の一部（y方向のチャンク）を構築

        Args:
            rows: 行インデックスを格納するリスト
            cols: 列インデックスを格納するリスト
            data: 値を格納するリスト
            grid_config: 2次元グリッド設定
            coeffs: 係数辞書
            j_start: 開始y座標インデックス
            j_end: 終了y座標インデックス
        """
        nx, ny = grid_config.nx, grid_config.ny
        hx, hy = grid_config.hx, grid_config.hy
        depth = 4  # 各点の未知数の数
        
        # 1. 内部点の方程式を構築
        for j in range(j_start, j_end):
            for i in range(nx):
                # 各点のベースインデックス
                base_idx = (j * nx + i) * depth
                
                # 1-1. f(i,j)の方程式
                self._build_f_equation(rows, cols, data, base_idx, i, j, nx, ny, hx, hy, depth)
                
                # 1-2. f_x(i,j)の方程式
                self._build_fx_equation(rows, cols, data, base_idx, i, j, nx, ny, hx, hy, depth)
                
                # 1-3. f_y(i,j)の方程式
                self._build_fy_equation(rows, cols, data, base_idx, i, j, nx, ny, hx, hy, depth)
                
                # 1-4. f_xx(i,j)の方程式
                self._build_fxx_equation(rows, cols, data, base_idx, i, j, nx, ny, hx, hy, depth)
        
        # 2. x方向の境界条件
        if grid_config.is_x_dirichlet:
            self._build_x_dirichlet_boundary(rows, cols, data, grid_config, j_start, j_end, nx, depth)
        
        if grid_config.is_x_neumann:
            self._build_x_neumann_boundary(rows, cols, data, grid_config, j_start, j_end, nx, depth)
        
        # 3. y方向の境界条件（チャンクの境界にある場合のみ）
        if j_start == 0 and grid_config.is_y_dirichlet:
            self._build_y_dirichlet_boundary(rows, cols, data, grid_config, True, nx, depth)
        
        if j_end == ny and grid_config.is_y_dirichlet:
            self._build_y_dirichlet_boundary(rows, cols, data, grid_config, False, nx, depth)
        
        if j_start == 0 and grid_config.is_y_neumann:
            self._build_y_neumann_boundary(rows, cols, data, grid_config, True, nx, depth)
        
        if j_end == ny and grid_config.is_y_neumann:
            self._build_y_neumann_boundary(rows, cols, data, grid_config, False, nx, depth)

    def _build_f_equation(
        self, rows, cols, data, base_idx, i, j, nx, ny, hx, hy, depth
    ):
        """f(i,j)の方程式を構築"""
        f_idx = base_idx
        
        # 対角成分（自身のf）
        rows.append(f_idx)
        cols.append(f_idx)
        data.append(1.0)  # f = f
        
        # 内部点の場合は差分式を追加
        if 0 < i < nx-1 and 0 < j < ny-1:
            # x方向の中心差分
            rows.append(f_idx)
            cols.append(base_idx - depth + 1)  # 左隣の点のf_x
            data.append(-0.5 * hx)
            
            rows.append(f_idx)
            cols.append(base_idx + depth + 1)  # 右隣の点のf_x
            data.append(0.5 * hx)
            
            # y方向の中心差分
            rows.append(f_idx)
            cols.append(base_idx - nx * depth + 2)  # 下隣の点のf_y
            data.append(-0.5 * hy)
            
            rows.append(f_idx)
            cols.append(base_idx + nx * depth + 2)  # 上隣の点のf_y
            data.append(0.5 * hy)
    
    def _build_fx_equation(
        self, rows, cols, data, base_idx, i, j, nx, ny, hx, hy, depth
    ):
        """f_x(i,j)の方程式を構築"""
        fx_idx = base_idx + 1
        
        # 対角成分（自身のf_x）
        rows.append(fx_idx)
        cols.append(fx_idx)
        data.append(1.0)  # f_x = f_x
        
        # x方向の差分式
        if i > 0 and i < nx-1:
            # 中心差分による関係式
            rows.append(fx_idx)
            cols.append(base_idx - depth)  # 左隣のf
            data.append(-0.5 / hx)
            
            rows.append(fx_idx)
            cols.append(base_idx + depth)  # 右隣のf
            data.append(0.5 / hx)
        elif i == 0:
            # 前方差分（左端）
            rows.append(fx_idx)
            cols.append(base_idx)  # 自身のf
            data.append(-1.0 / hx)
            
            rows.append(fx_idx)
            cols.append(base_idx + depth)  # 右隣のf
            data.append(1.0 / hx)
        elif i == nx-1:
            # 後方差分（右端）
            rows.append(fx_idx)
            cols.append(base_idx - depth)  # 左隣のf
            data.append(-1.0 / hx)
            
            rows.append(fx_idx)
            cols.append(base_idx)  # 自身のf
            data.append(1.0 / hx)
    
    def _build_fy_equation(
        self, rows, cols, data, base_idx, i, j, nx, ny, hx, hy, depth
    ):
        """f_y(i,j)の方程式を構築"""
        fy_idx = base_idx + 2
        
        # 対角成分（自身のf_y）
        rows.append(fy_idx)
        cols.append(fy_idx)
        data.append(1.0)  # f_y = f_y
        
        # y方向の差分式
        if j > 0 and j < ny-1:
            # 中心差分による関係式
            rows.append(fy_idx)
            cols.append(base_idx - nx * depth)  # 下隣のf
            data.append(-0.5 / hy)
            
            rows.append(fy_idx)
            cols.append(base_idx + nx * depth)  # 上隣のf
            data.append(0.5 / hy)
        elif j == 0:
            # 前方差分（下端）
            rows.append(fy_idx)
            cols.append(base_idx)  # 自身のf
            data.append(-1.0 / hy)
            
            rows.append(fy_idx)
            cols.append(base_idx + nx * depth)  # 上隣のf
            data.append(1.0 / hy)
        elif j == ny-1:
            # 後方差分（上端）
            rows.append(fy_idx)
            cols.append(base_idx - nx * depth)  # 下隣のf
            data.append(-1.0 / hy)
            
            rows.append(fy_idx)
            cols.append(base_idx)  # 自身のf
            data.append(1.0 / hy)
    
    def _build_fxx_equation(
        self, rows, cols, data, base_idx, i, j, nx, ny, hx, hy, depth
    ):
        """f_xx(i,j)の方程式を構築"""
        fxx_idx = base_idx + 3
        
        # 対角成分（自身のf_xx）
        rows.append(fxx_idx)
        cols.append(fxx_idx)
        data.append(1.0)  # f_xx = f_xx
        
        # 内部点の場合は中心差分式を追加
        if i > 0 and i < nx-1:
            # 中心差分による関係式
            rows.append(fxx_idx)
            cols.append(base_idx - depth)  # 左隣のf
            data.append(1.0 / (hx * hx))
            
            rows.append(fxx_idx)
            cols.append(base_idx)  # 中央のf
            data.append(-2.0 / (hx * hx))
            
            rows.append(fxx_idx)
            cols.append(base_idx + depth)  # 右隣のf
            data.append(1.0 / (hx * hx))
        elif i == 0:
            # 前方差分（左端）
            rows.append(fxx_idx)
            cols.append(base_idx)  # 自身のf
            data.append(2.0 / (hx * hx))
            
            rows.append(fxx_idx)
            cols.append(base_idx + depth)  # 右隣のf
            data.append(-5.0 / (hx * hx))
            
            rows.append(fxx_idx)
            cols.append(base_idx + 2 * depth)  # 右々隣のf
            data.append(4.0 / (hx * hx))
            
            rows.append(fxx_idx)
            cols.append(base_idx + 3 * depth)  # 右々々隣のf
            data.append(-1.0 / (hx * hx))
        elif i == nx-1:
            # 後方差分（右端）
            rows.append(fxx_idx)
            cols.append(base_idx)  # 自身のf
            data.append(2.0 / (hx * hx))
            
            rows.append(fxx_idx)
            cols.append(base_idx - depth)  # 左隣のf
            data.append(-5.0 / (hx * hx))
            
            rows.append(fxx_idx)
            cols.append(base_idx - 2 * depth)  # 左々隣のf
            data.append(4.0 / (hx * hx))
            
            rows.append(fxx_idx)
            cols.append(base_idx - 3 * depth)  # 左々々隣のf
            data.append(-1.0 / (hx * hx))

    def _build_x_dirichlet_boundary(
        self, rows, cols, data, grid_config, j_start, j_end, nx, depth
    ):
        """x方向のディリクレ境界条件を構築"""
        for j in range(j_start, j_end):
            if j < len(grid_config.x_dirichlet_values):
                # 左端（i=0）の関数値
                left_idx = (j * nx) * depth
                rows.append(left_idx)
                cols.append(left_idx)
                data.append(1.0)  # f = 境界値
                
                # 右端（i=nx-1）の関数値
                right_idx = (j * nx + nx - 1) * depth
                rows.append(right_idx)
                cols.append(right_idx)
                data.append(1.0)  # f = 境界値

    def _build_x_neumann_boundary(
        self, rows, cols, data, grid_config, j_start, j_end, nx, depth
    ):
        """x方向のノイマン境界条件を構築"""
        for j in range(j_start, j_end):
            if j < len(grid_config.x_neumann_values):
                # 左端（i=0）のx方向導関数
                left_idx = (j * nx) * depth + 1  # f_x
                rows.append(left_idx)
                cols.append(left_idx)
                data.append(1.0)  # f_x = 境界値
                
                # 右端（i=nx-1）のx方向導関数
                right_idx = (j * nx + nx - 1) * depth + 1  # f_x
                rows.append(right_idx)
                cols.append(right_idx)
                data.append(1.0)  # f_x = 境界値

    def _build_y_dirichlet_boundary(
        self, rows, cols, data, grid_config, is_bottom, nx, depth
    ):
        """y方向のディリクレ境界条件を構築"""
        j = 0 if is_bottom else grid_config.ny - 1
        
        for i in range(nx):
            if i < len(grid_config.y_dirichlet_values):
                # 関数値インデックス
                idx = (j * nx + i) * depth
                rows.append(idx)
                cols.append(idx)
                data.append(1.0)  # f = 境界値

    def _build_y_neumann_boundary(
        self, rows, cols, data, grid_config, is_bottom, nx, depth
    ):
        """y方向のノイマン境界条件を構築"""
        j = 0 if is_bottom else grid_config.ny - 1
        
        for i in range(nx):
            if i < len(grid_config.y_neumann_values):
                # y方向導関数インデックス
                idx = (j * nx + i) * depth + 2  # f_y
                rows.append(idx)
                cols.append(idx)
                data.append(1.0)  # f_y = 境界値

    def get_memory_stats(self) -> Dict[str, Dict[str, float]]:
        """メモリ使用統計情報を取得"""
        return self.memory_stats