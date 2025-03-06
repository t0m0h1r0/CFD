"""
2次元CCD右辺ベクトルビルダーモジュール（メモリ最適化版）

2次元CCD法の右辺ベクトルを生成するクラスを提供します。
メモリ使用量を最小化する実装です。
"""

import cupy as cp
import numpy as np
from typing import List, Optional, Dict, Tuple, Union, Any
import gc
import time

from grid2d_config import Grid2DConfig


class CCD2DRightHandBuilder:
    """2次元CCD右辺ベクトルを生成するクラス（CuPy対応、メモリ最適化版）"""

    def __init__(self):
        """初期化"""
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

    def build_vector(
        self,
        grid_config: Grid2DConfig,
        values: cp.ndarray,
        coeffs: Optional[Dict[str, float]] = None,
        x_dirichlet_enabled: bool = None,
        y_dirichlet_enabled: bool = None,
        x_neumann_enabled: bool = None,
        y_neumann_enabled: bool = None,
    ) -> cp.ndarray:
        """
        2次元関数値から右辺ベクトルを生成（メモリ効率版）

        Args:
            grid_config: 2次元グリッド設定
            values: 2次元格子上の関数値 (shape: (nx, ny))
            coeffs: 係数（省略時はgrid_configから取得）
            x_dirichlet_enabled: x方向ディリクレ境界条件の有効/無効
            y_dirichlet_enabled: y方向ディリクレ境界条件の有効/無効
            x_neumann_enabled: x方向ノイマン境界条件の有効/無効
            y_neumann_enabled: y方向ノイマン境界条件の有効/無効

        Returns:
            右辺ベクトル（CuPy配列）
        """
        start_time = time.time()
        self._log_memory_usage("開始")
        
        if coeffs is None:
            coeffs = grid_config.coeffs

        # 境界条件の状態を決定
        use_x_dirichlet = (
            grid_config.is_x_dirichlet if x_dirichlet_enabled is None else x_dirichlet_enabled
        )
        use_y_dirichlet = (
            grid_config.is_y_dirichlet if y_dirichlet_enabled is None else y_dirichlet_enabled
        )
        use_x_neumann = (
            grid_config.is_x_neumann if x_neumann_enabled is None else x_neumann_enabled
        )
        use_y_neumann = (
            grid_config.is_y_neumann if y_neumann_enabled is None else y_neumann_enabled
        )

        # 値の形状を確認
        nx, ny = grid_config.nx, grid_config.ny
        if values.shape != (nx, ny):
            print(f"警告: 関数値の形状 {values.shape} がグリッド設定 ({nx}, {ny}) と一致しません")
            # 適切なサイズに調整（必要に応じて）
            if values.size > 0:
                # 既存の値を使用して新しい配列を作成
                adjusted_values = cp.zeros((nx, ny), dtype=values.dtype)
                min_nx = min(nx, values.shape[0])
                min_ny = min(ny, values.shape[1])
                adjusted_values[:min_nx, :min_ny] = values[:min_nx, :min_ny]
                values = adjusted_values
                print(f"関数値を調整しました: {values.shape}")
            else:
                # 空の配列の場合は0で初期化
                values = cp.zeros((nx, ny), dtype=cp.float64)
                print("空の関数値配列を0で初期化しました")

        # 各点の未知数の数
        depth = 4  # f, f_x, f_y, f_xx

        # 合計未知数
        total_unknowns = nx * ny * depth
        
        # メモリ効率のため、ストリーミング方式で右辺ベクトルを構築
        print(f"右辺ベクトル構築 - サイズ: {total_unknowns}, メモリ効率モード")
        
        # 推定メモリ使用量の計算
        estimated_memory_mb = (total_unknowns * 8) / (1024 * 1024)  # 8バイト（倍精度浮動小数点）
        print(f"推定メモリ使用量: {estimated_memory_mb:.2f} MB")
        
        # 右辺ベクトルを直接生成する方法
        try:
            # 右辺ベクトルを初期化
            rhs = cp.zeros(total_unknowns, dtype=cp.float64)
            self._log_memory_usage("ベクトル初期化後")
            
            # チャンク処理（メモリ使用量を抑制）
            chunk_size = min(50, ny)  # y方向の分割サイズ
            
            # 分割処理
            for j_start in range(0, ny, chunk_size):
                j_end = min(j_start + chunk_size, ny)
                print(f"ベクトルチャンク構築中: y={j_start}～{j_end-1} ({j_end-j_start}/{ny}行)")
                
                # チャンク内での処理
                self._build_vector_chunk(
                    rhs, values, grid_config,
                    j_start, j_end, nx, ny, depth,
                    use_x_dirichlet, use_y_dirichlet,
                    use_x_neumann, use_y_neumann
                )
                
                # 必要に応じてガベージコレクションを強制実行
                if j_start % (chunk_size * 5) == 0:
                    gc.collect()
                    self._log_memory_usage(f"チャンク処理後 y={j_start}～{j_end-1}")
            
        except Exception as e:
            print(f"右辺ベクトル構築エラー: {e}")
            
            # 代替アプローチ（NumPy配列を使用してから変換）
            print("代替アプローチを試行中...")
            try:
                # まずNumPy配列として作成（メモリ使用量を抑制）
                rhs_np = np.zeros(total_unknowns, dtype=np.float64)
                
                # チャンク処理でNumPy配列を構築
                for j_start in range(0, ny, chunk_size):
                    j_end = min(j_start + chunk_size, ny)
                    print(f"Numpy ベクトルチャンク構築中: y={j_start}～{j_end-1}")
                    
                    # NumPy用の値配列を取得
                    values_np = cp.asnumpy(values) if isinstance(values, cp.ndarray) else values
                    
                    # チャンク内での処理（NumPy版）
                    self._build_vector_chunk_numpy(
                        rhs_np, values_np, grid_config,
                        j_start, j_end, nx, ny, depth,
                        use_x_dirichlet, use_y_dirichlet,
                        use_x_neumann, use_y_neumann
                    )
                
                # 最後にCuPy配列に変換
                rhs = cp.array(rhs_np)
                del rhs_np
                gc.collect()
                
            except Exception as e2:
                print(f"代替アプローチも失敗: {e2}")
                raise RuntimeError("右辺ベクトル構築に失敗しました。グリッドサイズの削減を検討してください。")
        
        # 境界条件のインデックスに対するデバッグ出力
        if False:  # デバッグが必要な場合はTrueに変更
            self._print_boundary_values(
                rhs, grid_config, nx, ny, depth,
                use_x_dirichlet, use_y_dirichlet,
                use_x_neumann, use_y_neumann
            )
        
        elapsed_time = time.time() - start_time
        print(f"右辺ベクトル構築完了 - 経過時間: {elapsed_time:.2f}秒")
        self._log_memory_usage("完了")
        
        return rhs
        
    def _build_vector_chunk(
        self,
        rhs: cp.ndarray,
        values: cp.ndarray,
        grid_config: Grid2DConfig,
        j_start: int,
        j_end: int,
        nx: int,
        ny: int,
        depth: int,
        use_x_dirichlet: bool,
        use_y_dirichlet: bool,
        use_x_neumann: bool,
        use_y_neumann: bool
    ) -> None:
        """
        右辺ベクトルのチャンクを構築（CuPy版）

        Args:
            rhs: 右辺ベクトル（更新される）
            values: 関数値の2次元配列
            grid_config: グリッド設定
            j_start: 開始y座標インデックス
            j_end: 終了y座標インデックス
            nx, ny: グリッドサイズ
            depth: 各点の未知数の数
            use_x_dirichlet: x方向ディリクレ境界条件の使用フラグ
            use_y_dirichlet: y方向ディリクレ境界条件の使用フラグ
            use_x_neumann: x方向ノイマン境界条件の使用フラグ
            use_y_neumann: y方向ノイマン境界条件の使用フラグ
        """
        # 1. 関数値（f）の設定
        # 2次元のインデックス→フラット化されたインデックスに変換しながら
        for j in range(j_start, j_end):
            for i in range(nx):
                # f の位置は各点の先頭（各点は depth=4 個の未知数を持つ）
                idx = (j * nx + i) * depth
                rhs[idx] = values[i, j]
        
        # 2. 境界条件の設定
        # x方向ディリクレ境界条件（左右端）
        if use_x_dirichlet and grid_config.x_dirichlet_values is not None:
            for j in range(j_start, j_end):
                if j < len(grid_config.x_dirichlet_values):
                    left_val, right_val = grid_config.x_dirichlet_values[j]
                    
                    # 左端（i=0）の関数値
                    left_idx = (j * nx) * depth
                    rhs[left_idx] = left_val
                    
                    # 右端（i=nx-1）の関数値
                    right_idx = (j * nx + nx - 1) * depth
                    rhs[right_idx] = right_val
        
        # y方向ディリクレ境界条件（上下端）
        if use_y_dirichlet and grid_config.y_dirichlet_values is not None:
            # 下端（j=0）
            if j_start == 0:
                for i in range(nx):
                    if i < len(grid_config.y_dirichlet_values):
                        bottom_val, _ = grid_config.y_dirichlet_values[i]
                        bottom_idx = i * depth
                        rhs[bottom_idx] = bottom_val
            
            # 上端（j=ny-1）
            if j_end > ny - 1:
                for i in range(nx):
                    if i < len(grid_config.y_dirichlet_values):
                        _, top_val = grid_config.y_dirichlet_values[i]
                        top_idx = ((ny - 1) * nx + i) * depth
                        rhs[top_idx] = top_val
        
        # x方向ノイマン境界条件
        if use_x_neumann and grid_config.x_neumann_values is not None:
            for j in range(j_start, j_end):
                if j < len(grid_config.x_neumann_values):
                    left_val, right_val = grid_config.x_neumann_values[j]
                    
                    # 左端のx方向導関数（f_xはインデックス+1の位置）
                    left_idx = (j * nx) * depth + 1
                    rhs[left_idx] = left_val
                    
                    # 右端のx方向導関数
                    right_idx = (j * nx + nx - 1) * depth + 1
                    rhs[right_idx] = right_val
        
        # y方向ノイマン境界条件
        if use_y_neumann and grid_config.y_neumann_values is not None:
            # 下端（j=0）
            if j_start == 0:
                for i in range(nx):
                    if i < len(grid_config.y_neumann_values):
                        bottom_val, _ = grid_config.y_neumann_values[i]
                        bottom_idx = i * depth + 2
                        rhs[bottom_idx] = bottom_val
            
            # 上端（j=ny-1）
            if j_end > ny - 1:
                for i in range(nx):
                    if i < len(grid_config.y_neumann_values):
                        _, top_val = grid_config.y_neumann_values[i]
                        top_idx = ((ny - 1) * nx + i) * depth + 2
                        rhs[top_idx] = top_val
    
    def _build_vector_chunk_numpy(
        self,
        rhs: np.ndarray,
        values: np.ndarray,
        grid_config: Grid2DConfig,
        j_start: int,
        j_end: int,
        nx: int,
        ny: int,
        depth: int,
        use_x_dirichlet: bool,
        use_y_dirichlet: bool,
        use_x_neumann: bool,
        use_y_neumann: bool
    ) -> None:
        """
        右辺ベクトルのチャンクを構築（NumPy版）
        
        基本的にCuPy版と同じだが、NumPy配列を使用
        """
        # 実装はCuPy版と同じだが、NumPy配列を操作
        # 1. 関数値（f）の設定
        for j in range(j_start, j_end):
            for i in range(nx):
                idx = (j * nx + i) * depth
                rhs[idx] = values[i, j]
        
        # 2. 境界条件の設定
        # x方向ディリクレ境界条件（左右端）
        if use_x_dirichlet and grid_config.x_dirichlet_values is not None:
            for j in range(j_start, j_end):
                if j < len(grid_config.x_dirichlet_values):
                    left_val, right_val = grid_config.x_dirichlet_values[j]
                    
                    # 左端（i=0）の関数値
                    left_idx = (j * nx) * depth
                    rhs[left_idx] = left_val
                    
                    # 右端（i=nx-1）の関数値
                    right_idx = (j * nx + nx - 1) * depth
                    rhs[right_idx] = right_val
        
        # 残りの境界条件も同様に処理（CuPy版と同様の処理）
        # y方向ディリクレ境界条件（上下端）
        if use_y_dirichlet and grid_config.y_dirichlet_values is not None:
            # 下端（j=0）
            if j_start == 0:
                for i in range(nx):
                    if i < len(grid_config.y_dirichlet_values):
                        bottom_val, _ = grid_config.y_dirichlet_values[i]
                        bottom_idx = i * depth
                        rhs[bottom_idx] = bottom_val
            
            # 上端（j=ny-1）
            if j_end > ny - 1:
                for i in range(nx):
                    if i < len(grid_config.y_dirichlet_values):
                        _, top_val = grid_config.y_dirichlet_values[i]
                        top_idx = ((ny - 1) * nx + i) * depth
                        rhs[top_idx] = top_val
        
        # x方向ノイマン境界条件
        if use_x_neumann and grid_config.x_neumann_values is not None:
            for j in range(j_start, j_end):
                if j < len(grid_config.x_neumann_values):
                    left_val, right_val = grid_config.x_neumann_values[j]
                    
                    # 左端のx方向導関数
                    left_idx = (j * nx) * depth + 1
                    rhs[left_idx] = left_val
                    
                    # 右端のx方向導関数
                    right_idx = (j * nx + nx - 1) * depth + 1
                    rhs[right_idx] = right_val
        
        # y方向ノイマン境界条件
        if use_y_neumann and grid_config.y_neumann_values is not None:
            # 下端（j=0）
            if j_start == 0:
                for i in range(nx):
                    if i < len(grid_config.y_neumann_values):
                        bottom_val, _ = grid_config.y_neumann_values[i]
                        bottom_idx = i * depth + 2
                        rhs[bottom_idx] = bottom_val
            
            # 上端（j=ny-1）
            if j_end > ny - 1:
                for i in range(nx):
                    if i < len(grid_config.y_neumann_values):
                        _, top_val = grid_config.y_neumann_values[i]
                        top_idx = ((ny - 1) * nx + i) * depth + 2
                        rhs[top_idx] = top_val
    
    def _print_boundary_values(
        self,
        rhs: cp.ndarray,
        grid_config: Grid2DConfig,
        nx: int,
        ny: int,
        depth: int,
        use_x_dirichlet: bool,
        use_y_dirichlet: bool,
        use_x_neumann: bool,
        use_y_neumann: bool
    ) -> None:
        """デバッグ用に境界値を表示"""
        print("\n=== 境界値の確認 ===")
        
        # x方向ディリクレ境界条件
        if use_x_dirichlet:
            print("x方向ディリクレ境界値:")
            for j in range(min(3, ny)):  # 最初の数行のみ表示
                left_idx = (j * nx) * depth
                right_idx = (j * nx + nx - 1) * depth
                print(f"  行 j={j}: 左端={rhs[left_idx]}, 右端={rhs[right_idx]}")
        
        # y方向ディリクレ境界条件
        if use_y_dirichlet:
            print("y方向ディリクレ境界値:")
            for i in range(min(3, nx)):  # 最初の数列のみ表示
                bottom_idx = i * depth
                top_idx = ((ny - 1) * nx + i) * depth
                print(f"  列 i={i}: 下端={rhs[bottom_idx]}, 上端={rhs[top_idx]}")
    
    def get_memory_stats(self) -> Dict[str, Dict[str, float]]:
        """メモリ使用統計情報を取得"""
        return self.memory_stats