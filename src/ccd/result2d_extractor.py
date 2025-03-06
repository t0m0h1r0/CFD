"""
2次元CCD結果抽出モジュール（メモリ最適化版）

2次元CCD法の解ベクトルから各成分を抽出するクラスを提供します。
効率的なメモリ使用のために最適化されています。
"""

import cupy as cp
import numpy as np
from typing import Dict, Optional, List, Any, Tuple
import gc
import time
import traceback

from grid2d_config import Grid2DConfig


class CCD2DResultExtractor:
    """2次元CCDソルバーの結果から各成分を抽出するクラス（CuPy対応、メモリ最適化版）"""

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

    def extract_components(
        self, grid_config: Grid2DConfig, solution: cp.ndarray
    ) -> Dict[str, cp.ndarray]:
        """
        解ベクトルから関数値と各階導関数を抽出
        チャンク処理と効率的なメモリ管理で最適化されています。

        Args:
            grid_config: 2次元グリッド設定
            solution: 解ベクトル

        Returns:
            {"f": 関数値, "f_x": x偏導関数, ... } の形式の辞書
        """
        start_time = time.time()
        self._log_memory_usage("開始")
        
        nx, ny = grid_config.nx, grid_config.ny
        depth = 4  # 各点の未知数の数（f, f_x, f_y, f_xx）
        expected_size = nx * ny * depth
        
        # デバッグ情報
        print(f"結果抽出開始 - 解ベクトルのサイズ: {solution.shape}")
        print(f"期待されるサイズ: {expected_size}")
        
        # 期待されるサイズよりも解ベクトルが小さい場合の対処
        if solution.shape[0] < expected_size:
            print("警告: 解ベクトルが期待されるサイズよりも小さいです。結果は不完全である可能性があります。")
        
        # 結果を格納する辞書を準備
        results = {}
        
        # 抽出する導関数のタイプと対応するインデックス
        deriv_types = {
            "f": 0,     # 関数値
            "f_x": 1,   # x方向の1階導関数
            "f_y": 2,   # y方向の1階導関数
            "f_xx": 3,  # x方向の2階導関数
        }
        
        # メモリ効率のために、一度に一つの導関数タイプを抽出
        for deriv_type, idx_offset in deriv_types.items():
            print(f"{deriv_type} の抽出中...")
            
            try:
                # チャンク処理でメモリ使用量を抑制
                result_array = self._extract_component_chunks(
                    solution, deriv_type, idx_offset, nx, ny, depth
                )
                
                # 結果を辞書に格納
                results[deriv_type] = result_array
                
            except Exception as e:
                print(f"警告: {deriv_type} の抽出中にエラーが発生しました: {e}")
                traceback.print_exc()
                
                # エラーが発生した場合は代替方法を試す
                try:
                    print(f"{deriv_type} の抽出を代替方法で再試行...")
                    
                    # NumPy配列として作成してからCuPyに変換（メモリ効率化）
                    result_array = self._extract_component_chunks_numpy(
                        solution, deriv_type, idx_offset, nx, ny, depth
                    )
                    
                    # CuPy配列に変換
                    results[deriv_type] = cp.array(result_array)
                    
                    # NumPy配列を削除
                    del result_array
                    gc.collect()
                    
                except Exception as e2:
                    print(f"代替方法も失敗: {e2}")
                    traceback.print_exc()
                    
                    # 最後の手段として0の配列を生成
                    results[deriv_type] = cp.zeros((nx, ny), dtype=cp.float64)
                    print(f"{deriv_type} は0で初期化されました")
                
            # 各イテレーション後にメモリ管理
            if deriv_type != list(deriv_types.keys())[-1]:  # 最後の要素でない場合
                gc.collect()
                self._log_memory_usage(f"{deriv_type} 抽出後")
        
        # 関数値に対して境界補正を適用
        if grid_config.enable_boundary_correction and "f" in results:
            print("境界補正を適用中...")
            results["f"] = grid_config.apply_boundary_correction(results["f"])
        
        # 混合偏導関数の近似計算（もし必要であれば）
        if "f_x" in results and "f_y" in results and "f_xy" not in results:
            try:
                print("混合偏導関数 f_xy を近似計算中...")
                f_xy = self._approximate_mixed_derivative(
                    results["f_x"], results["f_y"], grid_config
                )
                results["f_xy"] = f_xy
            except Exception as e:
                print(f"混合偏導関数の計算に失敗: {e}")
                traceback.print_exc()
        
        elapsed_time = time.time() - start_time
        print(f"結果抽出完了 - 経過時間: {elapsed_time:.2f}秒")
        self._log_memory_usage("完了")
        
        return results

    def _extract_component_chunks(
        self,
        solution: cp.ndarray,
        deriv_type: str,
        idx_offset: int,
        nx: int,
        ny: int,
        depth: int
    ) -> cp.ndarray:
        """
        解ベクトルからチャンク単位で特定の導関数成分を抽出（CuPy版）

        Args:
            solution: 解ベクトル
            deriv_type: 導関数タイプ
            idx_offset: インデックスオフセット
            nx, ny: グリッドサイズ
            depth: 各点の未知数の数

        Returns:
            抽出された導関数の2次元配列
        """
        # 結果を格納する2次元配列
        result_array = cp.zeros((nx, ny), dtype=cp.float64)
        
        # チャンク処理（メモリ使用量を抑制）
        chunk_size = min(50, ny)  # y方向の分割サイズ
        
        for j_start in range(0, ny, chunk_size):
            j_end = min(j_start + chunk_size, ny)
            # print(f"{deriv_type} 抽出チャンク: y={j_start}～{j_end-1}")
            
            # チャンク内の各点の値を抽出
            for j in range(j_start, j_end):
                for i in range(nx):
                    # 平坦化されたインデックスを計算
                    linear_idx = (j * nx + i) * depth + idx_offset
                    
                    # インデックスが有効範囲内かチェック
                    if linear_idx < solution.shape[0]:
                        result_array[i, j] = solution[linear_idx]
        
        return result_array

    def _extract_component_chunks_numpy(
        self,
        solution: cp.ndarray,
        deriv_type: str,
        idx_offset: int,
        nx: int,
        ny: int,
        depth: int
    ) -> np.ndarray:
        """
        解ベクトルからチャンク単位で特定の導関数成分を抽出（NumPy版）

        Args:
            solution: 解ベクトル（CuPy配列）
            deriv_type: 導関数タイプ
            idx_offset: インデックスオフセット
            nx, ny: グリッドサイズ
            depth: 各点の未知数の数

        Returns:
            抽出された導関数の2次元NumPy配列
        """
        # 結果を格納するNumPy配列
        result_array = np.zeros((nx, ny), dtype=np.float64)
        
        # 解ベクトルをNumPy配列に変換
        solution_np = cp.asnumpy(solution)
        
        # チャンク処理
        chunk_size = min(50, ny)
        
        for j_start in range(0, ny, chunk_size):
            j_end = min(j_start + chunk_size, ny)
            
            # チャンク内の各点の値を抽出
            for j in range(j_start, j_end):
                for i in range(nx):
                    # 平坦化されたインデックスを計算
                    linear_idx = (j * nx + i) * depth + idx_offset
                    
                    # インデックスが有効範囲内かチェック
                    if linear_idx < solution_np.shape[0]:
                        result_array[i, j] = solution_np[linear_idx]
        
        # NumPy配列のメモリ管理
        del solution_np
        
        return result_array

    def _approximate_mixed_derivative(
        self,
        f_x: cp.ndarray,
        f_y: cp.ndarray,
        grid_config: Grid2DConfig
    ) -> cp.ndarray:
        """
        混合偏導関数 f_xy を近似計算

        Args:
            f_x: x方向1階導関数の2次元配列
            f_y: y方向1階導関数の2次元配列
            grid_config: グリッド設定

        Returns:
            混合偏導関数 f_xy の2次元配列
        """
        nx, ny = grid_config.nx, grid_config.ny
        hx, hy = grid_config.hx, grid_config.hy
        
        # 結果を格納する配列
        f_xy = cp.zeros((nx, ny), dtype=cp.float64)
        
        # 内部点の混合偏導関数を中心差分で計算
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                # df_x/dy を中心差分で計算
                f_xy[i, j] = (f_x[i, j+1] - f_x[i, j-1]) / (2 * hy)
                
                # または、df_y/dx を中心差分で計算（同等）
                # f_xy[i, j] = (f_y[i+1, j] - f_y[i-1, j]) / (2 * hx)
        
        # 境界点の混合偏導関数を前方差分または後方差分で計算
        # 左端と右端
        for j in range(1, ny-1):
            # 左端 (i=0)
            f_xy[0, j] = (f_x[0, j+1] - f_x[0, j-1]) / (2 * hy)
            
            # 右端 (i=nx-1)
            f_xy[nx-1, j] = (f_x[nx-1, j+1] - f_x[nx-1, j-1]) / (2 * hy)
        
        # 下端と上端
        for i in range(1, nx-1):
            # 下端 (j=0)
            f_xy[i, 0] = (f_y[i+1, 0] - f_y[i-1, 0]) / (2 * hx)
            
            # 上端 (j=ny-1)
            f_xy[i, ny-1] = (f_y[i+1, ny-1] - f_y[i-1, ny-1]) / (2 * hx)
        
        # 四隅の点は隣接点の平均で近似
        # 左下 (i=0, j=0)
        f_xy[0, 0] = (f_xy[1, 0] + f_xy[0, 1]) / 2
        
        # 右下 (i=nx-1, j=0)
        f_xy[nx-1, 0] = (f_xy[nx-2, 0] + f_xy[nx-1, 1]) / 2
        
        # 左上 (i=0, j=ny-1)
        f_xy[0, ny-1] = (f_xy[1, ny-1] + f_xy[0, ny-2]) / 2
        
        # 右上 (i=nx-1, j=ny-1)
        f_xy[nx-1, ny-1] = (f_xy[nx-2, ny-1] + f_xy[nx-1, ny-2]) / 2
        
        return f_xy

    def extract_at_point(
        self, grid_config: Grid2DConfig, solution: cp.ndarray, i: int, j: int
    ) -> Dict[str, float]:
        """
        特定の格子点における解ベクトルから関数値と各階導関数を抽出

        Args:
            grid_config: 2次元グリッド設定
            solution: 解ベクトル
            i: x方向のインデックス
            j: y方向のインデックス

        Returns:
            {"f": 関数値, "f_x": x偏導関数, ... } の形式の辞書
        """
        nx, ny = grid_config.nx, grid_config.ny
        depth = 4  # 各点の未知数の数
        
        # インデックスが有効範囲内かチェック
        if i < 0 or i >= nx or j < 0 or j >= ny:
            raise ValueError(f"インデックス ({i}, {j}) が範囲外です (範囲: 0～{nx-1}, 0～{ny-1})")
        
        # 結果を格納する辞書
        results = {}

        # 抽出する導関数のタイプと対応するインデックス
        deriv_types = {
            "f": 0,     # 関数値
            "f_x": 1,   # x方向の1階導関数
            "f_y": 2,   # y方向の1階導関数
            "f_xx": 3,  # x方向の2階導関数
        }

        # 各導関数タイプの値を抽出
        for deriv_type, idx_offset in deriv_types.items():
            try:
                # 平坦化されたインデックスを計算
                linear_idx = (j * nx + i) * depth + idx_offset
                if linear_idx < solution.shape[0]:
                    results[deriv_type] = float(solution[linear_idx])
                else:
                    results[deriv_type] = 0.0
            except Exception as e:
                print(f"警告: 点 ({i}, {j}) における {deriv_type} の抽出中にエラーが発生しました: {e}")
                results[deriv_type] = 0.0

        return results
    
    def get_memory_stats(self) -> Dict[str, Dict[str, float]]:
        """メモリ使用統計情報を取得"""
        return self.memory_stats