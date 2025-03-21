"""
可視化基底クラスモジュール

このモジュールは、CCDソルバーの結果可視化のための基底クラスを定義します。
"""

import os
import abc
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Union, Tuple, Optional, Any

class BaseVisualizer(abc.ABC):
    """
    CCDソルバーの結果を可視化するための基底クラス
    
    各次元の具象クラスは、この基底クラスを継承して実装されます。
    """
    
    def __init__(self, output_dir: str = "results"):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリパス
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.min_log_value = 1e-16  # 対数スケール用の最小値
    
    def generate_filename(self, func_name: str, prefix: str = "", **kwargs) -> str:
        """
        保存ファイル名を生成
        
        Args:
            func_name: 関数名
            prefix: ファイル名の接頭辞
            **kwargs: その他のパラメータ（具象クラスで使用）
            
        Returns:
            ファイルパス
        """
        # 基本実装（具象クラスでオーバーライドする）
        if prefix:
            return f"{self.output_dir}/{prefix}_{func_name.lower()}.png"
        else:
            return f"{self.output_dir}/{func_name.lower()}.png"
    
    def _to_numpy(self, x: Any) -> np.ndarray:
        """
        配列をNumPy配列に変換（CuPy, JAX対応）
        
        Args:
            x: 変換する配列
            
        Returns:
            NumPy配列
        """
        if x is None:
            return None
        if hasattr(x, "get"):  # CuPy
            return x.get()
        elif hasattr(x, "device"):  # JAX/PyTorch
            return np.array(x)
        return x  # Already NumPy
    
    def _create_error_summary(self, 
                             function_name: str, 
                             solution_names: List[str], 
                             errors: List[float],
                             prefix: str = "", 
                             save: bool = True, 
                             show: bool = False, 
                             dpi: int = 150) -> str:
        """
        誤差のサマリーグラフを作成
        
        Args:
            function_name: テスト関数の名前
            solution_names: 解コンポーネントの名前リスト
            errors: 誤差のリスト
            prefix: ファイル名の接頭辞
            save: 図を保存するかどうか
            show: 図を表示するかどうか
            dpi: 保存する図のDPI
            
        Returns:
            保存したファイルのパス
        """
        # グラフ作成
        fig, ax = plt.subplots(figsize=(10, 6))
        x_pos = np.arange(len(solution_names))
        bars = ax.bar(x_pos, errors)
        
        # 対数スケール（ゼロを小さな値に置き換え）
        error_values = errors.copy()
        for i, err in enumerate(error_values):
            if err == 0:
                error_values[i] = self.min_log_value
        
        # プロット設定
        ax.set_yscale('log')
        ax.set_title(f"{function_name} Function: Error Summary")
        ax.set_xlabel('Solution Component')
        ax.set_ylabel('Maximum Error (log scale)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(solution_names)
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        
        # バーにラベルを追加
        for bar, err in zip(bars, errors):
            height = bar.get_height()
            ax.annotate(f'{err:.2e}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        plt.tight_layout()
        
        # ファイル保存
        filepath = ""
        if save:
            filepath = self.generate_filename(f"{function_name}_summary", prefix)
            plt.savefig(filepath, dpi=dpi)
        
        # 表示/クローズ
        if show:
            plt.show()
        else:
            plt.close(fig)
            
        return filepath
    
    def compare_all_functions_errors(self, 
                                   results_summary: Dict[str, List[float]], 
                                   grid_size: Union[int, Tuple[int, ...]], 
                                   prefix: str = "", 
                                   dpi: int = 150, 
                                   show: bool = False) -> str:
        """
        すべてのテスト関数の誤差を比較するグラフを生成（具象クラスでオーバーライド）
        
        Args:
            results_summary: テスト関数ごとの誤差リスト
            grid_size: グリッドサイズ
            prefix: ファイル名の接頭辞
            dpi: 保存する図のDPI
            show: 図を表示するかどうか
            
        Returns:
            保存したファイルのパス
        """
        raise NotImplementedError("具象クラスで実装する必要があります")
    
    @abc.abstractmethod
    def visualize_solution(self, 
                         grid, 
                         function_name: str, 
                         numerical: List[np.ndarray], 
                         exact: List[np.ndarray], 
                         errors: List[float], 
                         prefix: str = "", 
                         save: bool = True, 
                         show: bool = False, 
                         dpi: int = 150) -> bool:
        """
        解を可視化（具象クラスで実装）
        
        Args:
            grid: グリッドオブジェクト
            function_name: テスト関数の名前
            numerical: 数値解のリスト
            exact: 厳密解のリスト
            errors: 誤差のリスト
            prefix: ファイル名の接頭辞
            save: 図を保存するかどうか
            show: 図を表示するかどうか
            dpi: 保存する図のDPI
            
        Returns:
            成功したかどうかのブール値
        """
        pass
    
    @abc.abstractmethod
    def visualize_grid_convergence(self, 
                                 function_name: str, 
                                 grid_sizes: List[int], 
                                 results: Dict[int, List[float]], 
                                 prefix: str = "", 
                                 save: bool = True, 
                                 show: bool = False, 
                                 dpi: int = 150) -> bool:
        """
        グリッド収束性のグラフを生成（具象クラスで実装）
        
        Args:
            function_name: テスト関数の名前
            grid_sizes: グリッドサイズのリスト
            results: グリッドサイズごとの誤差リスト
            prefix: ファイル名の接頭辞
            save: 図を保存するかどうか
            show: 図を表示するかどうか
            dpi: 保存する図のDPI
            
        Returns:
            成功したかどうかのブール値
        """
        pass
