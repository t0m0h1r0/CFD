"""
2次元テスト関数モジュール

2次元CCDソルバーの評価に使用するテスト関数とその導関数を定義します。
"""

import cupy as cp
import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Dict, Tuple, Optional


@dataclass
class Test2DFunction:
    """2次元テスト関数とその導関数を保持するデータクラス"""

    name: str
    f: Callable[[cp.ndarray, cp.ndarray], cp.ndarray]  # 関数値: f(x, y)
    f_x: Callable[[cp.ndarray, cp.ndarray], cp.ndarray]  # x方向1階微分: ∂f/∂x
    f_y: Callable[[cp.ndarray, cp.ndarray], cp.ndarray]  # y方向1階微分: ∂f/∂y
    f_xx: Callable[[cp.ndarray, cp.ndarray], cp.ndarray]  # x方向2階微分: ∂²f/∂x²
    f_yy: Callable[[cp.ndarray, cp.ndarray], cp.ndarray]  # y方向2階微分: ∂²f/∂y²
    f_xy: Callable[[cp.ndarray, cp.ndarray], cp.ndarray]  # 混合微分: ∂²f/∂x∂y
    f_xxx: Optional[Callable[[cp.ndarray, cp.ndarray], cp.ndarray]] = None  # x方向3階微分: ∂³f/∂x³
    f_yyy: Optional[Callable[[cp.ndarray, cp.ndarray], cp.ndarray]] = None  # y方向3階微分: ∂³f/∂y³
    f_xxy: Optional[Callable[[cp.ndarray, cp.ndarray], cp.ndarray]] = None  # 混合微分: ∂³f/∂x²∂y
    f_xyy: Optional[Callable[[cp.ndarray, cp.ndarray], cp.ndarray]] = None  # 混合微分: ∂³f/∂x∂y²


class Test2DFunctionFactory:
    """2次元テスト関数を生成するファクトリークラス"""

    @staticmethod
    def create_standard_functions() -> List[Test2DFunction]:
        """標準的な2次元テスト関数セットを生成"""
        return [
            # 単純な多項式
            Test2DFunction(
                name="Poly2D",
                f=lambda x, y: 1.0 - x**2 - y**2,  # f(x,y) = 1 - x² - y²
                f_x=lambda x, y: -2 * x,  # ∂f/∂x = -2x
                f_y=lambda x, y: -2 * y,  # ∂f/∂y = -2y
                f_xx=lambda x, y: -2 * cp.ones_like(x),  # ∂²f/∂x² = -2
                f_yy=lambda x, y: -2 * cp.ones_like(y),  # ∂²f/∂y² = -2
                f_xy=lambda x, y: cp.zeros_like(x),  # ∂²f/∂x∂y = 0
                f_xxx=lambda x, y: cp.zeros_like(x),  # ∂³f/∂x³ = 0
                f_yyy=lambda x, y: cp.zeros_like(y),  # ∂³f/∂y³ = 0
            ),
            
            # 混合項を含む高次多項式
            Test2DFunction(
                name="MixedPoly",
                f=lambda x, y: x**2 * y**2 * (1 - x**2 - y**2),  # f(x,y) = x²y²(1-x²-y²)
                f_x=lambda x, y: 2 * x * y**2 * (1 - x**2 - y**2) - 2 * x**3 * y**2,  # ∂f/∂x
                f_y=lambda x, y: 2 * x**2 * y * (1 - x**2 - y**2) - 2 * x**2 * y**3,  # ∂f/∂y
                f_xx=lambda x, y: 2 * y**2 * (1 - x**2 - y**2) - 8 * x**2 * y**2 - 6 * x**2 * y**2,  # ∂²f/∂x²
                f_yy=lambda x, y: 2 * x**2 * (1 - x**2 - y**2) - 8 * x**2 * y**2 - 6 * x**2 * y**2,  # ∂²f/∂y²
                f_xy=lambda x, y: 4 * x * y * (1 - x**2 - y**2) - 4 * x**3 * y - 4 * x * y**3,  # ∂²f/∂x∂y
            ),
            
            # 三角関数
            Test2DFunction(
                name="Sine2D",
                f=lambda x, y: cp.sin(cp.pi * x) * cp.sin(cp.pi * y),  # f(x,y) = sin(πx)sin(πy)
                f_x=lambda x, y: cp.pi * cp.cos(cp.pi * x) * cp.sin(cp.pi * y),  # ∂f/∂x
                f_y=lambda x, y: cp.pi * cp.sin(cp.pi * x) * cp.cos(cp.pi * y),  # ∂f/∂y
                f_xx=lambda x, y: -cp.pi**2 * cp.sin(cp.pi * x) * cp.sin(cp.pi * y),  # ∂²f/∂x²
                f_yy=lambda x, y: -cp.pi**2 * cp.sin(cp.pi * x) * cp.sin(cp.pi * y),  # ∂²f/∂y²
                f_xy=lambda x, y: cp.pi**2 * cp.cos(cp.pi * x) * cp.cos(cp.pi * y),  # ∂²f/∂x∂y
                f_xxx=lambda x, y: -cp.pi**3 * cp.cos(cp.pi * x) * cp.sin(cp.pi * y),  # ∂³f/∂x³
                f_yyy=lambda x, y: -cp.pi**3 * cp.sin(cp.pi * x) * cp.cos(cp.pi * y),  # ∂³f/∂y³
            ),
            
            # ガウス関数
            Test2DFunction(
                name="Gaussian",
                f=lambda x, y: cp.exp(-(x**2 + y**2)),  # f(x,y) = exp(-(x²+y²))
                f_x=lambda x, y: -2 * x * cp.exp(-(x**2 + y**2)),  # ∂f/∂x
                f_y=lambda x, y: -2 * y * cp.exp(-(x**2 + y**2)),  # ∂f/∂y
                f_xx=lambda x, y: (-2 + 4 * x**2) * cp.exp(-(x**2 + y**2)),  # ∂²f/∂x²
                f_yy=lambda x, y: (-2 + 4 * y**2) * cp.exp(-(x**2 + y**2)),  # ∂²f/∂y²
                f_xy=lambda x, y: 4 * x * y * cp.exp(-(x**2 + y**2)),  # ∂²f/∂x∂y
            ),
            
            # 双曲線関数
            Test2DFunction(
                name="HyperbolicTangent",
                f=lambda x, y: cp.tanh(x * y),  # f(x,y) = tanh(xy)
                f_x=lambda x, y: y * (1 - cp.tanh(x * y)**2),  # ∂f/∂x
                f_y=lambda x, y: x * (1 - cp.tanh(x * y)**2),  # ∂f/∂y
                f_xx=lambda x, y: -2 * y**2 * cp.tanh(x * y) * (1 - cp.tanh(x * y)**2),  # ∂²f/∂x²
                f_yy=lambda x, y: -2 * x**2 * cp.tanh(x * y) * (1 - cp.tanh(x * y)**2),  # ∂²f/∂y²
                f_xy=lambda x, y: (1 - cp.tanh(x * y)**2) - 2 * x * y * cp.tanh(x * y) * (1 - cp.tanh(x * y)**2),  # ∂²f/∂x∂y
            ),
        ]

    @staticmethod
    def evaluate_function_on_grid(
        test_func: Test2DFunction, 
        grid_x: cp.ndarray, 
        grid_y: cp.ndarray
    ) -> Dict[str, cp.ndarray]:
        """
        グリッド上でテスト関数とその導関数を評価
        
        Args:
            test_func: 評価するテスト関数
            grid_x: xグリッド座標
            grid_y: yグリッド座標
            
        Returns:
            {"f": 関数値, "f_x": x方向1階微分, ...} の形式の辞書
        """
        results = {}
        
        # 関数値
        results["f"] = test_func.f(grid_x, grid_y)
        
        # 導関数
        results["f_x"] = test_func.f_x(grid_x, grid_y)
        results["f_y"] = test_func.f_y(grid_x, grid_y)
        results["f_xx"] = test_func.f_xx(grid_x, grid_y)
        results["f_yy"] = test_func.f_yy(grid_x, grid_y)
        results["f_xy"] = test_func.f_xy(grid_x, grid_y)
        
        # 3階導関数があれば評価
        if test_func.f_xxx is not None:
            results["f_xxx"] = test_func.f_xxx(grid_x, grid_y)
        if test_func.f_yyy is not None:
            results["f_yyy"] = test_func.f_yyy(grid_x, grid_y)
        if test_func.f_xxy is not None:
            results["f_xxy"] = test_func.f_xxy(grid_x, grid_y)
        if test_func.f_xyy is not None:
            results["f_xyy"] = test_func.f_xyy(grid_x, grid_y)
            
        return results

    @staticmethod
    def calculate_error_metrics(
        numerical: Dict[str, cp.ndarray], 
        analytical: Dict[str, cp.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """
        数値解と解析解の誤差メトリクスを計算
        
        Args:
            numerical: 数値解の辞書
            analytical: 解析解の辞書
            
        Returns:
            {"f_x": {"l2": L2誤差, "max": 最大誤差}, ...} の形式の2階層辞書
        """
        metrics = {}
        
        # 共通のキーでループ
        for key in set(numerical.keys()) & set(analytical.keys()):
            num = numerical[key]
            ana = analytical[key]
            
            # 誤差計算
            if num.shape == ana.shape:
                diff = num - ana
                l2_error = float(cp.sqrt(cp.mean(diff**2)))
                max_error = float(cp.max(cp.abs(diff)))
                relative_error = float(cp.mean(cp.abs(diff) / (cp.abs(ana) + 1e-10)))
                
                metrics[key] = {
                    "l2": l2_error,
                    "max": max_error,
                    "relative": relative_error
                }
            
        return metrics
