# test_functions.py
import cupy as cp  # NumPyではなくCuPyを使用
from dataclasses import dataclass
from typing import Callable, List

@dataclass
class TestFunction:
    """テスト関数とその導関数を保持するデータクラス"""
    
    name: str
    f: Callable[[float], float]
    df: Callable[[float], float]
    d2f: Callable[[float], float]
    d3f: Callable[[float], float]

class TestFunctionFactory:
    """テスト関数を生成するファクトリークラス"""
    
    @staticmethod
    def create_standard_functions() -> List[TestFunction]:
        """標準的なテスト関数セットを作成"""
        return [
            TestFunction(
                name="Sin",
                f=lambda x: cp.sin(cp.pi * x),
                df=lambda x: cp.pi * cp.cos(cp.pi * x),
                d2f=lambda x: -(cp.pi**2) * cp.sin(cp.pi * x),
                d3f=lambda x: -(cp.pi**3) * cp.cos(cp.pi * x),
            ),
            TestFunction(
                name="Cosine",
                f=lambda x: cp.cos(2 * cp.pi * x),
                df=lambda x: -2 * cp.pi * cp.sin(2 * cp.pi * x),
                d2f=lambda x: -4 * (cp.pi**2) * cp.cos(2 * cp.pi * x),
                d3f=lambda x: 8 * (cp.pi**3) * cp.sin(2 * cp.pi * x),
            ),
            TestFunction(
                name="Polynomial",
                f=lambda x: (1 - x**2),
                df=lambda x: -2 * x,
                d2f=lambda x: -2,
                d3f=lambda x: 0,
            ),
            TestFunction(
                name="CubicPoly",
                f=lambda x: (1 - x) * (1 + x) * (x + 0.5),
                df=lambda x: -(2 * x) * (x + 0.5) + (1 - x**2),
                d2f=lambda x: -2 * (x + 0.5) - 4 * x,
                d3f=lambda x: -6,
            ),
            TestFunction(
                name="ExpMod",
                f=lambda x: cp.exp(-(x**2)) - cp.exp(-1),
                df=lambda x: -2 * x * cp.exp(-(x**2)),
                d2f=lambda x: (-2 + 4 * x**2) * cp.exp(-(x**2)),
                d3f=lambda x: (12 * x - 8 * x**3) * cp.exp(-(x**2)),
            ),
            # 追加テスト関数
            TestFunction(
                name="HigherPoly",
                f=lambda x: x**4 - x**2,
                df=lambda x: 4 * x**3 - 2 * x,
                d2f=lambda x: 12 * x**2 - 2,
                d3f=lambda x: 24 * x,
            ),
        ]