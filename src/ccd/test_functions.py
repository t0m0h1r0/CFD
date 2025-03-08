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
                name="Zero",
                f=lambda x: 0.0,  # すでに両端でゼロ
                df=lambda x: 0.0,
                d2f=lambda x: 0.0,
                d3f=lambda x: 0.0,
            ),
            TestFunction(
                name="QuadPoly",
                f=lambda x: (1 - x**2),  # シンプルな2次関数
                df=lambda x: -2 * x,
                d2f=lambda x: -2,
                d3f=lambda x: 0,
            ),
            TestFunction(
                name="CubicPoly",
                f=lambda x: (1 - x) * (1 + x) * (x + 0.5),  # 3次関数
                df=lambda x: -(2 * x) * (x + 0.5) + (1 - x**2),
                d2f=lambda x: -2 * (x + 0.5) - 4 * x,
                d3f=lambda x: -6,
            ),
            TestFunction(
                name="Sine",
                f=lambda x: cp.sin(cp.pi * x),  # 両端でゼロ
                df=lambda x: cp.pi * cp.cos(cp.pi * x),
                d2f=lambda x: -(cp.pi**2) * cp.sin(cp.pi * x),
                d3f=lambda x: -(cp.pi**3) * cp.cos(cp.pi * x),
            ),
            TestFunction(
                name="Cosine",
                f=lambda x: cp.cos(2 * cp.pi * x),  # 平行移動で両端でゼロ
                df=lambda x: -2 * cp.pi * cp.sin(2 * cp.pi * x),
                d2f=lambda x: -4 * (cp.pi**2) * cp.cos(2 * cp.pi * x),
                d3f=lambda x: 8 * cp.pi**3 * cp.sin(2 * cp.pi * x),
            ),
            TestFunction(
                name="ExpMod",
                f=lambda x: cp.exp(-(x**2)) - cp.exp(-1),  # 平行移動で両端でゼロ
                df=lambda x: -2 * x * cp.exp(-(x**2)),
                d2f=lambda x: (-2 + 4 * x**2) * cp.exp(-(x**2)),
                d3f=lambda x: (12 * x - 8 * x**3) * cp.exp(-(x**2)),
            ),
            TestFunction(
                name="HigherPoly",
                f=lambda x: x**4 - x**2,  # 4次関数
                df=lambda x: 4 * x**3 - 2 * x,
                d2f=lambda x: 12 * x**2 - 2,
                d3f=lambda x: 24 * x,
            ),
            TestFunction(
                name="CompoundPoly",
                f=lambda x: x**2 * (1 - x**2),  # 両端でゼロの4次関数
                df=lambda x: 2 * x * (1 - x**2) - 2 * x**3,  # = 2x - 4x^3
                d2f=lambda x: 2 - 12 * x**2,  # 修正: 正しい2階導関数
                d3f=lambda x: -24 * x,  # 修正: 正しい3階導関数
            ),
            TestFunction(
                name="Runge",
                f=lambda x: 1 / (1 + 25 * x**2),  # Runge関数 (急峻な変化)
                df=lambda x: -50 * x / (1 + 25 * x**2) ** 2,
                d2f=lambda x: (50 * (75 * x**2 - 1)) / (1 + 25 * x**2) ** 3,
                d3f=lambda x: (50 * (1 - 25 * x**2) * (3750 * x**2 - 150))
                / (1 + 25 * x**2) ** 4,
            ),
            TestFunction(
                name="ModifiedExp",
                f=lambda x: cp.exp(-10 * cp.abs(x)),  # 指数関数 (0で導関数が不連続)
                df=lambda x: -10 * cp.sign(x) * cp.exp(-10 * cp.abs(x)),
                d2f=lambda x: 100 * cp.exp(-10 * cp.abs(x)),
                d3f=lambda x: -1000 * cp.sign(x) * cp.exp(-10 * cp.abs(x)),
            ),
            TestFunction(
                name="HighFreqSine",
                f=lambda x: cp.sin(20 * cp.pi * x),  # 高周波正弦波
                df=lambda x: 20 * cp.pi * cp.cos(20 * cp.pi * x),
                d2f=lambda x: -((20 * cp.pi) ** 2) * cp.sin(20 * cp.pi * x),
                d3f=lambda x: -((20 * cp.pi) ** 3) * cp.cos(20 * cp.pi * x),
            ),
        ]