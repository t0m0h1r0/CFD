"""
テスト関数の定義モジュール

このモジュールでは、CCD法の評価に使用するテスト関数とその導関数を定義します。
"""

import jax.numpy as jnp
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
        """標準的なテスト関数セットを生成"""
        return [
            TestFunction(
                name="Zero",
                f=lambda x: 0.0,  # すでに両端でゼロ
                df=lambda x: 0.0,
                d2f=lambda x: 0.0,
                d3f=lambda x: 0.0
            ),
            TestFunction(
                name="QuadPoly",
                f=lambda x: (1 - x**2),  # シンプルな2次関数
                df=lambda x: -2*x,
                d2f=lambda x: -2,
                d3f=lambda x: 0
            ),
            TestFunction(
                name="CubicPoly",
                f=lambda x: (1 - x)*(1 + x)*(x + 0.5),  # 3次関数
                df=lambda x: -(2*x)*(x + 0.5) + (1 - x**2),
                d2f=lambda x: -2*(x + 0.5) - 4*x,
                d3f=lambda x: -6
            ),
            TestFunction(
                name="Sine",
                f=lambda x: jnp.sin(jnp.pi*x),  # 両端でゼロ
                df=lambda x: jnp.pi*jnp.cos(jnp.pi*x),
                d2f=lambda x: -(jnp.pi**2)*jnp.sin(jnp.pi*x),
                d3f=lambda x: -(jnp.pi**3)*jnp.cos(jnp.pi*x)
            ),
            TestFunction(
                name="Cosine",
                f=lambda x: jnp.cos(2*jnp.pi*x),  # 平行移動で両端でゼロ
                df=lambda x: -2*jnp.pi*jnp.sin(2*jnp.pi*x),
                d2f=lambda x: -4*(jnp.pi**2)*jnp.cos(2*jnp.pi*x),
                d3f=lambda x: 8*jnp.pi**3*jnp.sin(2*jnp.pi*x)
            ),
            TestFunction(
                name="ExpMod",
                f=lambda x: jnp.exp(-x**2) - jnp.exp(-1),  # 平行移動で両端でゼロ
                df=lambda x: -2*x*jnp.exp(-x**2),
                d2f=lambda x: (-2 + 4*x**2)*jnp.exp(-x**2),
                d3f=lambda x: (12*x - 8*x**3)*jnp.exp(-x**2)
            ),
            TestFunction(
                name="HigherPoly",
                f=lambda x: x**4 - x**2,  # 4次関数
                df=lambda x: 4*x**3 - 2*x,
                d2f=lambda x: 12*x**2 - 2,
                d3f=lambda x: 24*x
            ),
            TestFunction(
                name="CompoundPoly",
                f=lambda x: x**2 * (1 - x**2),  # 両端でゼロの4次関数
                df=lambda x: 2*x*(1 - x**2) - 2*x**3,
                d2f=lambda x: 2*(1 - x**2) - 8*x**2,
                d3f=lambda x: -12*x
            )
        ]

    @staticmethod
    def create_challenging_functions() -> List[TestFunction]:
        """数値的に挑戦的なテスト関数セットを生成"""
        return [
            TestFunction(
                name="Runge",
                f=lambda x: 1 / (1 + 25 * x**2),  # Runge関数 (急峻な変化)
                df=lambda x: -50 * x / (1 + 25 * x**2)**2,
                d2f=lambda x: (50 * (75 * x**2 - 1)) / (1 + 25 * x**2)**3,
                d3f=lambda x: (50 * (1 - 25 * x**2) * (3750 * x**2 - 150)) / (1 + 25 * x**2)**4
            ),
            TestFunction(
                name="ModifiedExp",
                f=lambda x: jnp.exp(-10 * jnp.abs(x)),  # 指数関数 (0で導関数が不連続)
                df=lambda x: -10 * jnp.sign(x) * jnp.exp(-10 * jnp.abs(x)),
                d2f=lambda x: 100 * jnp.exp(-10 * jnp.abs(x)),
                d3f=lambda x: -1000 * jnp.sign(x) * jnp.exp(-10 * jnp.abs(x))
            ),
            TestFunction(
                name="HighFreqSine",
                f=lambda x: jnp.sin(20 * jnp.pi * x),  # 高周波正弦波
                df=lambda x: 20 * jnp.pi * jnp.cos(20 * jnp.pi * x),
                d2f=lambda x: -(20 * jnp.pi)**2 * jnp.sin(20 * jnp.pi * x),
                d3f=lambda x: -(20 * jnp.pi)**3 * jnp.cos(20 * jnp.pi * x)
            )
        ]