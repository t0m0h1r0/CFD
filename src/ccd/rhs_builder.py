"""
高精度コンパクト差分法 (CCD) 用の右辺ベクトル構築モジュール

このモジュールは、ポアソン方程式および高階微分方程式のための
右辺ベクトルを効率的に構築するクラス群を提供します。
後方互換性のために、リファクタリングされたクラスからのインポートを提供します。
"""

# 基底クラス
from base_rhs_builder import RHSBuilder

# 次元別実装
from rhs_builder1d import RHSBuilder1D
from rhs_builder2d import RHSBuilder2D

# 後方互換性のためにすべてのクラスをエクスポート
__all__ = ["RHSBuilder", "RHSBuilder1D", "RHSBuilder2D"]