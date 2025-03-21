"""
高精度コンパクト差分法 (CCD) の2次元結果可視化モジュール (レガシー)

このモジュールは、後方互換性のために、リファクタリングされた
2次元可視化クラスからのインポートを提供します。
"""

# 新しいモジュールからインポート
from visualizer2d import CCDVisualizer2D

# 後方互換性のためのクラス名エイリアス
CCD2DVisualizer = CCDVisualizer2D

# 後方互換性のためにエクスポート
__all__ = ["CCD2DVisualizer"]