"""
高精度コンパクト差分法 (CCD) の結果可視化モジュール

このモジュールは、後方互換性のために、リファクタリングされた
可視化クラスからのインポートを提供します。
"""

# 基底クラス
from base_visualizer import BaseVisualizer

# 次元別実装
from visualizer1d import CCDVisualizer1D
from visualizer2d import CCDVisualizer2D

# 後方互換性用のエイリアス
CCDVisualizer = CCDVisualizer1D
CCD2DVisualizer = CCDVisualizer2D

# 後方互換性のためにすべてのクラスをエクスポート
__all__ = ["BaseVisualizer", "CCDVisualizer1D", "CCDVisualizer2D", "CCDVisualizer", "CCD2DVisualizer"]
