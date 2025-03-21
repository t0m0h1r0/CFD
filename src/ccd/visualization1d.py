"""
高精度コンパクト差分法 (CCD) の1次元結果可視化モジュール (レガシー)

このモジュールは、後方互換性のために、リファクタリングされた
1次元可視化クラスからのインポートを提供します。
"""

# 新しいモジュールからインポート
from visualizer1d import CCDVisualizer1D

# 後方互換性のためのクラス名エイリアス
CCDVisualizer = CCDVisualizer1D

# 後方互換性のためにエクスポート
__all__ = ["CCDVisualizer"]