"""
高精度コンパクト差分法 (CCD) を用いた偏微分方程式ソルバーモジュール

このモジュールは、ポアソン方程式および高階微分方程式を1次元・2次元で
解くためのソルバークラスを提供します。後方互換性のために、リファクタリングされた
ソルバーからのインポートを提供します。
"""

# 新しいモジュールからインポート
from base_solver import BaseCCDSolver
from solver1d import CCDSolver1D
from solver2d import CCDSolver2D

# 後方互換性のためにすべてのクラスをエクスポート
__all__ = ["BaseCCDSolver", "CCDSolver1D", "CCDSolver2D"]