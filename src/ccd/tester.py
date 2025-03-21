"""
高精度コンパクト差分法 (CCD) テスターモジュール

このモジュールは、ポアソン方程式および高階微分方程式を1次元・2次元で
テストするためのクラスを提供します。後方互換性のために、リファクタリングされた
テスターからのインポートを提供します。
"""

# 新しいモジュールからインポート
from base_tester import CCDTester
from tester1d import CCDTester1D
from tester2d import CCDTester2D

# 後方互換性のためにすべてのクラスをエクスポート
__all__ = ["CCDTester", "CCDTester1D", "CCDTester2D"]