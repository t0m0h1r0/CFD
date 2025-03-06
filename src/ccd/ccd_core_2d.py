"""
2次元CCD法の中核モジュール

2次元CCD用のコンポーネントを簡単に作成する関数を提供します。
"""

from matrix_builder_2d import CCDLeftHandBuilder2D
from vector_builder_2d import CCDRightHandBuilder2D
from result_extractor_2d import CCDResultExtractor2D
from system_builder_2d import CCDSystemBuilder2D

# バージョン情報
VERSION = "1.0.0"


def create_system_builder_2d() -> CCDSystemBuilder2D:
    """
    2次元CCDシステムビルダーのインスタンスを作成

    Returns:
        構成済みのCCDSystemBuilder2Dインスタンス
    """
    return CCDSystemBuilder2D(
        CCDLeftHandBuilder2D(), CCDRightHandBuilder2D(), CCDResultExtractor2D()
    )