"""
CCD法の中核モジュール

各種モジュールをまとめて、使いやすいインターフェースを提供します。
このモジュールはリファクタリング後のファイルをインポートするためのラッパーです。
"""

# 後方互換性のためのインポート
from matrix_builder import CCDLeftHandBuilder
from vector_builder import CCDRightHandBuilder
from result_extractor import CCDResultExtractor
from system_builder import CCDSystemBuilder

# 将来的に必要になる可能性のある定数や関数もここで定義できる
VERSION = "1.0.0"


def create_system_builder() -> CCDSystemBuilder:
    """
    CCDシステムビルダーのインスタンスを作成

    Returns:
        構成済みのCCDSystemBuilderインスタンス
    """
    return CCDSystemBuilder(
        CCDLeftHandBuilder(),
        CCDRightHandBuilder(),
        CCDResultExtractor()
    )
