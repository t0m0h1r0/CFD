"""
プラグインローダー

スケーリングと正則化のプラグインを一括でロードする機能を提供します。
"""

import os
from typing import List, Optional, Dict, Any

from scaling_strategy import scaling_registry
from regularization_strategy import regularization_registry


class PluginLoader:
    """
    プラグインローダー

    スケーリングと正則化のプラグインを検出・登録します
    """

    _plugins_loaded = False

    @classmethod
    def load_plugins(
        cls, plugin_dirs: Optional[List[str]] = None, verbose: bool = True
    ) -> None:
        """
        プラグインを読み込む

        Args:
            plugin_dirs: プラグインディレクトリのリスト（Noneの場合はデフォルト）
            verbose: 詳細なログを出力するかどうか
        """
        if cls._plugins_loaded:
            if verbose:
                print("プラグインは既にロード済みです")
            return

        # ログ出力設定
        scaling_registry.verbose = verbose
        regularization_registry.verbose = verbose

        # プロジェクトのルートディレクトリを検出
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # デフォルトディレクトリを設定
        if plugin_dirs is None:
            plugin_dirs = [
                os.path.join(current_dir, "scaling"),
                os.path.join(current_dir, "regularization"),
            ]

        # 各ディレクトリからプラグインをスキャン
        for directory in plugin_dirs:
            if os.path.exists(directory) and os.path.isdir(directory):
                if "scaling" in os.path.basename(directory).lower():
                    scaling_registry.scan_directory(directory)
                elif "regularization" in os.path.basename(directory).lower():
                    regularization_registry.scan_directory(directory)

        cls._plugins_loaded = True

    @classmethod
    def available_scaling_methods(cls) -> List[str]:
        """
        利用可能なスケーリング手法のリストを返す

        Returns:
            スケーリング手法名のリスト
        """
        if not cls._plugins_loaded:
            cls.load_plugins(verbose=False)
        return scaling_registry.get_names()

    @classmethod
    def available_regularization_methods(cls) -> List[str]:
        """
        利用可能な正則化手法のリストを返す

        Returns:
            正則化手法名のリスト
        """
        if not cls._plugins_loaded:
            cls.load_plugins(verbose=False)
        return regularization_registry.get_names()

    @classmethod
    def get_param_info(cls, method_name: str) -> Dict[str, Dict[str, Any]]:
        """
        指定された手法のパラメータ情報を返す

        Args:
            method_name: 手法名

        Returns:
            パラメータ情報の辞書

        Raises:
            KeyError: 指定した手法が見つからない場合
        """
        if not cls._plugins_loaded:
            cls.load_plugins(verbose=False)

        # スケーリングから検索
        if scaling_registry.has(method_name):
            strategy_class = scaling_registry.get(method_name)
            return strategy_class.get_param_info()

        # 正則化から検索
        if regularization_registry.has(method_name):
            strategy_class = regularization_registry.get(method_name)
            return strategy_class.get_param_info()

        raise KeyError(f"手法 '{method_name}' が見つかりません")
