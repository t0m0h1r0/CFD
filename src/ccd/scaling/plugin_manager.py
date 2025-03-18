"""
スケーリングプラグインを管理するマネージャークラス
"""

import os
import importlib
import inspect
import logging
from pathlib import Path
from typing import Dict, List, Optional
from .base import BaseScaling

class ScalingPluginManager:
    """スケーリングプラグインを管理するマネージャークラス"""
    
    def __init__(self):
        """初期化"""
        self._plugins: Dict[str, BaseScaling] = {}
        self._default_plugin: Optional[BaseScaling] = None
        self._plugins_loaded: bool = False
        self._logger = logging.getLogger(__name__)
        
    def discover_plugins(self) -> Dict[str, BaseScaling]:
        """scalingパッケージ内のすべてのスケーリングプラグインを検出する"""
        if self._plugins_loaded:
            return self._plugins
            
        # scaling ディレクトリのパスを取得
        current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        os.makedirs(current_dir, exist_ok=True)
        
        # scaling ディレクトリ内のすべての .py ファイルを検索
        for py_file in current_dir.glob("*.py"):
            # 特定のファイルは除外
            if py_file.name in ["__init__.py", "base.py", "plugin_manager.py", "array_utils.py"]:
                continue
                
            module_name = py_file.stem
            
            try:
                # モジュールを動的にインポート
                module = importlib.import_module(f"scaling.{module_name}")
                
                # モジュール内のすべてのクラスを検査
                for _, obj in inspect.getmembers(module, inspect.isclass):
                    # BaseScalingのサブクラスを見つける
                    if issubclass(obj, BaseScaling) and obj is not BaseScaling:
                        instance = obj()  # インスタンスを作成
                        self._plugins[instance.name] = instance
                        
                        # デフォルトプラグイン設定
                        if self._default_plugin is None:
                            self._default_plugin = instance
                        
                        # NoScalingが見つかれば、それをデフォルトに設定
                        if instance.name.lower() == "noscaling":
                            self._default_plugin = instance
                            
            except Exception as e:
                self._logger.warning(f"プラグイン {module_name} の読み込み中にエラーが発生しました: {e}")
                
        self._plugins_loaded = True
        return self._plugins
    
    def get_plugin(self, name: Optional[str] = None) -> BaseScaling:
        """
        指定された名前のスケーリングプラグインを取得する
        
        Args:
            name: プラグイン名（Noneの場合はデフォルト）
            
        Returns:
            BaseScaling: スケーリングプラグインのインスタンス
        """
        if not self._plugins_loaded:
            self.discover_plugins()
            
        if name is None:
            return self._default_plugin
        
        # 大文字小文字を区別せずに検索
        for plugin_name, plugin in self._plugins.items():
            if plugin_name.lower() == name.lower():
                return plugin
                
        # 見つからない場合はデフォルトを返す
        self._logger.warning(f"警告: スケーリングプラグイン '{name}' が見つかりません。デフォルトを使用します。")
        return self._default_plugin
    
    def get_available_plugins(self) -> List[str]:
        """
        利用可能なすべてのプラグイン名を取得する
        
        Returns:
            list: プラグイン名のリスト
        """
        if not self._plugins_loaded:
            self.discover_plugins()
            
        return list(self._plugins.keys())