"""
スケーリングプラグインを管理するマネージャークラス
"""

import os
import importlib
import inspect
import logging
import sys
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
        
        # Set up logger with ERROR level by default (minimal output)
        self._logger.setLevel(logging.ERROR)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.ERROR)
            self._logger.addHandler(handler)
        
        # Verbose flag for controlling log level
        self._verbose = False
        
    @property
    def verbose(self):
        """詳細ログ出力の設定を取得"""
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        """詳細ログ出力の設定を更新"""
        self._verbose = value
        
        # Update log level based on verbose flag
        if value:
            self._logger.setLevel(logging.DEBUG)
            for handler in self._logger.handlers:
                handler.setLevel(logging.DEBUG)
        else:
            self._logger.setLevel(logging.ERROR)
            for handler in self._logger.handlers:
                handler.setLevel(logging.ERROR)
        
    def discover_plugins(self) -> Dict[str, BaseScaling]:
        """scalingパッケージ内のすべてのスケーリングプラグインを検出する"""
        if self._plugins_loaded:
            return self._plugins
            
        # scaling ディレクトリのパスを取得
        current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        self._logger.debug(f"Searching for plugins in: {current_dir}")
        
        # 追加: パスを表示
        self._logger.debug(f"Python path: {sys.path}")
        
        # scaling ディレクトリの存在を確認
        if not current_dir.exists():
            self._logger.warning(f"Directory does not exist: {current_dir}")
            return self._plugins
        
        # scaling ディレクトリ内のすべての .py ファイルを検索
        py_files = list(current_dir.glob("*.py"))
        self._logger.debug(f"Found {len(py_files)} Python files in the directory")
        
        for py_file in py_files:
            # 特定のファイルは除外
            if py_file.name in ["__init__.py", "base.py", "plugin_manager.py", "array_utils.py"]:
                self._logger.debug(f"Skipping special file: {py_file.name}")
                continue
                
            module_name = py_file.stem
            self._logger.debug(f"Trying to import module: {module_name}")
            
            try:
                # モジュールを動的にインポート
                # NOTE: パッケージ名を絶対パスで変更（相対インポートの問題対応）
                full_module_name = f"scaling.{module_name}"
                module = importlib.import_module(full_module_name)
                
                # モジュール内のすべてのクラスを検査
                classes_found = 0
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    # BaseScalingのサブクラスを見つける
                    if issubclass(obj, BaseScaling) and obj is not BaseScaling:
                        classes_found += 1
                        instance = obj()  # インスタンスを作成
                        self._plugins[instance.name] = instance
                        self._logger.debug(f"Added plugin: {instance.name} from {module_name}")
                        
                        # デフォルトプラグイン設定
                        if self._default_plugin is None:
                            self._default_plugin = instance
                        
                        # NoScalingが見つかれば、それをデフォルトに設定
                        if instance.name.lower() == "noscaling":
                            self._default_plugin = instance
                
                self._logger.debug(f"Found {classes_found} plugin classes in {module_name}")
                            
            except Exception as e:
                self._logger.warning(f"プラグイン {module_name} の読み込み中にエラーが発生しました: {e}")
                import traceback
                self._logger.debug(traceback.format_exc())
        
        # 明示的にDefaultScalingを作成
        if not self._plugins:
            from .no_scaling import NoScaling
            self._default_plugin = NoScaling()
            self._plugins[self._default_plugin.name] = self._default_plugin
            self._logger.debug("Created default NoScaling plugin")
        
        self._plugins_loaded = True
        self._logger.debug(f"Loaded {len(self._plugins)} plugins: {list(self._plugins.keys())}")
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
            if self._default_plugin is None:
                self._logger.warning("Default plugin is None. Creating NoScaling plugin.")
                from .no_scaling import NoScaling
                self._default_plugin = NoScaling()
                self._plugins[self._default_plugin.name] = self._default_plugin
            return self._default_plugin
        
        # 大文字小文字を区別せずに検索
        for plugin_name, plugin in self._plugins.items():
            if plugin_name.lower() == name.lower():
                return plugin
                
        # 見つからない場合はデフォルトを返す
        self._logger.warning(f"警告: スケーリングプラグイン '{name}' が見つかりません。デフォルトを使用します。")
        if self._default_plugin is None:
            self._logger.warning("Default plugin is None. Creating NoScaling plugin.")
            from .no_scaling import NoScaling
            self._default_plugin = NoScaling()
            self._plugins[self._default_plugin.name] = self._default_plugin
        return self._default_plugin
    
    def get_available_plugins(self) -> List[str]:
        """
        利用可能なすべてのプラグイン名を取得する
        
        Returns:
            list: プラグイン名のリスト
        """
        if not self._plugins_loaded:
            self.discover_plugins()
            
        plugin_names = list(self._plugins.keys())
        self._logger.debug(f"Available plugins: {plugin_names}")
        return plugin_names