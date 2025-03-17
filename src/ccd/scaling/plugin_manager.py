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
        self._plugins: Dict[str, type] = {}  # クラス型を保存
        self._default_plugin: Optional[str] = None
        self._plugins_loaded: bool = False
        self._logger = logging.getLogger(__name__)
        self._backend = 'numpy'  # デフォルトバックエンド
        
    def set_backend(self, backend: str):
        """
        全プラグインの計算バックエンドを設定
        
        Args:
            backend: 'numpy', 'cupy', 'jax' のいずれか
        """
        self._backend = backend
        
    def discover_plugins(self) -> Dict[str, type]:
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
                        self._plugins[obj().name] = obj
                        
                        # デフォルトプラグイン設定
                        if self._default_plugin is None:
                            self._default_plugin = obj().name
                        
                        # NoScalingが見つかれば、それをデフォルトに設定
                        if obj().name.lower() == "noscaling":
                            self._default_plugin = obj().name
                            
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
            # デフォルトプラグインを返す
            plugin_class = self._plugins.get(self._default_plugin)
            if plugin_class is None:
                # Fallback to NoScaling
                from .no_scaling import NoScaling
                return NoScaling(backend=self._backend)
            return plugin_class(backend=self._backend)
        
        # 大文字小文字を区別せずに検索
        for plugin_name, plugin_class in self._plugins.items():
            if plugin_name.lower() == name.lower():
                return plugin_class(backend=self._backend)
                
        # 見つからない場合はデフォルトを返す
        self._logger.warning(f"警告: スケーリングプラグイン '{name}' が見つかりません。デフォルトを使用します。")
        plugin_class = self._plugins.get(self._default_plugin)
        if plugin_class is None:
            # Fallback to NoScaling
            from .no_scaling import NoScaling
            return NoScaling(backend=self._backend)
        return plugin_class(backend=self._backend)
    
    def get_available_plugins(self) -> List[str]:
        """
        利用可能なすべてのプラグイン名を取得する
        
        Returns:
            list: プラグイン名のリスト
        """
        if not self._plugins_loaded:
            self.discover_plugins()
            
        return list(self._plugins.keys())
