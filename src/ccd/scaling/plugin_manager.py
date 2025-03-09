import os
import importlib
import inspect
from pathlib import Path
from .base import BaseScaling

class ScalingPluginManager:
    """スケーリングプラグインを管理するマネージャークラス"""
    
    def __init__(self):
        self._plugins = {}
        self._default_plugin = None
        self._plugins_loaded = False
        
    def discover_plugins(self):
        """scalingパッケージ内のすべてのスケーリングプラグインを検出する"""
        if self._plugins_loaded:
            return self._plugins
            
        # scaling ディレクトリのパスを取得
        current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        
        # ディレクトリが存在しない場合は作成
        os.makedirs(current_dir, exist_ok=True)
        
        # scaling ディレクトリ内のすべての .py ファイルを検索
        for py_file in current_dir.glob("*.py"):
            # __init__.py やこのファイル自体は除外
            if py_file.name in ["__init__.py", "base.py", "plugin_manager.py"]:
                continue
                
            module_name = py_file.stem  # .py 拡張子を除去
            
            try:
                # モジュールを動的にインポート
                module = importlib.import_module(f"scaling.{module_name}")
                
                # モジュール内のすべてのクラスを検査
                for _, obj in inspect.getmembers(module, inspect.isclass):
                    # BaseScalingのサブクラスでBaseScaling自身ではないか確認
                    if issubclass(obj, BaseScaling) and obj is not BaseScaling:
                        instance = obj()
                        self._plugins[instance.name] = instance
                        
                        # デフォルトプラグインがまだ設定されていなければ最初のプラグインを設定
                        if self._default_plugin is None:
                            self._default_plugin = instance
                        
                        # NoScalingが見つかれば、それをデフォルトに設定
                        if instance.name.lower() == "noscaling":
                            self._default_plugin = instance
                            
            except Exception as e:
                print(f"プラグイン {module_name} の読み込み中にエラーが発生しました: {e}")
                
        self._plugins_loaded = True
        return self._plugins
    
    def get_plugin(self, name=None):
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
        print(f"警告: スケーリングプラグイン '{name}' が見つかりません。デフォルトを使用します。")
        return self._default_plugin
    
    def get_available_plugins(self):
        """
        利用可能なすべてのプラグイン名を取得する
        
        Returns:
            list: プラグイン名のリスト
        """
        if not self._plugins_loaded:
            self.discover_plugins()
            
        return list(self._plugins.keys())