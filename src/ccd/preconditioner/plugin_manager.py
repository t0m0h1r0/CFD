# preconditioner/plugin_manager.py
"""
前処理器プラグインマネージャー

このモジュールは、前処理器プラグインを動的に検出して管理する
機能を提供します。
"""

import os
import importlib
import inspect
from .base import BasePreconditioner

class PreconditionerPluginManager:
    """前処理器プラグインマネージャー"""
    
    def __init__(self):
        """初期化"""
        self._plugins = {}
        self._default_plugin = None
    
    def discover_plugins(self):
        """
        利用可能なプラグインを検出
        
        Returns:
            検出されたプラグインの辞書 {名前: インスタンス}
        """
        if self._plugins:  # 既に検出済みの場合
            return self._plugins
            
        # 現在のパッケージディレクトリのパスを取得
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 対象外ファイル
        exclude_files = ['__init__.py', 'base.py', 'plugin_manager.py']
        
        # Pythonファイルを探索
        for filename in os.listdir(current_dir):
            if filename.endswith('.py') and filename not in exclude_files:
                module_name = filename[:-3]  # .pyを除去
                
                try:
                    # モジュールをインポート
                    module = importlib.import_module(f".{module_name}", package=__package__)
                    
                    # BasePreconditionerのサブクラスを検索
                    for obj_name, obj in inspect.getmembers(module, inspect.isclass):
                        if issubclass(obj, BasePreconditioner) and obj != BasePreconditioner:
                            # インスタンスを作成
                            instance = obj()
                            
                            # プラグイン辞書に追加
                            self._plugins[instance.name] = instance
                            
                            # 最初のプラグインをデフォルトに設定
                            if self._default_plugin is None:
                                self._default_plugin = instance
                                
                            # Identity前処理器をデフォルトとして優先
                            if 'Identity' in instance.name:
                                self._default_plugin = instance
                except Exception as e:
                    print(f"プラグイン '{module_name}' のロード中にエラー: {e}")
        
        # Identity前処理器がない場合は作成
        if not self._plugins or not any('Identity' in name for name in self._plugins):
            from .identity import IdentityPreconditioner
            identity = IdentityPreconditioner()
            self._plugins[identity.name] = identity
            self._default_plugin = identity
            
        return self._plugins
    
    def get_plugin(self, name=None):
        """
        指定した名前の前処理器を取得
        
        Args:
            name: 前処理器名（Noneの場合はデフォルト）
            
        Returns:
            前処理器インスタンス
        """
        # プラグインがまだ検出されていない場合
        if not self._plugins:
            self.discover_plugins()
            
        # Noneまたは空文字列の場合はデフォルト
        if name is None or name == '':
            return self._default_plugin
            
        # 名前で検索（大文字小文字を区別しない）
        for plugin_name, plugin in self._plugins.items():
            if plugin_name.lower() == name.lower():
                return plugin
                
        # 見つからない場合はデフォルト
        print(f"警告: 前処理器 '{name}' が見つかりません。デフォルトを使用します。")
        return self._default_plugin
    
    def list_plugins(self):
        """
        利用可能な前処理器の一覧を取得
        
        Returns:
            前処理器名のリスト
        """
        if not self._plugins:
            self.discover_plugins()
            
        return list(self._plugins.keys())