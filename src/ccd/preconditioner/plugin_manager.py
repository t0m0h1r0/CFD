"""
前処理プラグインマネージャー

このモジュールは、CCD法で使用する前処理手法のプラグインを
動的に検出して管理するマネージャークラスを提供します。
"""

import os
import importlib
import inspect
import logging
from pathlib import Path
from typing import Dict, Optional
from .base import BasePreconditioner

class PreconditionerPluginManager:
    """前処理プラグインマネージャー"""
    
    def __init__(self):
        """初期化"""
        self._plugins: Dict[str, BasePreconditioner] = {}
        self._default_plugin: Optional[BasePreconditioner] = None
        self._plugins_loaded: bool = False
        self._logger = logging.getLogger(__name__)
        
        # ロガー設定
        self._logger.setLevel(logging.ERROR)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.ERROR)
            self._logger.addHandler(handler)
        
        # 詳細出力用フラグ
        self._verbose = False
    
    @property
    def verbose(self):
        """詳細出力フラグの取得"""
        return self._verbose
    
    @verbose.setter
    def verbose(self, value):
        """詳細出力フラグの設定"""
        self._verbose = value
        
        # フラグに応じてログレベルを更新
        if value:
            self._logger.setLevel(logging.DEBUG)
            for handler in self._logger.handlers:
                handler.setLevel(logging.DEBUG)
        else:
            self._logger.setLevel(logging.ERROR)
            for handler in self._logger.handlers:
                handler.setLevel(logging.ERROR)
    
    def discover_plugins(self) -> Dict[str, BasePreconditioner]:
        """
        利用可能な前処理プラグインを検出
        
        Returns:
            Dict[str, BasePreconditioner]: 検出されたプラグインの辞書
        """
        if self._plugins_loaded:
            return self._plugins
        
        # プラグインディレクトリのパスを取得
        current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        self._logger.debug(f"プラグイン検索場所: {current_dir}")
        
        # ディレクトリの存在確認
        if not current_dir.exists():
            self._logger.warning(f"ディレクトリが存在しません: {current_dir}")
            return self._plugins
        
        # ディレクトリ内のPythonファイルを検索
        py_files = list(current_dir.glob("*.py"))
        self._logger.debug(f"ディレクトリ内の.pyファイル数: {len(py_files)}")
        
        for py_file in py_files:
            # 特殊ファイルはスキップ
            if py_file.name in ["__init__.py", "base.py", "plugin_manager.py"]:
                self._logger.debug(f"特殊ファイルをスキップ: {py_file.name}")
                continue
            
            module_name = py_file.stem
            self._logger.debug(f"モジュールインポート試行: {module_name}")
            
            try:
                # モジュールを動的にインポート
                full_module_name = f"preconditioner.{module_name}"
                module = importlib.import_module(full_module_name)
                
                # モジュール内のすべてのクラスを検査
                classes_found = 0
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    # BasePreconditionerのサブクラスを検索
                    if issubclass(obj, BasePreconditioner) and obj is not BasePreconditioner:
                        classes_found += 1
                        instance = obj()  # インスタンスを作成
                        self._plugins[instance.name] = instance
                        self._logger.debug(f"プラグイン追加: {instance.name} from {module_name}")
                        
                        # デフォルトプラグインを設定
                        if self._default_plugin is None:
                            self._default_plugin = instance
                        
                        # IdentityPreconditionerがあればデフォルトとして使用
                        if instance.name.lower() == "identitypreconditioner":
                            self._default_plugin = instance
                
                self._logger.debug(f"{module_name}で{classes_found}個のプラグインクラスを発見")
            
            except Exception as e:
                self._logger.warning(f"プラグイン {module_name} 読み込みエラー: {e}")
                import traceback
                self._logger.debug(traceback.format_exc())
        
        # プラグインが見つからない場合、デフォルトを作成
        if not self._plugins:
            from .identity import IdentityPreconditioner
            self._default_plugin = IdentityPreconditioner()
            self._plugins[self._default_plugin.name] = self._default_plugin
            self._logger.debug("デフォルトのIdentityPreconditionerを作成")
        
        self._plugins_loaded = True
        self._logger.debug(f"{len(self._plugins)}個のプラグインを読み込み: {list(self._plugins.keys())}")
        return self._plugins
    
    def get_plugin(self, name: Optional[str] = None) -> BasePreconditioner:
        """
        指定した名前の前処理プラグインを取得
        
        Args:
            name: プラグイン名（Noneの場合はデフォルト）
            
        Returns:
            BasePreconditioner: 前処理インスタンス
        """
        if not self._plugins_loaded:
            self.discover_plugins()
        
        if name is None:
            if self._default_plugin is None:
                self._logger.warning("デフォルトプラグインがNone。IdentityPreconditionerを作成。")
                from .identity import IdentityPreconditioner
                self._default_plugin = IdentityPreconditioner()
                self._plugins[self._default_plugin.name] = self._default_plugin
            return self._default_plugin
        
        # 大文字小文字を区別せずに検索
        for plugin_name, plugin in self._plugins.items():
            if plugin_name.lower() == name.lower():
                return plugin
        
        # 見つからない場合はデフォルトを返す
        self._logger.warning(f"警告: 前処理プラグイン '{name}' が見つかりません。デフォルトを使用します。")
        if self._default_plugin is None:
            self._logger.warning("デフォルトプラグインがNone。IdentityPreconditionerを作成。")
            from .identity import IdentityPreconditioner
            self._default_plugin = IdentityPreconditioner()
            self._plugins[self._default_plugin.name] = self._default_plugin
        return self._default_plugin