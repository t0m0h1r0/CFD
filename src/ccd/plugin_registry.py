"""
プラグインレジストリ

変換戦略を登録・管理するためのシンプルなレジストリシステム
"""

import os
import importlib
import importlib.util
import inspect
from typing import Dict, Type, List, TypeVar, Generic, Optional, Set

# 型変数の定義
T = TypeVar('T')


class PluginRegistry(Generic[T]):
    """
    プラグインレジストリ
    
    特定の基底クラスを継承するプラグインクラスを登録・管理します
    """
    
    # クラス変数 - 一度読み込んだモジュールを追跡
    _loaded_modules: Set[str] = set()
    
    def __init__(self, base_class: Type[T], registry_name: str, verbose: bool = True):
        """
        初期化
        
        Args:
            base_class: プラグインの基底クラス
            registry_name: レジストリ名（ログ表示用）
            verbose: 詳細なログを出力するかどうか
        """
        self.base_class = base_class
        self.registry_name = registry_name
        self.plugins: Dict[str, Type[T]] = {}
        self.verbose = verbose
    
    def _log(self, message: str):
        """
        条件付きでメッセージを出力
        
        Args:
            message: 出力するメッセージ
        """
        if self.verbose:
            print(message)
    
    def register(self, name: str, plugin_class: Type[T]) -> None:
        """
        プラグインを登録
        
        Args:
            name: プラグイン名（小文字のみ）
            plugin_class: 登録するプラグインクラス
            
        Raises:
            TypeError: plugin_classが基底クラスのサブクラスでない場合
        """
        if not issubclass(plugin_class, self.base_class):
            raise TypeError(f"{plugin_class.__name__}は{self.base_class.__name__}のサブクラスではありません")
        
        # 名前を小文字に正規化
        name = name.lower()
        
        # 既に同じ名前のプラグインが登録されている場合は上書き
        if name in self.plugins:
            self._log(f"{self.registry_name}: '{name}'を上書き登録しました")
        else:
            self._log(f"{self.registry_name}: '{name}'を登録しました")
        
        # プラグインを登録
        self.plugins[name] = plugin_class
    
    def unregister(self, name: str) -> None:
        """
        プラグインの登録を解除
        
        Args:
            name: 解除するプラグイン名
        """
        name = name.lower()
        if name in self.plugins:
            del self.plugins[name]
            self._log(f"{self.registry_name}: '{name}'の登録を解除しました")
    
    def get(self, name: str) -> Type[T]:
        """
        名前からプラグインクラスを取得
        
        Args:
            name: プラグイン名（大文字小文字は区別されない）
            
        Returns:
            プラグインクラス
            
        Raises:
            KeyError: 指定した名前のプラグインが見つからない場合
        """
        name = name.lower()
        
        # 直接名前で検索
        if name in self.plugins:
            return self.plugins[name]
        
        # 見つからない場合はエラー
        raise KeyError(f"{name}という名前の{self.registry_name}は登録されていません")
    
    def has(self, name: str) -> bool:
        """
        指定された名前のプラグインが存在するか確認
        
        Args:
            name: プラグイン名
            
        Returns:
            存在すればTrue、そうでなければFalse
        """
        return name.lower() in self.plugins
    
    def get_names(self) -> List[str]:
        """
        登録されている全プラグインの名前リストを取得
        
        Returns:
            プラグイン名のリスト
        """
        return list(self.plugins.keys())
    
    def get_all(self) -> Dict[str, Type[T]]:
        """
        登録されている全プラグインを取得
        
        Returns:
            {プラグイン名: プラグインクラス} の辞書
        """
        return self.plugins.copy()
    
    def scan_directory(self, directory: str) -> None:
        """
        指定ディレクトリ内のPythonファイルをスキャンし、
        基底クラスを継承するクラスを全て自動登録
        
        Args:
            directory: スキャンするディレクトリパス
        """
        # ディレクトリが存在しない場合は何もしない
        if not os.path.exists(directory) or not os.path.isdir(directory):
            self._log(f"警告: ディレクトリ {directory} が存在しないか、ディレクトリではありません")
            return
        
        # デバッグ出力
        self._log(f"{self.registry_name}: {directory} のスキャンを開始")
        
        for filename in os.listdir(directory):
            if filename.endswith('.py') and not filename.startswith('__'):
                # ファイル名からモジュール名を生成
                module_name = os.path.splitext(filename)[0]
                module_path = os.path.join(directory, filename)
                
                # 既に処理したモジュールならスキップ
                if module_path in self._loaded_modules:
                    continue
                
                try:
                    # 絶対パスからモジュールをロード
                    spec = importlib.util.spec_from_file_location(module_name, module_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        # 処理したモジュールを記録
                        self._loaded_modules.add(module_path)
                        
                        # モジュール内の全クラスを検査
                        for class_name, obj in inspect.getmembers(module, inspect.isclass):
                            # 基底クラスを継承し、かつモジュール内で定義されたクラスなら登録
                            if (issubclass(obj, self.base_class) and 
                                obj.__module__ == module.__name__ and 
                                obj is not self.base_class):
                                
                                # クラス名から登録名を生成
                                plugin_name = self._generate_plugin_name(class_name)
                                self.register(plugin_name, obj)
                
                except Exception as e:
                    self._log(f"警告: モジュール {module_name} のロード中にエラーが発生しました: {e}")
        
        self._log(f"{self.registry_name}: {len(self.plugins)} 個のプラグインを登録済み")
    
    def _generate_plugin_name(self, class_name: str) -> str:
        """
        クラス名からプラグイン名を生成
        
        Args:
            class_name: クラス名
            
        Returns:
            生成されたプラグイン名
        """
        # 末尾の "Strategy", "Scaling", "Regularization" などの接尾辞を削除
        name = class_name
        for suffix in ('Strategy', 'Scaling', 'Regularization'):
            if name.endswith(suffix):
                name = name[:-len(suffix)]
                break
        
        # 特殊ケース：大文字のみの名前（略称）は小文字に変換するだけ
        if name.isupper():
            return name.lower()
        
        # キャメルケースをスネークケースに変換
        result = ''
        for i, char in enumerate(name):
            if char.isupper() and i > 0:
                result += '_'
            result += char.lower()
        
        return result
