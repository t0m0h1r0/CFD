"""
プラグイン管理モジュール

CCDソルバーのスケーリング戦略と正則化戦略をプラグイン形式で管理するモジュール
"""

import os
import importlib
import inspect
from typing import Dict, Type, List, TypeVar, Generic, Any

# 型変数の定義
T = TypeVar('T')


class PluginRegistry(Generic[T]):
    """
    プラグインレジストリクラス
    特定の基底クラスを継承する全てのプラグインクラスを検出・登録する
    """
    
    def __init__(self, base_class: Type[T], plugin_name: str):
        """
        Args:
            base_class: プラグインの基底クラス
            plugin_name: プラグインのタイプ名（ログ表示用）
        """
        self.base_class = base_class
        self.plugin_name = plugin_name
        self.plugins: Dict[str, Type[T]] = {}
        self._registered_files = set()  # 既に登録したファイルを追跡
    
    def register(self, name: str, plugin_class: Type[T], silent: bool = False) -> None:
        """
        プラグインを手動で登録
        
        Args:
            name: プラグイン名（小文字のみ）
            plugin_class: 登録するプラグインクラス
            silent: Trueの場合、登録メッセージを表示しない
        """
        if not issubclass(plugin_class, self.base_class):
            raise TypeError(f"{plugin_class.__name__} は {self.base_class.__name__} のサブクラスではありません")
        
        # 名前を小文字に統一
        name = name.lower()
        
        # すでに同名のプラグインが登録されている場合は上書きしない
        if name in self.plugins:
            if self.plugins[name] is plugin_class:
                # 全く同じクラスなら何もしない（無視）
                return
            elif not silent:
                print(f"警告: {name} という名前の{self.plugin_name}は既に登録されています。上書きします。")
        
        self.plugins[name] = plugin_class
        if not silent:
            print(f"{self.plugin_name} '{name}' を登録しました")
    
    def unregister(self, name: str) -> None:
        """
        プラグインを登録解除
        
        Args:
            name: 解除するプラグイン名（小文字）
        """
        name = name.lower()
        if name in self.plugins:
            del self.plugins[name]
            print(f"{self.plugin_name} '{name}' の登録を解除しました")
    
    def get(self, name: str) -> Type[T]:
        """
        名前からプラグインクラスを取得
        
        Args:
            name: プラグイン名（大文字小文字は区別しない）
            
        Returns:
            プラグインクラス
            
        Raises:
            KeyError: 指定した名前のプラグインが見つからない場合
        """
        name = name.lower()
        if name not in self.plugins:
            raise KeyError(f"{name} という名前の{self.plugin_name}は登録されていません")
        
        return self.plugins[name]
    
    def get_all(self) -> Dict[str, Type[T]]:
        """
        登録されている全プラグインを取得
        
        Returns:
            {プラグイン名: プラグインクラス} の辞書
        """
        return self.plugins.copy()
    
    def get_names(self) -> List[str]:
        """
        登録されている全プラグインの名前リストを取得
        
        Returns:
            プラグイン名のリスト
        """
        return list(self.plugins.keys())
    
    def scan_directory(self, directory: str) -> None:
        """
        指定ディレクトリ内のPythonファイルをスキャンし、
        基底クラスを継承するクラスを全て自動登録
        
        Args:
            directory: スキャンするディレクトリパス
        """
        # ディレクトリ内の全Pythonファイルを取得
        if not os.path.exists(directory):
            print(f"警告: ディレクトリ {directory} が存在しません")
            return
        
        for filename in os.listdir(directory):
            if filename.endswith('.py') and not filename.startswith('__'):
                # ファイル名からモジュール名を生成
                module_name = os.path.splitext(filename)[0]
                module_path = os.path.join(directory, filename)
                
                # 既に処理したファイルならスキップ
                if module_path in self._registered_files:
                    continue
                
                try:
                    # 絶対パスからモジュールをロード
                    spec = importlib.util.spec_from_file_location(module_name, module_path)
                    if spec:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        # モジュール内の全クラスを検査
                        for name, obj in inspect.getmembers(module, inspect.isclass):
                            # 基底クラスを継承し、かつモジュール内で定義されたクラスなら登録
                            if (issubclass(obj, self.base_class) and 
                                obj.__module__ == module.__name__ and 
                                obj is not self.base_class):
                                
                                # クラス名から登録名を生成
                                # 例: FooBarStrategy -> foo_bar
                                if name.endswith(('Strategy', 'Scaling', 'Regularization')):
                                    # 末尾の "Strategy", "Scaling", "Regularization" を削除
                                    for suffix in ('Strategy', 'Scaling', 'Regularization'):
                                        if name.endswith(suffix):
                                            name = name[:-len(suffix)]
                                            break
                                
                                # キャメルケースをスネークケースに変換
                                plugin_name = ''
                                for i, char in enumerate(name):
                                    if char.isupper() and i > 0:
                                        plugin_name += '_'
                                    plugin_name += char.lower()
                                
                                self.register(plugin_name, obj)
                        
                        # 処理完了後、ファイルを記録
                        self._registered_files.add(module_path)
                
                except (ImportError, AttributeError) as e:
                    print(f"警告: モジュール {module_name} のロード中にエラーが発生しました: {e}")
    
    def scan_package(self, package_name: str) -> None:
        """
        指定パッケージ内のモジュールをスキャンし、
        基底クラスを継承するクラスを全て自動登録
        
        Args:
            package_name: スキャンするパッケージ名
        """
        try:
            package = importlib.import_module(package_name)
            package_path = os.path.dirname(package.__file__)
            self.scan_directory(package_path)
        except ImportError as e:
            print(f"警告: パッケージ {package_name} のインポート中にエラーが発生しました: {e}")