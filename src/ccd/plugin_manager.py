"""
プラグイン管理モジュール

CCDソルバーのスケーリング戦略と正則化戦略をプラグイン形式で管理するモジュール
"""

import os
import sys
import importlib
import importlib.util
import inspect
from typing import Dict, Type, List, TypeVar, Generic, Any, Set

# 型変数の定義
T = TypeVar('T')


class PluginRegistry(Generic[T]):
    """
    プラグインレジストリクラス
    特定の基底クラスを継承する全てのプラグインクラスを検出・登録する
    """
    
    # クラス変数として共有状態を管理
    _loaded_modules: Set[str] = set()  # 既にロードしたモジュールのパス
    _silent_mode: bool = False  # 静かモード（出力抑制）
    
    def __init__(self, base_class: Type[T], plugin_name: str):
        """
        Args:
            base_class: プラグインの基底クラス
            plugin_name: プラグインのタイプ名（ログ表示用）
        """
        self.base_class = base_class
        self.plugin_name = plugin_name
        self.plugins: Dict[str, Type[T]] = {}
        self.displayed_methods = False  # メソッド一覧を表示したかどうか
    
    @classmethod
    def enable_silent_mode(cls):
        """出力を抑制する静かモードを有効化"""
        cls._silent_mode = True
    
    @classmethod
    def disable_silent_mode(cls):
        """静かモードを無効化（通常の出力に戻す）"""
        cls._silent_mode = False
    
    def _log(self, message: str):
        """
        条件付きでメッセージを出力
        静かモードでなければ出力する
        """
        if not self._silent_mode:
            print(message)
    
    def register(self, name: str, plugin_class: Type[T]) -> None:
        """
        プラグインを手動で登録
        
        Args:
            name: プラグイン名（小文字のみ）
            plugin_class: 登録するプラグインクラス
        """
        if not issubclass(plugin_class, self.base_class):
            raise TypeError(f"{plugin_class.__name__} は {self.base_class.__name__} のサブクラスではありません")
        
        # 名前を小文字に統一
        name = name.lower()
        
        # すでに同名のプラグインが登録されていたら、同じクラスならスキップ
        if name in self.plugins:
            # クラスの完全修飾名を比較して同一性をチェック
            existing_class_name = f"{self.plugins[name].__module__}.{self.plugins[name].__name__}"
            new_class_name = f"{plugin_class.__module__}.{plugin_class.__name__}"
            
            if existing_class_name == new_class_name:
                # 同じクラスなら何もしない
                return
            else:
                # 異なるクラスなら警告（ただし静かモードでは表示しない）
                self._log(f"警告: {name} という名前の{self.plugin_name}は既に登録されています。上書きします。")
        
        self.plugins[name] = plugin_class
        self._log(f"{self.plugin_name} '{name}' を登録しました")
    
    def unregister(self, name: str) -> None:
        """
        プラグインを登録解除
        
        Args:
            name: 解除するプラグイン名（小文字）
        """
        name = name.lower()
        if name in self.plugins:
            del self.plugins[name]
            self._log(f"{self.plugin_name} '{name}' の登録を解除しました")
    
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
    
    def display_available_methods(self, get_param_info_func=None):
        """
        利用可能な手法と各パラメータを表示
        
        Args:
            get_param_info_func: パラメータ情報を取得する関数（オプション）
        """
        if self.displayed_methods:
            return  # 既に表示済みならスキップ
        
        self._log(f"=== 使用可能な{self.plugin_name} ===")
        for method in self.get_names():
            if get_param_info_func:
                param_info = get_param_info_func(method)
                if param_info:
                    params = ", ".join([f"{k} ({v['help']}, デフォルト: {v['default']})" for k, v in param_info.items()])
                    self._log(f"- {method} - パラメータ: {params}")
                    continue
            self._log(f"- {method}")
        
        self.displayed_methods = True
    
    def scan_directory(self, directory: str) -> None:
        """
        指定ディレクトリ内のPythonファイルをスキャンし、
        基底クラスを継承するクラスを全て自動登録
        
        Args:
            directory: スキャンするディレクトリパス
        """
        # ディレクトリ内の全Pythonファイルを取得
        if not os.path.exists(directory):
            self._log(f"警告: ディレクトリ {directory} が存在しません")
            return
        
        for filename in os.listdir(directory):
            if filename.endswith('.py') and not filename.startswith('__'):
                # ファイル名からモジュール名を生成
                module_name = os.path.splitext(filename)[0]
                module_path = os.path.join(directory, filename)
                
                # 既に処理したファイルならスキップ
                if module_path in self._loaded_modules:
                    continue
                
                try:
                    # 絶対パスからモジュールをロード
                    spec = importlib.util.spec_from_file_location(module_name, module_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        # 処理したファイルを記録
                        self._loaded_modules.add(module_path)
                        
                        # モジュール内の全クラスを検査
                        for name, obj in inspect.getmembers(module, inspect.isclass):
                            # 基底クラスを継承し、かつモジュール内で定義されたクラスなら登録
                            if (issubclass(obj, self.base_class) and 
                                obj.__module__ == module.__name__ and 
                                obj is not self.base_class):
                                
                                # クラス名から登録名を生成
                                # 例: FooBarStrategy -> foo_bar
                                plugin_name = self._generate_plugin_name(name)
                                
                                # スキップすべき余分な変換名を検出
                                if self._is_redundant_name(name, plugin_name):
                                    continue
                                
                                self.register(plugin_name, obj)
                
                except (ImportError, AttributeError) as e:
                    self._log(f"警告: モジュール {module_name} のロード中にエラーが発生しました: {e}")
    
    def _generate_plugin_name(self, class_name: str) -> str:
        """
        クラス名からプラグイン名を生成
        
        Args:
            class_name: クラス名
            
        Returns:
            生成されたプラグイン名
        """
        # 末尾の "Strategy", "Scaling", "Regularization" を削除
        name = class_name
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
        
        return plugin_name
    
    def _is_redundant_name(self, class_name: str, plugin_name: str) -> bool:
        """
        冗長な名前変換かどうかを判定
        
        Args:
            class_name: 元のクラス名
            plugin_name: 生成されたプラグイン名
            
        Returns:
            True: 冗長な名前変換（スキップすべき）
            False: 正常な名前変換
        """
        # 冗長な変換の例: SVD -> s_v_d （svdが既に登録されている場合）
        # 大文字のみの名前をアンダースコア区切りにした場合、元の小文字版が既に存在するかチェック
        if class_name.isupper() and '_' in plugin_name:
            simple_name = class_name.lower()
            if simple_name in self.plugins:
                return True
        
        return False
    
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
            self._log(f"警告: パッケージ {package_name} のインポート中にエラーが発生しました: {e}")