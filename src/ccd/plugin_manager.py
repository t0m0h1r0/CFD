"""
プラグイン管理モジュール

CCDソルバーのスケーリング戦略と正則化戦略をプラグイン形式で管理するモジュール
正則化戦略が表示されない問題を修正
"""

import os
import importlib
import importlib.util
import inspect
from typing import Dict, Type, List, TypeVar, Generic, Set

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
    _scaling_initialized: bool = False  # スケーリング初期化完了フラグ
    _regularization_initialized: bool = False  # 正則化初期化完了フラグ
    
    # クラス初期化
    _instances = []
    
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
        
        # 同じ基底クラスのレジストリが既に初期化されているかチェック
        for registry in PluginRegistry._instances:
            if registry.base_class == base_class:
                # 既存のレジストリの状態をコピー
                self.plugins = registry.plugins.copy()
                self.displayed_methods = registry.displayed_methods
                return
        
        # 新しいレジストリをインスタンスリストに追加
        PluginRegistry._instances.append(self)
    
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
        
        # 重複チェック: 大文字小文字を無視して同等の名前かどうかをチェック
        # 例: SVD, s_v_d, svd は同等とみなす
        normalized_name = self._normalize_name(name)
        for existing_name in list(self.plugins.keys()):
            if self._normalize_name(existing_name) == normalized_name:
                # 既存の名前と同等の名前が存在する場合
                existing_class = self.plugins[existing_name]
                existing_class_name = f"{existing_class.__module__}.{existing_class.__name__}"
                new_class_name = f"{plugin_class.__module__}.{plugin_class.__name__}"
                
                if existing_class_name == new_class_name:
                    # 同じクラスなら何もしない
                    return
                else:
                    # 別のクラスの場合は、既存のものを優先
                    self._log(f"警告: {name} は既存の {existing_name} と同等とみなされ、スキップされました")
                    return
        
        # 新しいプラグインを登録
        self.plugins[name] = plugin_class
        self._log(f"{self.plugin_name} '{name}' を登録しました")
    
    def _normalize_name(self, name: str) -> str:
        """
        名前を正規化して比較可能にする
        例: SVD, s_v_d, svd -> svd
        
        Args:
            name: 正規化する名前
            
        Returns:
            正規化された名前
        """
        # アンダースコアを削除
        name = name.replace('_', '')
        # 小文字に変換
        return name.lower()
    
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
        
        # 直接名前で検索
        if name in self.plugins:
            return self.plugins[name]
        
        # 正規化した名前で検索
        normalized_name = self._normalize_name(name)
        for existing_name, plugin_class in self.plugins.items():
            if self._normalize_name(existing_name) == normalized_name:
                return plugin_class
        
        # 見つからない場合はエラー
        raise KeyError(f"{name} という名前の{self.plugin_name}は登録されていません")
    
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
        
        # 重複を除去した名前の正規化セット
        normalized_names = {}
        
        for name in self.get_names():
            normalized = self._normalize_name(name)
            
            # 既に同等の名前が処理済みならスキップ
            if normalized in normalized_names:
                continue
            
            normalized_names[normalized] = name
            
            if get_param_info_func:
                param_info = get_param_info_func(name)
                if param_info:
                    params = ", ".join([f"{k} ({v['help']}, デフォルト: {v['default']})" for k, v in param_info.items()])
                    self._log(f"- {name} - パラメータ: {params}")
                    continue
            
            self._log(f"- {name}")
        
        self.displayed_methods = True
    
    def scan_directory(self, directory: str) -> None:
        """
        指定ディレクトリ内のPythonファイルをスキャンし、
        基底クラスを継承するクラスを全て自動登録
        
        Args:
            directory: スキャンするディレクトリパス
        """
        # 正則化とスケーリングで異なる初期化フラグをチェック
        is_scaling = "scaling" in directory.lower()
        is_regularization = "regularization" in directory.lower()
        
        if (is_scaling and PluginRegistry._scaling_initialized) or \
           (is_regularization and PluginRegistry._regularization_initialized):
            # すでに対応する種類が初期化済みならスキップ
            return
        
        # ディレクトリ内の全Pythonファイルを取得
        if not os.path.exists(directory):
            self._log(f"警告: ディレクトリ {directory} が存在しません")
            return
        
        # デバッグ出力
        self._log(f"スキャン開始: {directory}")
        
        for filename in os.listdir(directory):
            if filename.endswith('.py') and not filename.startswith('__'):
                # ファイル名からモジュール名を生成
                module_name = os.path.splitext(filename)[0]
                module_path = os.path.join(directory, filename)
                
                # 既に処理したファイルならスキップ
                if module_path in self._loaded_modules:
                    continue
                
                # デバッグ出力
                self._log(f"モジュール読み込み: {module_path}")
                
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
                                self.register(plugin_name, obj)
                
                except (ImportError, AttributeError) as e:
                    self._log(f"警告: モジュール {module_name} のロード中にエラーが発生しました: {e}")
        
        # 初期化完了フラグを設定（種類ごとに異なるフラグ）
        if is_scaling:
            PluginRegistry._scaling_initialized = True
            self._log("スケーリング戦略の初期化完了")
        elif is_regularization:
            PluginRegistry._regularization_initialized = True
            self._log("正則化戦略の初期化完了")
    
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
        
        # 特殊ケース：大文字のみの名前（略称）は小文字に変換するだけ
        if name.isupper():
            return name.lower()
        
        # キャメルケースをスネークケースに変換
        plugin_name = ''
        for i, char in enumerate(name):
            if char.isupper() and i > 0:
                plugin_name += '_'
            plugin_name += char.lower()
        
        return plugin_name
    
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