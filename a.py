#!/usr/bin/env python3
"""
CCDパッケージのディレクトリ構造を再編成するスクリプト

このスクリプトは以下の処理を行います：
1. 新しいディレクトリ構造を作成
2. ファイルを新しい場所にコピー
3. すべてのPythonファイル内のインポート文を更新
4. 元のファイルを削除
"""

import os
import re
import shutil
from pathlib import Path

# ソースディレクトリを定義
SRC_DIR = Path('src/ccd')

# ファイルの移動先を定義（旧パス -> 新パス）
FILE_MOVEMENTS = {
    # 基底クラス
    'base_equation_set.py': 'core/base/base_equation_set.py',
    'base_equation_system.py': 'core/base/base_equation_system.py',
    'base_grid.py': 'core/base/base_grid.py',
    'base_rhs_builder.py': 'core/base/base_rhs_builder.py',
    'base_solver.py': 'core/base/base_solver.py',
    'base_test_function.py': 'core/base/base_test_function.py',
    'base_tester.py': 'core/base/base_tester.py',
    'base_visualizer.py': 'core/base/base_visualizer.py',
    
    # グリッド実装
    'grid1d.py': 'core/grid/grid1d.py',
    'grid2d.py': 'core/grid/grid2d.py',
    
    # 方程式システム実装
    'equation_system1d.py': 'core/equation_system/equation_system1d.py',
    'equation_system2d.py': 'core/equation_system/equation_system2d.py',
    
    # RHSビルダー実装
    'rhs_builder1d.py': 'core/rhs_builder/rhs_builder1d.py',
    'rhs_builder2d.py': 'core/rhs_builder/rhs_builder2d.py',
    
    # ソルバー実装
    'solver1d.py': 'core/solver/solver1d.py',
    'solver2d.py': 'core/solver/solver2d.py',
    
    # 方程式セット
    'equation_sets.py': 'equation_set/equation_sets.py',
    'equation_set1d.py': 'equation_set/equation_set1d.py',
    'equation_set2d.py': 'equation_set/equation_set2d.py',
    
    # テスト関数
    'test_function_factory.py': 'test_function/test_function_factory.py',
    'test_function1d.py': 'test_function/test_function1d.py',
    'test_function2d.py': 'test_function/test_function2d.py',
    
    # テスター
    'tester1d.py': 'tester/tester1d.py',
    'tester2d.py': 'tester/tester2d.py',
    
    # 可視化
    'matrix_visualizer.py': 'visualizer/matrix_visualizer.py',
    'visualizer1d.py': 'visualizer/visualizer1d.py',
    'visualizer2d.py': 'visualizer/visualizer2d.py',
}

# モジュールのインポートパス変更マッピング（モジュール名 -> 新しいインポートパス）
MODULE_MAPPINGS = {
    'base_equation_set': 'core.base.base_equation_set',
    'base_equation_system': 'core.base.base_equation_system',
    'base_grid': 'core.base.base_grid',
    'base_rhs_builder': 'core.base.base_rhs_builder',
    'base_solver': 'core.base.base_solver',
    'base_test_function': 'core.base.base_test_function',
    'base_tester': 'core.base.base_tester',
    'base_visualizer': 'core.base.base_visualizer',
    'grid1d': 'core.grid.grid1d',
    'grid2d': 'core.grid.grid2d',
    'equation_system1d': 'core.equation_system.equation_system1d',
    'equation_system2d': 'core.equation_system.equation_system2d',
    'rhs_builder1d': 'core.rhs_builder.rhs_builder1d',
    'rhs_builder2d': 'core.rhs_builder.rhs_builder2d',
    'solver1d': 'core.solver.solver1d',
    'solver2d': 'core.solver.solver2d',
    'equation_sets': 'equation_set.equation_sets',
    'equation_set1d': 'equation_set.equation_set1d',
    'equation_set2d': 'equation_set.equation_set2d',
    'test_function_factory': 'test_function.test_function_factory',
    'test_function1d': 'test_function.test_function1d',
    'test_function2d': 'test_function.test_function2d',
    'tester1d': 'tester.tester1d',
    'tester2d': 'tester.tester2d',
    'matrix_visualizer': 'visualizer.matrix_visualizer',
    'visualizer1d': 'visualizer.visualizer1d',
    'visualizer2d': 'visualizer.visualizer2d',
}

def create_directory_structure():
    """必要なディレクトリ構造を作成します"""
    # 新しいパスのディレクトリを抽出
    dirs = set()
    for new_path in FILE_MOVEMENTS.values():
        dir_path = Path(new_path).parent
        dirs.add(dir_path)
    
    # ディレクトリと__init__.pyファイルを作成
    for dir_path in sorted(dirs):
        full_path = SRC_DIR / dir_path
        os.makedirs(full_path, exist_ok=True)
        
        # __init__.pyファイルを作成（存在しない場合）
        init_file = full_path / '__init__.py'
        if not init_file.exists():
            package_name = str(dir_path).replace('/', '.')
            with open(init_file, 'w', encoding='utf-8') as f:
                f.write(f'"""\n{package_name} パッケージ\n"""\n')

def update_imports_in_file(file_path):
    """ファイル内のインポート文を更新します"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 移動するモジュールのインポートのみを更新
    updated_content = content
    for old_module, new_module in MODULE_MAPPINGS.items():
        # 異なるインポート文のパターンに対応
        patterns = [
            (rf'from\s+{re.escape(old_module)}\s+import', f'from {new_module} import'),
            (rf'import\s+{re.escape(old_module)}(\s|$)', f'import {new_module}\\1'),
        ]
        
        # すべてのパターンを適用
        for pattern, replacement in patterns:
            updated_content = re.sub(pattern, replacement, updated_content)
    
    # 変更があれば書き戻す
    if updated_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        return True
    
    return False

def reorganize_files():
    """ファイルを再編成する主関数"""
    # ディレクトリ構造を作成
    create_directory_structure()
    
    # ファイルを新しい場所にコピー
    for old_path, new_path in FILE_MOVEMENTS.items():
        src_file = SRC_DIR / old_path
        dst_file = SRC_DIR / new_path
        
        if src_file.exists():
            print(f"{old_path} を {new_path} にコピー中")
            shutil.copy2(src_file, dst_file)
    
    # すべてのPythonファイルのインポートを更新
    updated_files = []
    for root, dirs, files in os.walk(SRC_DIR):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if update_imports_in_file(file_path):
                    updated_files.append(os.path.relpath(file_path, SRC_DIR))
    
    # すべての更新後に元のファイルを削除
    for old_path in FILE_MOVEMENTS:
        src_file = SRC_DIR / old_path
        if src_file.exists():
            print(f"元のファイル {old_path} を削除中")
            os.remove(src_file)
    
    # 結果サマリーを表示
    print(f"{len(updated_files)} ファイルのインポートを更新しました:")
    for file in updated_files:
        print(f"  - {file}")
    print("再編成が完了しました！")

if __name__ == "__main__":
    reorganize_files()
