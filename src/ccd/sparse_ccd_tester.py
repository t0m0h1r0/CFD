#!/usr/bin/env python3
"""
密行列と疎行列のCCDソルバーを詳細に比較するスクリプト
"""

import numpy as np
import cupy as cp
from grid_config import GridConfig
from composite_solver import CCDCompositeSolver
from sparse_ccd_solver import SparseCompositeSolver
from test_functions import TestFunctionFactory

def compare_solvers(
    n_points=64, 
    x_range=(-1.0, 1.0), 
    coeffs=None,
    test_func_name='Sine'
):
    """
    密行列と疎行列のソルバーを詳細に比較
    
    Args:
        n_points: グリッド点の数
        x_range: x軸の範囲
        coeffs: 係数リスト [a, b, c, d]
        test_func_name: テスト関数名
    """
    # デフォルト係数
    if coeffs is None:
        coeffs = [1.0, 0.0, 0.0, 0.0]
    
    # テスト関数の選択
    test_functions = TestFunctionFactory.create_standard_functions()
    test_func = next((f for f in test_functions if f.name == test_func_name), test_functions[0])
    
    # グリッド設定
    L = x_range[1] - x_range[0]
    grid_config = GridConfig(n_points=n_points, h=L / (n_points - 1), coeffs=coeffs)
    
    # グリッド点の計算
    x_points = cp.array([x_range[0] + i * grid_config.h for i in range(n_points)])
    
    # 関数値と導関数を計算
    f_values = cp.array([test_func.f(x) for x in x_points])
    df_values = cp.array([test_func.df(x) for x in x_points])
    d2f_values = cp.array([test_func.d2f(x) for x in x_points])
    d3f_values = cp.array([test_func.d3f(x) for x in x_points])
    
    # 係数に基づいて入力関数値を計算
    a, b, c, d = coeffs
    rhs_values = a * f_values + b * df_values + c * d2f_values + d * d3f_values
    
    # ディリクレ境界条件の設定
    dirichlet_values = [f_values[0], f_values[-1]]
    
    # 密行列ソルバーの設定
    dense_solver = CCDCompositeSolver(
        grid_config,
        scaling='none',
        regularization='none'
    )
    
    # 疎行列ソルバーの設定
    sparse_solver = SparseCompositeSolver(
        grid_config,
        scaling='none',
        regularization='none'
    )
    
    # 密行列ソルバーで解を計算
    dense_psi, dense_psi_prime, dense_psi_second, dense_psi_third = dense_solver.solve(rhs_values)
    
    # 疎行列ソルバーで解を計算
    sparse_psi, sparse_psi_prime, sparse_psi_second, sparse_psi_third = sparse_solver.solve(rhs_values)
    
    # 結果の比較
    print(f"\n===== {test_func_name} 関数での密行列と疎行列の比較 =====")
    print(f"係数: {coeffs}")
    print("\n--- ψ (関数値) ---")
    print_comparison(dense_psi, sparse_psi, "関数値")
    
    print("\n--- ψ' (1階微分) ---")
    print_comparison(dense_psi_prime, sparse_psi_prime, "1階微分")
    
    print("\n--- ψ'' (2階微分) ---")
    print_comparison(dense_psi_second, sparse_psi_second, "2階微分")
    
    print("\n--- ψ''' (3階微分) ---")
    print_comparison(dense_psi_third, sparse_psi_third, "3階微分")

def print_comparison(dense_result, sparse_result, label):
    """
    密行列と疎行列の結果を比較して表示
    
    Args:
        dense_result: 密行列の計算結果
        sparse_result: 疎行列の計算結果
        label: 比較対象の説明ラベル
    """
    # NumPy配列に変換
    dense_array = dense_result.get()
    sparse_array = sparse_result.get()
    
    # 絶対誤差を計算
    abs_error = np.abs(dense_array - sparse_array)
    
    # 相対誤差を計算（0除算を防ぐため）
    relative_error = np.abs(abs_error / (np.abs(dense_array) + 1e-10))
    
    print(f"{label}:")
    print(f"  最大絶対誤差: {np.max(abs_error):.2e}")
    print(f"  最大相対誤差: {np.max(relative_error):.2e}")
    print(f"  平均絶対誤差: {np.mean(abs_error):.2e}")
    print(f"  平均相対誤差: {np.mean(relative_error):.2e}")

def main():
    """
    各種係数と関数で比較を実行
    """
    # テスト関数と係数の組み合わせ
    test_cases = [
        # 関数名, 係数
        ('Sine', [1.0, 0.0, 0.0, 0.0]),     # f = ψ
        ('Sine', [0.0, 1.0, 0.0, 0.0]),     # f = ψ'
        ('Sine', [0.0, 0.0, 1.0, 0.0]),     # f = ψ''
        ('Sine', [1.0, 1.0, 0.0, 0.0]),     # f = ψ + ψ'
        ('QuadPoly', [1.0, 0.0, 1.0, 0.0]), # より複雑な関数と係数
        ('Cosine', [1.0, 0.0, 0.0, 0.0]),   # 別の三角関数
    ]
    
    for test_func_name, coeffs in test_cases:
        print("\n" + "="*50)
        print(f"テストケース: {test_func_name}, 係数: {coeffs}")
        print("="*50)
        compare_solvers(
            n_points=64, 
            x_range=(-1.0, 1.0), 
            coeffs=coeffs,
            test_func_name=test_func_name
        )

if __name__ == "__main__":
    main()