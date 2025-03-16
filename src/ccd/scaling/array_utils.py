"""
NumPyとCuPyの間で選択的に操作を行うためのユーティリティモジュール

このモジュールは、NumPyとCuPyの互換性を保ちながら、コードを書き換えることなく
CPU/GPU処理を切り替えられるようにするためのユーティリティ関数群を提供します。
"""

import numpy as np
import os

# 環境変数からフラグを取得（デバッグ用）
_FORCE_CPU = os.environ.get('CCD_FORCE_CPU', '0').lower() in ('1', 'true', 'yes')
_VERBOSE = os.environ.get('CCD_VERBOSE', '0').lower() in ('1', 'true', 'yes')

# デフォルトはNumPy
USE_GPU = False

def set_force_cpu(value):
    """CPUを強制的に使用するかどうかを設定"""
    global _FORCE_CPU
    _FORCE_CPU = bool(value)
    
    if _VERBOSE:
        print(f"CPU強制使用フラグを {_FORCE_CPU} に設定しました")

def set_use_gpu(value):
    """GPUを使用するかどうかを設定"""
    global USE_GPU
    USE_GPU = bool(value) and not _FORCE_CPU
    
    if _VERBOSE:
        print(f"GPU使用フラグを {USE_GPU} に設定しました (FORCE_CPU={_FORCE_CPU})")

def get_array_module(arr=None):
    """
    配列の型からNumPyまたはCuPyを取得
    
    Args:
        arr: 配列（省略時はフラグに基づいて決定）
        
    Returns:
        モジュール (np または cp)
    """
    if _FORCE_CPU:
        return np
        
    if arr is not None:
        # 配列の型からモジュールを判断
        if hasattr(arr, 'get'):
            import cupy as cp
            return cp
        return np
    
    # フラグに基づいて決定
    if USE_GPU:
        try:
            import cupy as cp
            return cp
        except ImportError:
            if _VERBOSE:
                print("CuPyが利用できません。NumPyを使用します")
            return np
    
    return np

def array(data, dtype=None):
    """
    配列を作成（NumPyまたはCuPy）
    
    Args:
        data: 配列データ
        dtype: データ型
        
    Returns:
        NumPyまたはCuPy配列
    """
    if _FORCE_CPU:
        return np.array(data, dtype=dtype)
        
    if USE_GPU:
        try:
            import cupy as cp
            return cp.array(data, dtype=dtype)
        except ImportError:
            return np.array(data, dtype=dtype)
    
    return np.array(data, dtype=dtype)

def zeros(shape, dtype=None):
    """
    ゼロ配列を作成（NumPyまたはCuPy）
    
    Args:
        shape: 配列形状
        dtype: データ型
        
    Returns:
        NumPyまたはCuPy配列
    """
    if _FORCE_CPU:
        return np.zeros(shape, dtype=dtype)
        
    if USE_GPU:
        try:
            import cupy as cp
            return cp.zeros(shape, dtype=dtype)
        except ImportError:
            return np.zeros(shape, dtype=dtype)
    
    return np.zeros(shape, dtype=dtype)

def ones(shape, dtype=None):
    """
    1の配列を作成（NumPyまたはCuPy）
    
    Args:
        shape: 配列形状
        dtype: データ型
        
    Returns:
        NumPyまたはCuPy配列
    """
    if _FORCE_CPU:
        return np.ones(shape, dtype=dtype)
        
    if USE_GPU:
        try:
            import cupy as cp
            return cp.ones(shape, dtype=dtype)
        except ImportError:
            return np.ones(shape, dtype=dtype)
    
    return np.ones(shape, dtype=dtype)

def eye(N, M=None, dtype=None):
    """
    単位行列を作成（NumPyまたはCuPy）
    
    Args:
        N: 行数
        M: 列数（Noneの場合はN）
        dtype: データ型
        
    Returns:
        NumPyまたはCuPy配列
    """
    if _FORCE_CPU:
        return np.eye(N, M, dtype=dtype)
        
    if USE_GPU:
        try:
            import cupy as cp
            return cp.eye(N, M, dtype=dtype)
        except ImportError:
            return np.eye(N, M, dtype=dtype)
    
    return np.eye(N, M, dtype=dtype)

def to_numpy(arr):
    """
    CuPy配列をNumPy配列に変換（NumPy配列はそのまま）
    
    Args:
        arr: 変換する配列
        
    Returns:
        NumPy配列
    """
    if arr is None:
        return None
        
    # CuPy配列をNumPyに変換
    if hasattr(arr, 'get'):
        return arr.get()
    
    # 既にNumPy配列の場合はそのまま
    return arr