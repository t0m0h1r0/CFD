"""
プリセット設定モジュール

CCDソルバーのプリセット設定を定義し、プリセット設定を取得するための関数を提供します。
"""

from typing import List, Tuple, Dict, Any


def get_scaling_presets() -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    スケーリングプリセットを取得
    
    Returns:
        [("表示名", "スケーリング名", パラメータ辞書), ...] の形式のリスト
    """
    presets = [
        ("None", "none", {}),
        ("Normalization", "normalization", {}),
        ("Rehu", "rehu", {}),
        ("Equalization", "equalization", {}),
        ("Iterative", "iterative", {"max_iter": 10, "tol": 1e-8}),
        ("VanDerSluis", "van_der_sluis", {}),
        ("DiagonalDominance", "diagonal_dominance", {}),
        ("SquareSum", "square_sum", {}),
        ("MaxElement", "max_element", {})
    ]
    
    return presets


def get_regularization_presets() -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    正則化プリセットを取得
    
    Returns:
        [("表示名", "正則化名", パラメータ辞書), ...] の形式のリスト
    """
    presets = [
        ("None", "none", {}),
        ("Tikhonov", "tikhonov", {"alpha": 1e-6}),
        ("Tikhonov_Strong", "tikhonov", {"alpha": 1e-4}),
        ("SVD", "svd", {"threshold": 1e-10}),
        ("TSVD", "tsvd", {"threshold_ratio": 1e-5}),
        ("TSVD_10", "tsvd", {"rank": 10}),
        ("Landweber", "landweber", {"iterations": 20, "relaxation": 0.1}),
        ("PrecomputedLandweber", "precomputed_landweber", {"iterations": 20, "relaxation": 0.1}),
        ("LSQR", "lsqr", {"iterations": 20, "damp": 0}),
        ("LSQR_Damped", "lsqr", {"iterations": 20, "damp": 1e-4}),
        ("TotalVariation", "total_variation", {"alpha": 1e-4, "iterations": 50, "tol": 1e-6}),
        ("L1", "l1", {"alpha": 1e-4, "iterations": 100, "tol": 1e-6}),
        ("ElasticNet", "elastic_net", {"alpha": 1e-4, "l1_ratio": 0.5, "iterations": 100, "tol": 1e-6})
    ]
    
    return presets


def get_combined_presets() -> List[Tuple[str, str, str, Dict[str, Any]]]:
    """
    スケーリングと正則化を組み合わせたプリセットを取得
    
    Returns:
        [("表示名", "スケーリング名", "正則化名", パラメータ辞書), ...] の形式のリスト
    """
    presets = [
        ("Basic", "none", "none", {}),
        ("Normalization", "normalization", "none", {}),
        ("Rehu", "rehu", "none", {}),
        ("Equalization", "equalization", "none", {}),
        ("Iterative", "iterative", "none", {"max_iter": 10, "tol": 1e-8}),
        ("Tikhonov", "none", "tikhonov", {"alpha": 1e-6}),
        ("Tikhonov_Strong", "none", "tikhonov", {"alpha": 1e-4}),
        ("SVD", "none", "svd", {"threshold": 1e-10}),
        ("TSVD", "none", "tsvd", {"threshold_ratio": 1e-5}),
        ("Landweber", "none", "landweber", {"iterations": 20, "relaxation": 0.1}),
        ("PrecomputedLandweber", "none", "precomputed_landweber", {"iterations": 20, "relaxation": 0.1}),
        ("LSQR", "none", "lsqr", {"iterations": 20, "damp": 0}),
        ("TotalVariation", "none", "total_variation", {"alpha": 1e-4, "iterations": 50, "tol": 1e-6}),
        ("L1", "none", "l1", {"alpha": 1e-4, "iterations": 100, "tol": 1e-6}),
        ("ElasticNet", "none", "elastic_net", {"alpha": 1e-4, "l1_ratio": 0.5, "iterations": 100, "tol": 1e-6}),
        # 組み合わせプリセット
        ("Rehu_Tikhonov", "rehu", "tikhonov", {"alpha": 1e-6}),
        ("Rehu_SVD", "rehu", "svd", {"threshold": 1e-10}),
        ("Normalization_Landweber", "normalization", "landweber", {"iterations": 20, "relaxation": 0.1}),
        ("Equalization_TSVD", "equalization", "tsvd", {"threshold_ratio": 1e-5}),
        ("VanDerSluis_TotalVariation", "van_der_sluis", "total_variation", {"alpha": 1e-4, "iterations": 50, "tol": 1e-6}),
        ("Iterative_ElasticNet", "iterative", "elastic_net", {"alpha": 1e-4, "l1_ratio": 0.5, "iterations": 100, "tol": 1e-8, "max_iter": 10})
    ]
    
    return presets


def get_preset_by_name(name: str) -> Tuple[str, str, Dict[str, Any]]:
    """
    名前からプリセット設定を取得
    
    Args:
        name: プリセット名
        
    Returns:
        (スケーリング名, 正則化名, パラメータ辞書)
    """
    for preset_name, scaling, regularization, params in get_combined_presets():
        if preset_name.lower() == name.lower():
            return scaling, regularization, params
    
    # デフォルト値を返す
    return "none", "none", {}