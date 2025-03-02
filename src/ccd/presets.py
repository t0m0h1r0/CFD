"""
プリセット設定モジュール

CCDソルバーのプリセット設定を定義し、プリセット設定を取得するための関数を提供します。
"""

from typing import List, Tuple, Dict, Any

# 統合ソルバーから利用可能な戦略を取得
from composite_solver import CCDCompositeSolver


def get_scaling_presets() -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    スケーリングプリセットを取得

    Returns:
        [("表示名", "スケーリング名", パラメータ辞書), ...] の形式のリスト
    """
    # プラグインをロードして利用可能なスケーリング戦略を取得
    scaling_methods = CCDCompositeSolver.available_scaling_methods()

    # 基本プリセットを定義
    presets = [
        ("None", "none", {}),
    ]

    # 利用可能なスケーリング戦略に基づいて追加
    if "normalization" in scaling_methods:
        presets.append(("Normalization", "normalization", {}))

    if "rehu" in scaling_methods:
        presets.append(("Rehu", "rehu", {}))

    if "equalization" in scaling_methods:
        presets.append(("Equalization", "equalization", {}))

    if "iterative" in scaling_methods:
        presets.append(("Iterative", "iterative", {"max_iter": 10, "tol": 1e-8}))

    if "van_der_sluis" in scaling_methods:
        presets.append(("VanDerSluis", "van_der_sluis", {}))

    if "diagonal_dominance" in scaling_methods:
        presets.append(("DiagonalDominance", "diagonal_dominance", {}))

    if "square_sum" in scaling_methods:
        presets.append(("SquareSum", "square_sum", {}))

    if "max_element" in scaling_methods:
        presets.append(("MaxElement", "max_element", {}))

    return presets


def get_regularization_presets() -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    正則化プリセットを取得

    Returns:
        [("表示名", "正則化名", パラメータ辞書), ...] の形式のリスト
    """
    # プラグインをロードして利用可能な正則化戦略を取得
    regularization_methods = CCDCompositeSolver.available_regularization_methods()

    # 基本プリセットを定義
    presets = [
        ("None", "none", {}),
    ]

    # 利用可能な正則化戦略に基づいて追加
    if "tikhonov" in regularization_methods:
        presets.extend(
            [
                ("Tikhonov", "tikhonov", {"alpha": 1e-6}),
                ("Tikhonov_Strong", "tikhonov", {"alpha": 1e-4}),
            ]
        )

    if "svd" in regularization_methods:
        presets.append(("SVD", "svd", {"threshold": 1e-10}))

    if "tsvd" in regularization_methods:
        presets.extend(
            [
                ("TSVD", "tsvd", {"threshold_ratio": 1e-5}),
                ("TSVD_10", "tsvd", {"rank": 10}),
            ]
        )

    if "landweber" in regularization_methods:
        presets.append(
            ("Landweber", "landweber", {"iterations": 20, "relaxation": 0.1})
        )

    if "precomputed_landweber" in regularization_methods:
        presets.append(
            (
                "PrecomputedLandweber",
                "precomputed_landweber",
                {"iterations": 20, "relaxation": 0.1},
            )
        )

    if "lsqr" in regularization_methods:
        presets.extend(
            [
                ("LSQR", "lsqr", {"iterations": 20, "damp": 0}),
                ("LSQR_Damped", "lsqr", {"iterations": 20, "damp": 1e-4}),
            ]
        )

    if "total_variation" in regularization_methods:
        presets.append(
            (
                "TotalVariation",
                "total_variation",
                {"alpha": 1e-4, "iterations": 50, "tol": 1e-6},
            )
        )

    if "l1" in regularization_methods:
        presets.append(("L1", "l1", {"alpha": 1e-4, "iterations": 100, "tol": 1e-6}))

    if "elastic_net" in regularization_methods:
        presets.append(
            (
                "ElasticNet",
                "elastic_net",
                {"alpha": 1e-4, "l1_ratio": 0.5, "iterations": 100, "tol": 1e-6},
            )
        )

    return presets


def get_combined_presets() -> List[Tuple[str, str, str, Dict[str, Any]]]:
    """
    スケーリングと正則化を組み合わせたプリセットを取得

    Returns:
        [("表示名", "スケーリング名", "正則化名", パラメータ辞書), ...] の形式のリスト
    """
    # 個別のプリセットを取得
    scaling_presets = get_scaling_presets()
    regularization_presets = get_regularization_presets()

    # 基本的な組み合わせを作成
    presets = []

    # スケーリングのみのプリセット
    for name, scaling, params in scaling_presets:
        presets.append((name, scaling, "none", params.copy()))

    # 正則化のみのプリセット（"None"は重複するため除外）
    for name, regularization, params in regularization_presets:
        if name != "None":  # スケーリングの"None"と重複を避けるため
            presets.append((name, "none", regularization, params.copy()))

    # 組み合わせプリセット
    # 特定の組み合わせを追加
    combinations = [
        ("Rehu_Tikhonov", "rehu", "tikhonov", {"alpha": 1e-6}),
        ("Rehu_SVD", "rehu", "svd", {"threshold": 1e-10}),
        (
            "Normalization_Landweber",
            "normalization",
            "landweber",
            {"iterations": 20, "relaxation": 0.1},
        ),
        ("Equalization_TSVD", "equalization", "tsvd", {"threshold_ratio": 1e-5}),
        (
            "VanDerSluis_TotalVariation",
            "van_der_sluis",
            "total_variation",
            {"alpha": 1e-4, "iterations": 50, "tol": 1e-6},
        ),
        (
            "Iterative_ElasticNet",
            "iterative",
            "elastic_net",
            {
                "alpha": 1e-4,
                "l1_ratio": 0.5,
                "iterations": 100,
                "tol": 1e-8,
                "max_iter": 10,
            },
        ),
    ]

    # 利用可能な戦略のみを組み合わせプリセットに追加
    scaling_methods = CCDCompositeSolver.available_scaling_methods()
    regularization_methods = CCDCompositeSolver.available_regularization_methods()

    for name, scaling, regularization, params in combinations:
        if scaling in scaling_methods and regularization in regularization_methods:
            presets.append((name, scaling, regularization, params.copy()))

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
