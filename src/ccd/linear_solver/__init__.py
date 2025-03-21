"""
線形方程式系 Ax=b を解くためのソルバーモジュール

このパッケージは、様々なバックエンド（CPU/SciPy、GPU/CuPy、JAX）を使用して
線形方程式系を効率的に解くためのクラス群を提供します。
"""

# 明示的にエクスポートするクラスとモジュールを指定
__all__ = ['LinearSolver', 'CPULinearSolver', 'GPULinearSolver', 'JAXLinearSolver', 'create_solver']

# ベースクラスと実装クラスをインポート
from .base import LinearSolver
from .cpu_solver import CPULinearSolver
from .gpu_solver import GPULinearSolver
from .jax_solver import JAXLinearSolver

def create_solver(A, enable_dirichlet=False, enable_neumann=False, scaling_method=None, backend="cuda"):
    """
    適切な線形ソルバーを作成するファクトリ関数
    
    Args:
        A: システム行列
        enable_dirichlet: ディリクレ境界条件を使用するか
        enable_neumann: ノイマン境界条件を使用するか
        scaling_method: スケーリング手法名（オプション）
        backend: 計算バックエンド ("cpu", "cuda", "jax")
        
    Returns:
        LinearSolver: 適切なソルバーインスタンス
    """
    # バックエンドに応じたソルバークラスを選択
    if backend == "cuda":
        solver_class = GPULinearSolver
    elif backend == "jax":
        solver_class = JAXLinearSolver
    else:
        solver_class = CPULinearSolver
    
    try:
        return solver_class(A, enable_dirichlet, enable_neumann, scaling_method)
    except Exception as e:
        print(f"{backend}ソルバー初期化エラー: {e}")
        if backend != "cpu":
            print("CPUソルバーにフォールバック")
            return CPULinearSolver(A, enable_dirichlet, enable_neumann, scaling_method)
        raise

# 便宜上のエイリアス
create = create_solver