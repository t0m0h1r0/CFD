"""
代数的マルチグリッド前処理

このモジュールは、PyAMGライブラリを使用した
強力なマルチグリッド前処理手法を提供します。
特に楕円型PDEに対して優れた性能を発揮します。
"""

import numpy as np
import pyamg
from .base import BasePreconditioner

class AMGPreconditioner(BasePreconditioner):
    """代数的マルチグリッド前処理"""
    
    def __init__(self, max_levels=10, cycle_type='V', strength=0.0, 
                 presmoother=('gauss_seidel', {'sweep': 'forward'}),
                 postsmoother=('gauss_seidel', {'sweep': 'forward'}),
                 coarse_solver='direct', aggregate='standard',
                 smooth=('jacobi', {'omega': 4.0/3.0, 'degree': 1})):
        """
        初期化
        
        Args:
            max_levels: マルチグリッドの最大レベル数
            cycle_type: サイクルタイプ ('V', 'W', 'F')
            strength: 強連結性の閾値 (0～1)
            presmoother: 前スムーザー
            postsmoother: 後スムーザー
            coarse_solver: 粗いグリッドのソルバー
            aggregate: 集約方式 ('standard', 'lloyd', 'naive')
            smooth: 補間演算子のスムージング
        """
        super().__init__()
        self.max_levels = max_levels
        self.cycle_type = cycle_type
        self.strength = strength
        self.presmoother = presmoother
        self.postsmoother = postsmoother
        self.coarse_solver = coarse_solver
        self.aggregate = aggregate
        self.smooth = smooth
        self.ml = None
    
    def setup(self, A):
        """
        AMG前処理の設定
        
        Args:
            A: システム行列
            
        Returns:
            self: メソッドチェーン用
        """
        try:
            # CSR形式に変換（PyAMGは通常CSRを想定）
            if hasattr(A, 'tocsr'):
                A_csr = A.tocsr()
            elif hasattr(A, 'toarray'):
                from scipy import sparse
                A_csr = sparse.csr_matrix(A.toarray())
            else:
                from scipy import sparse
                A_csr = sparse.csr_matrix(A)
            
            # CuPy/JAX配列をNumPyに変換
            if hasattr(A_csr, 'get'):
                A_csr = A_csr.get()
            
            # 対称性チェック
            is_symmetric = self._check_symmetry(A_csr)
            
            print(f"PyAMGを使用してAMG前処理器を構築中... (行列サイズ: {A_csr.shape[0]}x{A_csr.shape[1]}, 対称性: {is_symmetric})")
            
            # 適切なAMGソルバーを選択
            if is_symmetric:
                # 対称的な場合はSmoothed Aggregation
                self.ml = pyamg.smoothed_aggregation_solver(
                    A_csr,
                    max_levels=self.max_levels,
                    strength=('symmetric', self.strength),
                    aggregate=self.aggregate,
                    smooth=self.smooth,
                    presmoother=self.presmoother,
                    postsmoother=self.postsmoother,
                    coarse_solver=self.coarse_solver
                )
            else:
                # 非対称的な場合はRuge-Stuben
                try:
                    self.ml = pyamg.ruge_stuben_solver(
                        A_csr,
                        max_levels=self.max_levels,
                        CF='RS',
                        presmoother=self.presmoother,
                        postsmoother=self.postsmoother,
                        coarse_solver=self.coarse_solver
                    )
                except Exception as e:
                    print(f"Ruge-Stuben法失敗 ({e})。Smoothed Aggregationへフォールバック")
                    self.ml = pyamg.smoothed_aggregation_solver(
                        A_csr,
                        max_levels=self.max_levels,
                        strength=('symmetric', self.strength),
                        aggregate=self.aggregate,
                        smooth=self.smooth,
                        presmoother=self.presmoother,
                        postsmoother=self.postsmoother,
                        coarse_solver=self.coarse_solver
                    )
            
            # 構築された多レベル構造の情報
            grid_complexity = self.ml.complexity
            operator_complexity = self.ml.operator_complexity
            
            print(f"AMG構造: {len(self.ml.levels)} レベル")
            print(f"グリッド複雑度: {grid_complexity:.2f}, 演算子複雑度: {operator_complexity:.2f}")
            
            # 各レベルの詳細情報
            for i, level in enumerate(self.ml.levels):
                if hasattr(level, 'A'):
                    A_level = level.A
                    n = A_level.shape[0]
                    nnz = A_level.nnz
                    nnz_per_row = nnz / n if n > 0 else 0
                    print(f"  レベル {i}: サイズ {n}x{n}, 非ゼロ要素 {nnz}, 行あたり {nnz_per_row:.1f}")
            
            return self
            
        except Exception as e:
            print(f"AMG前処理設定エラー: {e}")
            import traceback
            traceback.print_exc()
            self.ml = None
            return self
    
    def _check_symmetry(self, A, tol=1e-8):
        """
        行列の対称性をチェック
        
        Args:
            A: チェックする行列
            tol: 許容誤差
            
        Returns:
            対称かどうかを示すブール値
        """
        # 高速なサンプリングによる近似チェック
        n = A.shape[0]
        if n <= 100:
            # 小さな行列は完全チェック
            diff = A - A.T
            return abs(diff).max() < tol
        else:
            # 大きな行列はサンプリング
            samples = min(100, n // 10)
            indices = np.random.choice(n, samples, replace=False)
            
            for i in indices:
                for j in indices:
                    if i != j:
                        aij = A[i, j]
                        aji = A[j, i]
                        if abs(aij - aji) > tol * (abs(aij) + abs(aji) + 1e-10):
                            return False
            
            return True
    
    def __call__(self, b):
        """
        前処理を適用（近似解を返す）
        
        Args:
            b: 右辺ベクトル
            
        Returns:
            前処理したベクトル
        """
        if self.ml is None:
            return b
            
        # NumPy配列に変換
        if hasattr(b, 'get'):  # CuPy
            b_np = b.get()
        elif 'jax' in str(type(b)):  # JAX
            b_np = np.array(b)
        else:
            b_np = b
            
        # AMGソルバーを適用
        try:
            # 1回のマルチグリッドサイクルを適用
            x = self.ml.solve(b_np, tol=1e-12, maxiter=1, cycle=self.cycle_type)
            
            # 元の形式に戻す
            if 'cupy' in str(type(b)):  # CuPy
                import cupy as cp
                return cp.array(x)
            elif 'jax' in str(type(b)):  # JAX
                import jax.numpy as jnp
                return jnp.array(x)
            else:
                return x
                
        except Exception as e:
            print(f"AMG前処理適用エラー: {e}")
            return b
    
    @property
    def description(self):
        """前処理器の説明"""
        return f"代数的マルチグリッド前処理 (PyAMG, max_levels={self.max_levels}, cycle_type={self.cycle_type})"