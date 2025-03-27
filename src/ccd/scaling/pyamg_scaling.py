# scaling/pyamg_scaling.py

import numpy as np
from scaling.base import ScalingBase

class PyAMGScaling(ScalingBase):
    """
    PyAMGを使用した代数的マルチグリッド前処理スケーリングプラグイン
    
    このプラグインは前処理子として機能し、以下の利点があります：
    1. 良質な初期値を生成（反復回数の削減）
    2. 大規模な疎行列系を効率的に解くための階層的手法を提供
    3. メッシュ細分化に対して堅牢な収束性を実現
    """
    
    def __init__(self, **kwargs):
        """
        PyAMG前処理スケーリングを初期化
        
        Args:
            **kwargs: AMGソルバーのための追加パラメータ
                method: AMG手法 ('smoothed_aggregation_solver', 'ruge_stuben_solver')
                strength: 接続強度手法 ('classical', 'evolution', etc.)
                smooth: スムーサータイプ ('jacobi', 'gauss_seidel', etc.)
                presweeps: 前平滑化ステップ数
                postsweeps: 後平滑化ステップ数
                coarse_solver: 粗格子ソルバー ('splu', 'lu', etc.)
                max_levels: 階層の最大レベル数
                max_coarse: 最粗格子の最大サイズ
                aggregate: 集約手法 ('standard', 'lloyd', etc.)
                initial_guess_cycles: 初期値生成用のAMGサイクル数
                final_cycles: 後処理改善用のAMGサイクル数
        """
        super().__init__()
        self.name = "PyAMGScaling"
        
        # AMGソルバーパラメータ
        self.amg_method = kwargs.get('method', 'smoothed_aggregation_solver')
        self.strength = kwargs.get('strength', 'classical')
        self.smooth = kwargs.get('smooth', 'jacobi')
        self.presweeps = kwargs.get('presweeps', 1)
        self.postsweeps = kwargs.get('postsweeps', 1)
        self.coarse_solver = kwargs.get('coarse_solver', 'splu')
        self.max_levels = kwargs.get('max_levels', 10)
        self.max_coarse = kwargs.get('max_coarse', 500)
        self.aggregate = kwargs.get('aggregate', 'standard')
        
        # 制御パラメータ
        self.initial_guess_cycles = kwargs.get('initial_guess_cycles', 5)
        self.final_cycles = kwargs.get('final_cycles', 2)
        self.initial_tol = kwargs.get('initial_tol', 1e-5)
        self.final_tol = kwargs.get('final_tol', 1e-8)
        
        # 内部状態
        self.ml = None  # マルチレベルソルバー
    
    def scale(self, A, b):
        """
        AMG前処理子を構築し、システムに適用準備を行う
        
        Args:
            A: システム行列 (scipy.sparse形式)
            b: 右辺ベクトル
            
        Returns:
            Tuple[scipy.sparse.spmatrix, numpy.ndarray, Dict]: 
                元の行列、元の右辺ベクトル、スケーリング情報
        """
        try:
            import pyamg
        except ImportError:
            print("警告: PyAMGが利用できません。pip install pyamgで導入してください。")
            return A, b, {}
        
        # PyAMG用にCSR形式に変換
        if not hasattr(A, 'format') or A.format != 'csr':
            try:
                A = A.tocsr()
            except:
                print("警告: 行列をCSR形式に変換できません。前処理なしで続行します。")
                return A, b, {}
        
        # AMGソルバーを作成
        try:
            # 理論的には、Poisson方程式にはSmoothed Aggregationが効果的
            if self.amg_method == 'smoothed_aggregation_solver':
                self.ml = pyamg.smoothed_aggregation_solver(
                    A, 
                    strength=self.strength,
                    smooth=self.smooth,
                    presweeps=self.presweeps,
                    postsweeps=self.postsweeps,
                    coarse_solver=self.coarse_solver,
                    max_levels=self.max_levels,
                    max_coarse=self.max_coarse,
                    aggregate=self.aggregate
                )
            elif self.amg_method == 'ruge_stuben_solver':
                # Ruge-Stuben法は古典的なAMGであり、より広範な問題に適用可能
                self.ml = pyamg.ruge_stuben_solver(
                    A, 
                    strength=self.strength,
                    presweeps=self.presweeps,
                    postsweeps=self.postsweeps,
                    coarse_solver=self.coarse_solver,
                    max_levels=self.max_levels,
                    max_coarse=self.max_coarse
                )
            else:
                print(f"警告: 未知のAMGメソッド: {self.amg_method}。smoothed_aggregationを使用します。")
                self.ml = pyamg.smoothed_aggregation_solver(A)
            
            # 階層構造情報を表示
            levels_info = ", ".join([f"Level {i}: {level.A.shape[0]}x{level.A.shape[1]}" 
                                    for i, level in enumerate(self.ml.levels)])
            print(f"AMG階層構造: {levels_info}")
            
            # 後で使用するための情報を保存
            scaling_info = {
                'ml': self.ml,
                'b': b,
                'initial_guess_cycles': self.initial_guess_cycles,
                'final_cycles': self.final_cycles,
                'initial_tol': self.initial_tol,
                'final_tol': self.final_tol
            }
            
            return A, b, scaling_info
            
        except Exception as e:
            import traceback
            print(f"警告: PyAMGソルバー生成エラー: {e}")
            traceback.print_exc()
            return A, b, {}
    
    def scale_b_only(self, b, scaling_info):
        """
        右辺ベクトルのみにスケーリングを適用（AMGを使用した良質な初期値生成）
        
        Args:
            b: 右辺ベクトル
            scaling_info: scale()から得たスケーリング情報
            
        Returns:
            numpy.ndarray: 元の右辺ベクトル
        """
        ml = scaling_info.get('ml')
        cycles = scaling_info.get('initial_guess_cycles', 5)
        tol = scaling_info.get('initial_tol', 1e-5)
        
        if ml is not None:
            try:
                # 厳密解ではなく、近似解としての初期値を生成
                print(f"AMGで{cycles}サイクルの初期値を生成中...")
                x0 = ml.solve(b, tol=tol, maxiter=cycles, cycle='V')
                scaling_info['x0'] = x0
                
                # 残差ノルムを計算して表示
                residual = np.linalg.norm(b - ml.levels[0].A.dot(x0)) / np.linalg.norm(b)
                print(f"AMG初期値生成完了: 残差ノルム = {residual:.6e}")
            except Exception as e:
                print(f"警告: AMG初期値生成エラー: {e}")
        
        # 右辺ベクトルは実際には変更しない
        return b
    
    def unscale(self, x, scaling_info):
        """
        AMGを使用して最終解を改善
        
        Args:
            x: 反復法からの解ベクトル
            scaling_info: スケーリング情報
            
        Returns:
            numpy.ndarray: 改善された解ベクトル
        """
        ml = scaling_info.get('ml')
        x0 = scaling_info.get('x0')
        b = scaling_info.get('b')
        cycles = scaling_info.get('final_cycles', 2)
        tol = scaling_info.get('final_tol', 1e-8)
        
        # scale_b_onlyで生成した良質な初期値の利用
        if ml is not None and x0 is not None and np.allclose(x, np.zeros_like(x)):
            # ソルバーがゼロ初期値を使用している場合、AMG生成の初期値を代わりに使用
            print("AMG生成の初期値を使用します")
            return x0
            
        # 最終解の品質向上のための追加AMGサイクル適用
        if ml is not None and cycles > 0 and b is not None:
            try:
                # 解の質を向上させるための後処理
                print(f"AMGで{cycles}サイクルの後処理を適用中...")
                improved_x = ml.solve(b, x0=x, tol=tol, maxiter=cycles, cycle='V')
                
                # 残差ノルムを計算して改善度を表示
                orig_res = np.linalg.norm(b - ml.levels[0].A.dot(x)) / np.linalg.norm(b)
                new_res = np.linalg.norm(b - ml.levels[0].A.dot(improved_x)) / np.linalg.norm(b)
                print(f"AMG後処理による改善: 残差ノルム {orig_res:.6e} → {new_res:.6e}")
                
                return improved_x
            except Exception as e:
                print(f"警告: AMG後処理エラー: {e}")
        
        return x