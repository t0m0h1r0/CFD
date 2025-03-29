"""
代数的マルチグリッド前処理

このモジュールは、マルチグリッド手法を用いた
高効率な前処理手法を提供します。
特に楕円型PDEに対して優れた性能を発揮します。
"""

from .base import BasePreconditioner

class AMGPreconditioner(BasePreconditioner):
    """代数的マルチグリッド前処理"""
    
    def __init__(self, max_levels=10, cycle_type='V', strength_threshold=0.0):
        """
        初期化
        
        Args:
            max_levels: マルチグリッドの最大レベル数
            cycle_type: サイクルタイプ ('V' or 'W')
            strength_threshold: 粗視化のための閾値
        """
        super().__init__()
        self.max_levels = max_levels
        self.cycle_type = cycle_type
        self.strength_threshold = strength_threshold
        self.ml = None
        self.solver = None
    
    def setup(self, A):
        """
        AMG前処理の設定
        
        Args:
            A: システム行列
            
        Returns:
            self: メソッドチェーン用
        """
        try:
            from pyamg import smoothed_aggregation_solver
            
            # 代数的マルチグリッドのセットアップ
            self.ml = smoothed_aggregation_solver(
                A,
                max_levels=self.max_levels,
                strength=('symmetric', self.strength_threshold)
            )
            
            # ソルバークラスの定義
            self.solver = self.ml.aspreconditioner(cycle=self.cycle_type)
            
            # 前処理行列を設定
            self.M = self.solver
            
        except ImportError:
            print("AMG前処理にはpyamgパッケージが必要です。")
            print("次のコマンドでインストールしてください: pip install pyamg")
            self.M = None
        except Exception as e:
            print(f"AMG前処理設定エラー: {e}")
            self.M = None
        
        return self
    
    def __call__(self, b):
        """
        AMG前処理を適用
        
        Args:
            b: 右辺ベクトル
            
        Returns:
            M^-1 * b
        """
        if self.solver is not None:
            return self.solver(b)
        return b
    
    @property
    def name(self):
        """前処理名を返す"""
        return "AMGPreconditioner"
    
    @property
    def description(self):
        """前処理の説明を返す"""
        return f"代数的マルチグリッド前処理 (max_levels={self.max_levels}, cycle_type={self.cycle_type})"