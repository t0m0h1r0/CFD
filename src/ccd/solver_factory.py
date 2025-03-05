"""
ソルバーファクトリーモジュール

異なる種類のCCDソルバーを生成するためのファクトリークラスを提供します。
"""

from typing import Optional, Dict, Any, List

from grid_config import GridConfig
from system_builder import CCDSystemBuilder
from ccd_solver import CCDSolver, VectorizedCCDSolver


class CCDSolverFactory:
    """CCD法ソルバーを生成するファクトリークラス"""
    
    @staticmethod
    def create_solver(
        grid_config: GridConfig,
        solver_type: str = "standard",
        use_iterative: bool = False,
        coeffs: Optional[List[float]] = None,
        solver_kwargs: Optional[Dict[str, Any]] = None,
        system_builder: Optional[CCDSystemBuilder] = None,
    ) -> CCDSolver:
        """
        適切なCCDソルバーインスタンスを生成
        
        Args:
            grid_config: グリッド設定
            solver_type: ソルバーの種類 ("standard" または "vectorized")
            use_iterative: 反復法を使用するかどうか
            coeffs: 係数 [a, b, c, d]
            solver_kwargs: ソルバーのパラメータ
            system_builder: システムビルダーのインスタンス
            
        Returns:
            設定されたCCDソルバーインスタンス
            
        Raises:
            ValueError: サポートされていないソルバータイプの場合
        """
        solver_kwargs = solver_kwargs or {}
        
        # ソルバーの種類に基づいてクラスを選択
        if solver_type.lower() == "standard":
            solver_class = CCDSolver
        elif solver_type.lower() == "vectorized":
            solver_class = VectorizedCCDSolver
        else:
            raise ValueError(f"サポートされていないソルバータイプ: {solver_type}")
        
        # ソルバーのインスタンスを作成
        return solver_class(
            grid_config=grid_config,
            coeffs=coeffs,
            use_iterative=use_iterative,
            solver_kwargs=solver_kwargs,
            system_builder=system_builder
        )
    
    @staticmethod
    def get_available_solver_types() -> List[str]:
        """
        利用可能なソルバータイプのリストを返す
        
        Returns:
            ソルバータイプのリスト
        """
        return ["standard", "vectorized"]
