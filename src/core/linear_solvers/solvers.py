from typing import Optional
from dataclasses import dataclass

import jax.numpy as jnp
from jax.typing import ArrayLike

from .base import PoissonSolverBase, PoissonSolverConfig, DerivativeOutput
from .iterative.cg import ConjugateGradient
from .iterative.sor import SORSolver
from ..spatial_discretization.operators.ccd import CombinedCompactDifference

@dataclass
class CCDCGConfig(PoissonSolverConfig):
    """CCD+CG法の設定"""
    use_preconditioned_cg: bool = True

class CCDCGSolver(PoissonSolverBase):
    """CCD法とCG法を組み合わせたポアソンソルバ"""
    
    def __init__(self,
                 discretization: CombinedCompactDifference,
                 config: Optional[CCDCGConfig] = None):
        """
        CCD+CGソルバの初期化
        
        Args:
            discretization: CCD離散化スキーム
            config: ソルバの設定
        """
        config = config or CCDCGConfig()
        linear_solver = ConjugateGradient(
            max_iterations=config.max_iterations,
            tolerance=config.tolerance
        )
        super().__init__(discretization, linear_solver, config)

@dataclass
class CCDSORConfig(PoissonSolverConfig):
    """CCD+SOR法の設定"""
    omega: float = 1.5  # 緩和係数

class CCDSORSolver(PoissonSolverBase):
    """CCD法とSOR法を組み合わせたポアソンソルバ"""
    
    def __init__(self,
                 discretization: CombinedCompactDifference,
                 config: Optional[CCDSORConfig] = None):
        """
        CCD+SORソルバの初期化
        
        Args:
            discretization: CCD離散化スキーム
            config: ソルバの設定
        """
        config = config or CCDSORConfig()
        linear_solver = SORSolver(
            omega=config.omega,
            max_iterations=config.max_iterations,
            tolerance=config.tolerance
        )
        super().__init__(discretization, linear_solver, config)