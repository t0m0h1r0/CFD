# equation/poisson.py
import cupy as np
from typing import Dict, Callable
from .base import Equation


class EssentialEquation(Equation):
    def __init__(self, f_func: Callable[[float], float]):
        self.f_func = f_func

    def get_stencil_coefficients(
        self, i: int, n: int, h: float
    ) -> Dict[int, np.ndarray]:
        coeffs = {
            0: np.array([1, 0, 0, 0]),
        }

        return coeffs

    def get_rhs(self, i: int, n: int, h: float) -> float:
        x = i * h + self.grid.x_min if hasattr(self, "grid") else i * h
        return self.f_func(x)

    def is_valid_at(self, i: int, n: int) -> bool:
        return True
