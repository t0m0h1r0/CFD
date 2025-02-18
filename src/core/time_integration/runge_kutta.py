from typing import Tuple, Optional, List
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

from .base import (
    TimeIntegratorBase,
    TimeIntegrationConfig,
    ODESystem,
    State,
    Derivative
)

@dataclass
class ButcherTableau:
    """Butcher tableau for Runge-Kutta methods."""
    a: jnp.ndarray  # Runge-Kutta matrix
    b: jnp.ndarray  # Weights
    c: jnp.ndarray  # Nodes
    order: int      # Order of accuracy
    
    @staticmethod
    def rk4() -> 'ButcherTableau':
        """Create classical RK4 tableau."""
        a = jnp.array([
            [0., 0., 0., 0.],
            [0.5, 0., 0., 0.],
            [0., 0.5, 0., 0.],
            [0., 0., 1., 0.]
        ])
        b = jnp.array([1/6, 1/3, 1/3, 1/6])
        c = jnp.array([0., 0.5, 0.5, 1.])
        return ButcherTableau(a=a, b=b, c=c, order=4)
    
    @staticmethod
    def fehlberg() -> Tuple['ButcherTableau', jnp.ndarray]:
        """Create Fehlberg RK4(5) tableau with error estimator."""
        a = jnp.zeros((6, 6))
        a = a.at[1, 0].set(1/4)
        a = a.at[2, 0:2].set(jnp.array([3/32, 9/32]))
        a = a.at[3, 0:3].set(jnp.array([1932/2197, -7200/2197, 7296/2197]))
        a = a.at[4, 0:4].set(jnp.array([439/216, -8, 3680/513, -845/4104]))
        a = a.at[5, 0:5].set(jnp.array([-8/27, 2, -3544/2565, 1859/4104, -11/40]))
        
        b4 = jnp.array([25/216, 0, 1408/2565, 2197/4104, -1/5, 0])
        b5 = jnp.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])
        
        c = jnp.array([0, 1/4, 3/8, 12/13, 1, 1/2])
        
        return ButcherTableau(a=a, b=b4, c=c, order=4), b5

def _rk_step(system_fn, tableau: ButcherTableau, t: float, y: State, dt: float) -> State:
    """Helper function for Runge-Kutta step that can be jitted."""
    # Initialize stage values
    k = []
    
    # Compute stage values
    for i in range(len(tableau.c)):
        t_stage = t + tableau.c[i] * dt
        y_stage = y
        
        # Add contributions from previous stages
        for j in range(i):
            if tableau.a[i, j] != 0:
                dy = system_fn.apply_update(
                    y,
                    k[j],
                    dt * tableau.a[i, j]
                )
                y_stage = dy
        
        # Compute stage derivative
        k.append(system_fn(t_stage, y_stage))
    
    # Compute final update
    y_new = y
    for i in range(len(k)):
        if tableau.b[i] != 0:
            dy = system_fn.apply_update(
                y,
                k[i],
                dt * tableau.b[i]
            )
            y_new = dy
            
    return y_new

class RungeKutta(TimeIntegratorBase[State, Derivative]):
    """General Runge-Kutta implementation."""
    
    def __init__(self,
                 config: TimeIntegrationConfig,
                 tableau: ButcherTableau):
        """
        Initialize Runge-Kutta integrator.
        
        Args:
            config: Time integration configuration
            tableau: Butcher tableau
        """
        super().__init__(config)
        self.tableau = tableau
    
    def step(self,
            system: ODESystem[State, Derivative],
            t: float,
            y: State,
            dt: float) -> State:
        """
        Perform single Runge-Kutta step.
        
        Args:
            system: ODE system to integrate
            t: Current time
            y: Current state
            dt: Time step size
            
        Returns:
            New state
        """
        return _rk_step(system, self.tableau, t, y, dt)
    
    def get_order(self) -> int:
        """
        Get order of accuracy.
        
        Returns:
            Order of accuracy from Butcher tableau
        """
        return self.tableau.order

class RK4(RungeKutta[State, Derivative]):
    """Classical fourth-order Runge-Kutta method."""
    
    def __init__(self, config: TimeIntegrationConfig):
        """
        Initialize RK4 integrator.
        
        Args:
            config: Time integration configuration
        """
        super().__init__(config, ButcherTableau.rk4())

class FehlbergRK45(RungeKutta[State, Derivative]):
    """Fehlberg's adaptive RK4(5) method."""
    
    def __init__(self,
                 config: TimeIntegrationConfig,
                 atol: float = 1e-6,
                 rtol: float = 1e-3):
        """
        Initialize Fehlberg RK4(5) integrator.
        
        Args:
            config: Time integration configuration
            atol: Absolute tolerance
            rtol: Relative tolerance
        """
        tableau, self.b_hat = ButcherTableau.fehlberg()
        super().__init__(config, tableau)
        self.atol = atol
        self.rtol = rtol
    
    def step(self,
            system: ODESystem[State, Derivative],
            t: float,
            y: State,
            dt: float) -> State:
        """
        Perform adaptive RK step.
        
        Args:
            system: ODE system to integrate
            t: Current time
            y: Current state
            dt: Time step size
            
        Returns:
            New state
        """
        # Compute both solutions
        y_low = _rk_step(system, self.tableau, t, y, dt)
        
        # For error estimation, create a tableau with b_hat
        tableau_high = ButcherTableau(
            a=self.tableau.a,
            b=self.b_hat,
            c=self.tableau.c,
            order=self.tableau.order + 1
        )
        y_high = _rk_step(system, tableau_high, t, y, dt)
        
        # Compute error estimate
        error = jnp.linalg.norm(y_high - y_low)
        scale = self.atol + jnp.linalg.norm(y_high) * self.rtol
        error_normalized = error / scale
        
        # Adjust time step if needed
        if self.config.adaptive_dt and error_normalized > 1.0:
            # Reduce time step using PI controller
            dt_new = max(
                0.1 * dt,
                0.9 * dt * (1.0 / error_normalized) ** (1.0 / (self.tableau.order + 1))
            )
            return self.step(system, t, y, dt_new)
        
        return y_high