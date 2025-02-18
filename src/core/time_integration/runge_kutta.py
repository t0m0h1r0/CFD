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
    def heun() -> 'ButcherTableau':
        """Create Heun's method tableau (RK2)."""
        a = jnp.array([[0., 0.], [1., 0.]])
        b = jnp.array([0.5, 0.5])
        c = jnp.array([0., 1.])
        return ButcherTableau(a=a, b=b, c=c, order=2)
    
    @staticmethod
    def fehlberg() -> 'ButcherTableau':
        """Create Fehlberg RK4(5) tableau for error estimation."""
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

class RungeKutta(TimeIntegratorBase[State, Derivative]):
    """General Runge-Kutta implementation."""
    
    def __init__(self,
                 config: TimeIntegrationConfig,
                 tableau: ButcherTableau):
        """
        Initialize Runge-Kutta integrator.
        
        Args:
            config: Time integration configuration
            tableau: Butcher tableau defining the method
        """
        super().__init__(config)
        self.tableau = tableau
        self.stages = len(tableau.b)
    
    @partial(jax.jit, static_argnums=(0, 1))
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
        # Initialize stage values
        k = [None] * self.stages
        
        # Compute stage values
        for i in range(self.stages):
            # Compute stage time
            t_stage = t + self.tableau.c[i] * dt
            
            # Initialize stage state
            y_stage = y
            
            # Add contributions from previous stages
            for j in range(i):
                if self.tableau.a[i, j] != 0:
                    dy = system.apply_update(
                        y,
                        k[j],
                        dt * self.tableau.a[i, j]
                    )
                    y_stage = dy
            
            # Compute stage derivative
            k[i] = system(t_stage, y_stage)
        
        # Compute final update
        y_new = y
        for i in range(self.stages):
            if self.tableau.b[i] != 0:
                dy = system.apply_update(
                    y,
                    k[i],
                    dt * self.tableau.b[i]
                )
                y_new = dy
                
        return y_new
    
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

class AdaptiveRK(RungeKutta[State, Derivative]):
    """Adaptive Runge-Kutta with error estimation."""
    
    def __init__(self,
                 config: TimeIntegrationConfig,
                 tableau: ButcherTableau,
                 b_hat: jnp.ndarray,
                 atol: float = 1e-6,
                 rtol: float = 1e-3):
        """
        Initialize adaptive RK integrator.
        
        Args:
            config: Time integration configuration
            tableau: Butcher tableau
            b_hat: Weights for error estimation
            atol: Absolute tolerance
            rtol: Relative tolerance
        """
        super().__init__(config, tableau)
        self.b_hat = b_hat
        self.atol = atol
        self.rtol = rtol
    
    def step(self,
            system: ODESystem[State, Derivative],
            t: float,
            y: State,
            dt: float) -> tuple[State, float]:
        """
        Perform adaptive RK step.
        
        Args:
            system: ODE system to integrate
            t: Current time
            y: Current state
            dt: Time step size
            
        Returns:
            Tuple of (new_state, error_estimate)
        """
        # Compute stage values as in base class
        k = [None] * self.stages
        for i in range(self.stages):
            t_stage = t + self.tableau.c[i] * dt
            y_stage = y
            
            for j in range(i):
                if self.tableau.a[i, j] != 0:
                    dy = system.apply_update(y, k[j], dt * self.tableau.a[i, j])
                    y_stage = dy
            
            k[i] = system(t_stage, y_stage)
        
        # Compute solutions with both sets of weights
        y_new = y
        y_hat = y
        
        for i in range(self.stages):
            if self.tableau.b[i] != 0:
                dy = system.apply_update(y, k[i], dt * self.tableau.b[i])
                y_new = dy
            if self.b_hat[i] != 0:
                dy = system.apply_update(y, k[i], dt * self.b_hat[i])
                y_hat = dy
        
        # Compute error estimate
        error = jnp.linalg.norm(y_new - y_hat)
        scale = self.atol + jnp.linalg.norm(y_new) * self.rtol
        error_normalized = error / scale
        
        # Adjust time step if needed
        if self.config.adaptive_dt and error_normalized > 1.0:
            # Reduce time step using PI controller
            dt_new = max(
                0.1 * dt,
                0.9 * dt * (1.0 / error_normalized) ** (1.0 / (self.tableau.order + 1))
            )
            return self.step(system, t, y, dt_new)
        
        return y_new, error_normalized

class FehlbergRK45(AdaptiveRK[State, Derivative]):
    """Fehlberg's adaptive RK4(5) method."""
    
    def __init__(self, config: TimeIntegrationConfig):
        """
        Initialize Fehlberg RK4(5) integrator.
        
        Args:
            config: Time integration configuration
        """
        tableau, b_hat = ButcherTableau.fehlberg()
        super().__init__(config, tableau, b_hat)