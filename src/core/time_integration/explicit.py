from typing import TypeVar, Generic
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

def _euler_step(system_fn, t: float, y: State, dt: float) -> State:
    """Helper function for Euler step that can be jitted."""
    dy = system_fn(t, y)
    return system_fn.apply_update(y, dy, dt)

class ExplicitEuler(TimeIntegratorBase[State, Derivative]):
    """Implementation of explicit Euler method."""
    
    def __init__(self, config: TimeIntegrationConfig):
        """
        Initialize explicit Euler integrator.
        
        Args:
            config: Time integration configuration
        """
        super().__init__(config)
    
    def step(self,
            system: ODESystem[State, Derivative],
            t: float,
            y: State,
            dt: float) -> State:
        """
        Perform single Euler step.
        
        Args:
            system: ODE system to integrate
            t: Current time
            y: Current state
            dt: Time step size
            
        Returns:
            New state
        """
        # Use helper function to enable jit
        return _euler_step(system, t, y, dt)
    
    @staticmethod
    def get_order() -> int:
        """
        Get order of accuracy.
        
        Returns:
            Order of accuracy (1 for Euler)
        """
        return 1

class AdaptiveExplicitEuler(ExplicitEuler):
    """Adaptive explicit Euler method with error estimation."""
    
    def step(self,
            system: ODESystem[State, Derivative],
            t: float,
            y: State,
            dt: float) -> State:
        """
        Perform adaptive Euler step.
        
        Args:
            system: ODE system
            t: Current time
            y: Current state
            dt: Time step size
            
        Returns:
            New state
        """
        # Compute full step
        y_full = _euler_step(system, t, y, dt)
        
        # Compute two half steps
        dt_half = dt / 2
        y_half = _euler_step(system, t, y, dt_half)
        y_half2 = _euler_step(system, t + dt_half, y_half, dt_half)
        
        # Estimate error
        error = jnp.linalg.norm(y_full - y_half2)
        
        # Adjust time step if needed
        if self.config.adaptive_dt:
            tolerance = self.config.safety_factor * dt
            if error > tolerance:
                # Reduce time step
                dt_new = dt * jnp.sqrt(tolerance / error)
                return self.step(system, t, y, dt_new)
            
        return y_full

class SemiImplicitEuler(TimeIntegratorBase[State, Derivative]):
    """Implementation of semi-implicit Euler method."""
    
    def __init__(self,
                 config: TimeIntegrationConfig,
                 implicit_system: ODESystem[State, Derivative]):
        """
        Initialize semi-implicit Euler integrator.
        
        Args:
            config: Time integration configuration
            implicit_system: System to be treated implicitly
        """
        super().__init__(config)
        self.implicit_system = implicit_system
    
    def step(self,
            system: ODESystem[State, Derivative],
            t: float,
            y: State,
            dt: float) -> State:
        """
        Perform semi-implicit Euler step.
        
        Args:
            system: Explicit part of ODE system
            t: Current time
            y: Current state
            dt: Time step size
            
        Returns:
            New state
        """
        # Explicit part
        dy_explicit = system(t, y)
        y_star = system.apply_update(y, dy_explicit, dt)
        
        # Implicit part
        dy_implicit = self.implicit_system(t + dt, y_star)
        return system.apply_update(y_star, dy_implicit, dt)
    
    @staticmethod
    def get_order() -> int:
        """
        Get order of accuracy.
        
        Returns:
            Order of accuracy (1 for semi-implicit Euler)
        """
        return 1