from abc import ABC, abstractmethod
from typing import Protocol, Callable, TypeVar, Generic, Tuple, Optional
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

State = TypeVar('State')
Derivative = TypeVar('Derivative')

class ODESystem(Protocol, Generic[State, Derivative]):
    """Protocol for ODE system to be integrated."""
    def __call__(self, t: float, y: State) -> Derivative:
        """
        Compute right-hand side of ODE system.
        
        Args:
            t: Current time
            y: Current state
            
        Returns:
            Time derivative of state
        """
        ...
        
    def apply_update(self, y: State, dy: Derivative, dt: float) -> State:
        """
        Apply state update.
        
        Args:
            y: Current state
            dy: State derivative
            dt: Time step
            
        Returns:
            Updated state
        """
        ...

@dataclass
class TimeIntegrationConfig:
    """Configuration for time integration."""
    dt: float
    t_final: float
    save_frequency: int = 1
    check_stability: bool = True
    adaptive_dt: bool = False
    safety_factor: float = 0.9
    
    def validate(self) -> None:
        """Validate configuration."""
        if self.dt <= 0:
            raise ValueError(f"Invalid time step: {self.dt}")
        if self.t_final <= 0:
            raise ValueError(f"Invalid final time: {self.t_final}")
        if self.save_frequency <= 0:
            raise ValueError(f"Invalid save frequency: {self.save_frequency}")
        if self.safety_factor <= 0 or self.safety_factor >= 1:
            raise ValueError(f"Invalid safety factor: {self.safety_factor}")

@dataclass
class TimeStepInfo:
    """Information about current time step."""
    t: float
    dt: float
    step_number: int
    is_saved: bool = False
    stability_number: Optional[float] = None

class TimeIntegratorBase(ABC, Generic[State, Derivative]):
    """Base class for time integration schemes."""
    
    def __init__(self, config: TimeIntegrationConfig):
        """
        Initialize time integrator.
        
        Args:
            config: Time integration configuration
        """
        self.config = config
        self.config.validate()
        
    @abstractmethod
    def step(self,
            system: ODESystem[State, Derivative],
            t: float,
            y: State,
            dt: float) -> State:
        """
        Perform single time step.
        
        Args:
            system: ODE system to integrate
            t: Current time
            y: Current state
            dt: Time step size
            
        Returns:
            New state
        """
        pass
        
    def integrate(self,
                 system: ODESystem[State, Derivative],
                 y0: State,
                 callback: Optional[Callable[[State, TimeStepInfo], None]] = None
                 ) -> Tuple[State, list[TimeStepInfo]]:
        """
        Integrate system from t=0 to t_final.
        
        Args:
            system: ODE system to integrate
            y0: Initial state
            callback: Optional callback function for each time step
            
        Returns:
            Tuple of (final_state, time_step_info_list)
        """
        t = 0.0
        y = y0
        step_number = 0
        history = []
        
        while t < self.config.t_final:
            # Adjust final step size if needed
            dt = min(self.config.dt, self.config.t_final - t)
            
            # Record current step info
            info = TimeStepInfo(
                t=t,
                dt=dt,
                step_number=step_number,
                is_saved=step_number % self.config.save_frequency == 0
            )
            history.append(info)
            
            # Optional callback
            if callback is not None and info.is_saved:
                callback(y, info)
            
            # Take time step
            y = self.step(system, t, y, dt)
            
            # Update time and step counter
            t += dt
            step_number += 1
            
        return y, history
    
    def estimate_stability_number(self,
                                system: ODESystem[State, Derivative],
                                y: State,
                                t: float) -> float:
        """
        Estimate stability number for current state.
        
        Args:
            system: ODE system
            y: Current state
            t: Current time
            
        Returns:
            Stability number estimate
        """
        dy = system(t, y)
        dydt_norm = jnp.linalg.norm(dy)
        y_norm = jnp.linalg.norm(y)
        
        if y_norm > 0:
            return (dydt_norm / y_norm) * self.config.dt
        else:
            return dydt_norm * self.config.dt
            
    def check_stability(self,
                       stability_number: float,
                       threshold: float = 1.0) -> bool:
        """
        Check if integration is stable.
        
        Args:
            stability_number: Current stability number
            threshold: Stability threshold
            
        Returns:
            True if integration is stable
        """
        return stability_number <= threshold
    
    @staticmethod
    def get_order() -> int:
        """
        Get order of accuracy of the scheme.
        
        Returns:
            Order of accuracy
        """
        return 1  # Default to first order