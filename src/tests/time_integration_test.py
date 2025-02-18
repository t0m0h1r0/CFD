import os
import jax.numpy as jnp
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple

from src.core.time_integration.base import (
    TimeIntegrationConfig, 
    ODESystem, 
    TimeIntegratorBase
)
from src.core.time_integration.explicit import ExplicitEuler, RK4
from src.core.time_integration.runge_kutta import FehlbergRK45

class TimeIntegrationTestSuite:
    """Test suite for time integration schemes"""
    
    @dataclass
    class LinearOscillator:
        """Linear oscillator ODE system"""
        omega: float = 2 * jnp.pi
        
        def __call__(self, t: float, y: Tuple[float, float]) -> Tuple[float, float]:
            """Define ODE system for linear oscillator"""
            pos, vel = y
            return vel, -self.omega**2 * pos
        
        def apply_update(self, 
                         y: Tuple[float, float], 
                         dy: Tuple[float, float], 
                         dt: float) -> Tuple[float, float]:
            """Update state"""
            pos, vel = y
            dpos, dvel = dy
            return pos + dt * dpos, vel + dt * dvel
        
        def exact_solution(self, 
                           t: float, 
                           y0: Tuple[float, float]) -> Tuple[float, float]:
            """Analytical solution for linear oscillator"""
            pos0, vel0 = y0
            pos = pos0 * jnp.cos(self.omega * t) + vel0/self.omega * jnp.sin(self.omega * t)
            vel = -pos0 * self.omega * jnp.sin(self.omega * t) + vel0 * jnp.cos(self.omega * t)
            return pos, vel
    
    @classmethod
    def convergence_test(
        cls, 
        integrator_class: type[TimeIntegratorBase],
        system: ODESystem,
        y0: Tuple[float, float],
        t_final: float,
        dt_values: jnp.ndarray
    ) -> dict:
        """
        Perform convergence test for time integration schemes
        
        Args:
            integrator_class: Time integration scheme to test
            system: ODE system
            y0: Initial conditions
            t_final: Final time
            dt_values: Array of time step sizes to test
        
        Returns:
            Dictionary of convergence results
        """
        results = {}
        
        for dt in dt_values:
            # Create configuration
            config = TimeIntegrationConfig(
                dt=dt, 
                t_final=t_final,
                adaptive_dt=False
            )
            
            # Create integrator
            integrator = integrator_class(config)
            
            # Integrate
            y_final, _ = integrator.integrate(system, y0)
            
            # Compute exact solution
            y_exact = system.exact_solution(t_final, y0)
            
            # Compute error
            if isinstance(y_final, tuple):
                error = jnp.max(jnp.abs(jnp.array(y_final) - jnp.array(y_exact)))
            else:
                error = jnp.abs(y_final - y_exact)
            
            results[dt] = error
        
        return results
    
    @classmethod
    def plot_convergence(
        cls, 
        dt_values: jnp.ndarray, 
        results: dict, 
        title: str, 
        filename: str
    ) -> plt.Figure:
        """
        Plot convergence results
        
        Args:
            dt_values: Array of time step sizes
            results: Convergence results dictionary
            title: Plot title
            filename: Output filename
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot errors
        errors = jnp.array([results[dt] for dt in dt_values])
        ax.loglog(dt_values, errors, 'o-')
        
        # Compute convergence rate
        log_dt = jnp.log(dt_values)
        log_errors = jnp.log(errors)
        slope, _ = jnp.polyfit(log_dt, log_errors, 1)
        
        ax.set_xlabel('Time Step Size')
        ax.set_ylabel('Maximum Error')
        ax.set_title(f'{title} (Convergence Rate: {slope:.2f})')
        ax.grid(True, which='both', ls='-', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        
        return fig
    
    @classmethod
    def run_tests(cls):
        """Run comprehensive time integration tests"""
        # Create output directory
        os.makedirs('test_results/time_integration', exist_ok=True)
        
        # Test parameters
        y0 = (1.0, 0.0)  # Initial conditions
        t_final = 10.0
        dt_values = jnp.logspace(-3, -1, 10)
        
        # Integrators to test
        integrators = [
            ExplicitEuler,
            RK4,
            FehlbergRK45
        ]
        
        # Create system
        system = cls.LinearOscillator()
        
        # Store test results
        test_results = {}
        
        # Run convergence tests
        for integrator_class in integrators:
            name = integrator_class.__name__
            
            # Perform convergence test
            results = cls.convergence_test(
                integrator_class, 
                system, 
                y0, 
                t_final, 
                dt_values
            )
            
            # Plot convergence
            cls.plot_convergence(
                dt_values, 
                results, 
                f'Convergence of {name}', 
                f'test_results/time_integration/{name.lower()}_convergence.png'
            )
            
            # Compute convergence rate
            log_dt = jnp.log(dt_values)
            log_errors = jnp.log(jnp.array([results[dt] for dt in dt_values]))
            convergence_rate, _ = jnp.polyfit(log_dt, log_errors, 1)
            
            # Store results
            test_results[name] = {
                'results': results,
                'convergence_rate': float(convergence_rate)
            }
        
        # Print results
        print("Time Integration Test Results:")
        for name, result in test_results.items():
            print(f"{name}:")
            print(f"  Convergence Rate: {result['convergence_rate']:.4f}")
        
        return test_results

# Run tests when script is executed
if __name__ == '__main__':
    TimeIntegrationTestSuite.run_tests()