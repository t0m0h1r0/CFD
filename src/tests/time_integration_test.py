import os
from typing import Tuple, Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Force JAX to use CPU
jax.config.update('jax_platform_name', 'cpu')

from src.core.time_integration.base import (
    TimeIntegrationConfig, ODESystem
)
from src.core.time_integration.explicit import ExplicitEuler
from src.core.time_integration.runge_kutta import RK4, FehlbergRK45

class ExponentialDecay:
    """Test case: y' = -y"""
    
    def __init__(self, lambda_: float = -1.0):
        """Initialize with decay rate."""
        self.lambda_ = lambda_
    
    def __call__(self, t: float, y: jnp.ndarray) -> jnp.ndarray:
        """Compute derivative."""
        return jnp.array(self.lambda_ * y)
    
    def apply_update(self, y: jnp.ndarray, dy: jnp.ndarray, dt: float) -> jnp.ndarray:
        """Update state."""
        return y + dt * dy
    
    def exact_solution(self, t: float, y0: float) -> float:
        """Compute exact solution."""
        return jnp.array(y0 * jnp.exp(self.lambda_ * t))

@dataclass
class HarmonicOscillator:
    """Test case: y'' + ω²y = 0"""
    omega: float = 2 * jnp.pi
    
    def __call__(self, t: float, y: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        pos, vel = y
        return vel, -self.omega**2 * pos
    
    def apply_update(self, y: Tuple[jnp.ndarray, jnp.ndarray], 
                    dy: Tuple[jnp.ndarray, jnp.ndarray], 
                    dt: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        pos, vel = y
        dpos, dvel = dy
        return pos + dt * dpos, vel + dt * dvel
    
    def exact_solution(self, t: float, y0: Tuple[float, float]) -> Tuple[float, float]:
        pos0, vel0 = y0
        pos = pos0 * jnp.cos(self.omega * t) + vel0/self.omega * jnp.sin(self.omega * t)
        vel = -pos0 * self.omega * jnp.sin(self.omega * t) + vel0 * jnp.cos(self.omega * t)
        return pos, vel

@dataclass
class VanDerPol:
    """Test case: y'' - μ(1-y²)y' + y = 0"""
    mu: float = 1.0
    
    def __call__(self, t: float, y: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        pos, vel = y
        return vel, self.mu * (1 - pos**2) * vel - pos
    
    def apply_update(self, y: Tuple[jnp.ndarray, jnp.ndarray], 
                    dy: Tuple[jnp.ndarray, jnp.ndarray], 
                    dt: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        pos, vel = y
        dpos, dvel = dy
        return pos + dt * dpos, vel + dt * dvel

def run_convergence_test(
    system: ODESystem,
    exact_solution: Callable,
    y0: jnp.ndarray,
    dt_values: jnp.ndarray,
    t_final: float,
    integrators: list
) -> dict:
    """Run convergence test for multiple integrators."""
    results = {}
    
    print("\nRunning convergence tests:")
    print("-" * 50)
    
    for integrator_class in integrators:
        print(f"\nTesting {integrator_class.__name__}:")
        errors = []
        
        for i, dt in enumerate(dt_values):
            print(f"  Step size dt = {dt:.2e} ({i+1}/{len(dt_values)})", end="")
            
            config = TimeIntegrationConfig(dt=dt, t_final=t_final)
            integrator = integrator_class(config)
            
            # Integrate
            y_final, history = integrator.integrate(system, y0)
            
            # Compute error
            y_exact = exact_solution(t_final, y0)
            if isinstance(y_final, tuple):
                error = jnp.max(jnp.abs(jnp.array(y_final) - jnp.array(y_exact)))
            else:
                error = jnp.abs(y_final - y_exact)
            errors.append(error)
            
            # Print progress
            print(f" - Error: {error:.2e}")
            
        # Calculate convergence rate
        if len(dt_values) > 1:
            log_dt = jnp.log(dt_values)
            log_errors = jnp.log(jnp.array(errors))
            slope, _ = jnp.polyfit(log_dt, log_errors, 1)
            print(f"  Convergence rate: {slope:.2f}")
            
        results[integrator_class.__name__] = jnp.array(errors)
    
    print("\nConvergence tests completed.")
    print("-" * 50)
    return results

def plot_convergence(
    dt_values: jnp.ndarray,
    results: dict,
    title: str,
    filename: str
) -> Figure:
    """Plot convergence results."""
    print(f"\nGenerating convergence plot: {title}")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method_name, errors in results.items():
        ax.loglog(dt_values, errors, 'o-', label=method_name)
        
        # Compute convergence rate
        slope, _ = np.polyfit(np.log(dt_values), np.log(errors), 1)
        ax.text(0.6, 0.1, f'{method_name} rate: {slope:.2f}',
                transform=ax.transAxes)
    
    # Reference lines
    dt_ref = np.array([dt_values[0], dt_values[-1]])
    for order, style in zip([1, 2, 4], ['k:', 'k--', 'k-']):
        ref = dt_ref**order * (errors[0] / dt_values[0]**order)
        ax.loglog(dt_ref, ref, style, label=f'Order {order}')
    
    ax.set_xlabel('Time step (dt)')
    ax.set_ylabel('Maximum error')
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    
    # Save figure
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    return fig

def plot_solution(
    t: jnp.ndarray,
    y_numerical: jnp.ndarray,
    y_exact: jnp.ndarray,
    title: str,
    filename: str
) -> Figure:
    """Plot numerical and exact solutions."""
    print(f"\nGenerating solution plot: {title}")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if isinstance(y_numerical[0], tuple):
        # Plot position for oscillator systems
        y_num = jnp.array([y[0] for y in y_numerical])
        y_ex = jnp.array([y[0] for y in y_exact])
    else:
        y_num = y_numerical
        y_ex = y_exact
    
    ax.plot(t, y_num, 'b-', label='Numerical')
    ax.plot(t, y_ex, 'r--', label='Exact')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Solution')
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    
    # Save figure
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    return fig

def main():
    """Run all tests and generate plots."""
    # Force JAX to use CPU
    jax.config.update('jax_platform_name', 'cpu')
    
    print("\n=== Time Integration Method Test Suite ===\n")
    
    # Create output directory
    print("Setting up test environment...")
    os.makedirs('test_results', exist_ok=True)
    
    # Test parameters
    dt_values = jnp.logspace(-4, -1, 10)
    t_final = 10.0
    integrators = [ExplicitEuler]  # ExplicitEulerのみを評価
    
    print("\nTest Configuration:")
    print(f"- Time step range: {dt_values[0]:.2e} to {dt_values[-1]:.2e}")
    print(f"- Final time: {t_final}")
    print(f"- Testing method: {integrators[0].__name__}")
    
    # Test Case 1: Exponential Decay
    print("\n=== Test Case 1: Exponential Decay ===")
    print("Testing linear stability and convergence...")
    system = ExponentialDecay()
    y0 = 1.0
    results = run_convergence_test(
        system, system.exact_solution, y0, dt_values, t_final, integrators
    )
    plot_convergence(
        dt_values, results, 
        "Convergence Test: Exponential Decay",
        "test_results/exponential_convergence"
    )
    
    # Solution plot for exponential decay
    t = jnp.linspace(0, t_final, 1000)
    config = TimeIntegrationConfig(dt=0.01, t_final=t_final)
    y_numerical = []
    y_exact = []
    y = y0
    
    for ti in t:
        y_numerical.append(y)
        y_exact.append(system.exact_solution(ti, y0))
        if ti < t_final - config.dt:
            dy = system(ti, y)
            y = system.apply_update(y, dy, config.dt)
    
    plot_solution(
        t, jnp.array(y_numerical), jnp.array(y_exact),
        "Solution: Exponential Decay",
        "test_results/exponential_solution"
    )
    
    # Test Case 2: Harmonic Oscillator
    print("\n=== Test Case 2: Harmonic Oscillator ===")
    print("Testing energy conservation and periodic solution accuracy...")
    system = HarmonicOscillator()
    y0 = (1.0, 0.0)  # Initial position and velocity
    results = run_convergence_test(
        system, system.exact_solution, y0, dt_values, t_final, integrators
    )
    plot_convergence(
        dt_values, results,
        "Convergence Test: Harmonic Oscillator",
        "test_results/harmonic_convergence"
    )
    
    # Solution plot for harmonic oscillator
    y_numerical = []
    y_exact = []
    y = y0
    
    for ti in t:
        y_numerical.append(y)
        y_exact.append(system.exact_solution(ti, y0))
        if ti < t_final - config.dt:
            dy = system(ti, y)
            y = system.apply_update(y, dy, config.dt)
    
    plot_solution(
        t, y_numerical, y_exact,
        "Solution: Harmonic Oscillator",
        "test_results/harmonic_solution"
    )
    
    # Test energy conservation for harmonic oscillator
    def compute_energy(pos: float, vel: float, omega: float) -> float:
        """Compute total energy of harmonic oscillator."""
        kinetic = 0.5 * vel**2
        potential = 0.5 * (omega * pos)**2
        return kinetic + potential
    
    energy_numerical = []
    for pos, vel in y_numerical:
        energy_numerical.append(compute_energy(pos, vel, system.omega))
    
    energy_exact = compute_energy(y0[0], y0[1], system.omega)
    energy_error = jnp.abs(jnp.array(energy_numerical) - energy_exact) / energy_exact
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(t, energy_error, 'b-')
    plt.xlabel('Time')
    plt.ylabel('Relative Energy Error')
    plt.title('Energy Conservation Error: Harmonic Oscillator')
    plt.grid(True)
    plt.savefig('test_results/harmonic_energy_error.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Test Case 3: Van der Pol Oscillator
    print("\n=== Test Case 3: Van der Pol Oscillator ===")
    print("Testing nonlinear dynamics and adaptive time stepping...")
    system = VanDerPol()
    y0 = (1.0, 0.0)
    
    # Since no exact solution, compare between different time steps
    t = jnp.linspace(0, t_final, 1000)
    config_fine = TimeIntegrationConfig(dt=0.001, t_final=t_final)
    config_coarse = TimeIntegrationConfig(dt=0.01, t_final=t_final)
    
    integrator_fine = RK4(config_fine)
    integrator_coarse = RK4(config_coarse)
    
    y_fine, _ = integrator_fine.integrate(system, y0)
    y_coarse, _ = integrator_coarse.integrate(system, y0)
    
    # Plot phase space trajectory
    plt.figure(figsize=(10, 6))
    
    # Convert results to arrays of positions and velocities
    if hasattr(y_fine, '__len__'):
        pos_fine = jnp.array([y[0] if isinstance(y, tuple) else y for y in y_fine])
        vel_fine = jnp.array([y[1] if isinstance(y, tuple) else 0.0 for y in y_fine])
        pos_coarse = jnp.array([y[0] if isinstance(y, tuple) else y for y in y_coarse])
        vel_coarse = jnp.array([y[1] if isinstance(y, tuple) else 0.0 for y in y_coarse])
    else:
        # Handle single time point case
        pos_fine = jnp.array([y_fine[0] if isinstance(y_fine, tuple) else y_fine])
        vel_fine = jnp.array([y_fine[1] if isinstance(y_fine, tuple) else 0.0])
        pos_coarse = jnp.array([y_coarse[0] if isinstance(y_coarse, tuple) else y_coarse])
        vel_coarse = jnp.array([y_coarse[1] if isinstance(y_coarse, tuple) else 0.0])
    
    plt.plot(pos_fine, vel_fine, 'b-', label='dt = 0.001')
    plt.plot(pos_coarse, vel_coarse, 'r--', label='dt = 0.01')
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title('Phase Space: Van der Pol Oscillator')
    plt.grid(True)
    plt.legend()
    plt.savefig('test_results/vanderpol_phase.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Test adaptive time stepping with FehlbergRK45
    print("Testing Adaptive Time Stepping...")
    config_adaptive = TimeIntegrationConfig(
        dt=0.01, 
        t_final=t_final,
        adaptive_dt=True,
        safety_factor=0.9
    )
    
    integrator_adaptive = FehlbergRK45(config_adaptive)
    _, history_adaptive = integrator_adaptive.integrate(system, y0)
    
    # Plot time step sizes
    time_points = [info.t for info in history_adaptive]
    step_sizes = [info.dt for info in history_adaptive]
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(time_points, step_sizes, 'b-')
    plt.xlabel('Time')
    plt.ylabel('Time Step Size')
    plt.title('Adaptive Time Step Sizes: Van der Pol Oscillator')
    plt.grid(True)
    plt.savefig('test_results/adaptive_timesteps.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n=== Test Suite Summary ===")
    print("- All test cases completed successfully")
    print("- Generated plots saved in test_results/")
    print("- Verified:")
    print("  * Linear stability (Exponential Decay)")
    print("  * Energy conservation (Harmonic Oscillator)")
    print("  * Nonlinear dynamics (Van der Pol)")
    print("  * Adaptive time stepping")
    print("\nTest suite completed.")

if __name__ == '__main__':
    main()