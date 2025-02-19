import os
import unittest
from typing import Dict, Optional
from dataclasses import dataclass

import jax.numpy as jnp
import matplotlib.pyplot as plt

from src.core.time_integration.base import TimeIntegrationConfig
from src.core.time_integration.euler import ExplicitEuler, ImplicitEuler
from src.core.time_integration.runge_kutta import RungeKutta4, AdaptiveRungeKutta4

@dataclass
class TestCase:
    name: str
    initial_condition: jnp.ndarray
    exact_solution: Optional[callable] = None

class TimeIntegrationTest(unittest.TestCase):
    def setUp(self):
        os.makedirs('test_results/time_integration', exist_ok=True)
        
        self.dt = 0.01
        self.t_final = 1.0
        
        self.test_cases = {
            'exponential_decay': TestCase(
                name="Exponential Decay",
                initial_condition=jnp.array(1.0),
                exact_solution=lambda t: jnp.exp(-t)
            ),
            'exponential_growth': TestCase(
                name="Exponential Growth", 
                initial_condition=jnp.array(1.0),
                exact_solution=lambda t: jnp.exp(2*t)
            ),
            'harmonic_oscillator': TestCase(
                name="Harmonic Oscillator",
                initial_condition=jnp.array([1.0, 0.0]),
                exact_solution=lambda t: jnp.array([
                    jnp.cos(2*jnp.pi*t),
                    -2*jnp.pi*jnp.sin(2*jnp.pi*t)
                ])
            ),
            'damped_oscillator': TestCase(
                name="Damped Oscillator",
                initial_condition=jnp.array([1.0, 0.0]),
                exact_solution=lambda t: jnp.exp(-0.5*t) * jnp.array([
                    jnp.cos(2*jnp.pi*t),
                    -0.5*jnp.cos(2*jnp.pi*t) - 2*jnp.pi*jnp.sin(2*jnp.pi*t)
                ])
            )
        }
        
        self.schemes = self._create_schemes()
        
    def _create_schemes(self) -> Dict:
        config = TimeIntegrationConfig(dt=self.dt)
        return {
            'explicit_euler': ExplicitEuler(config),
            'implicit_euler': ImplicitEuler(config),
            'rk4': RungeKutta4(config),
            'adaptive_rk4': AdaptiveRungeKutta4(
                config, relative_tolerance=1e-6, absolute_tolerance=1e-8
            )
        }
    
    def _compute_derivatives(self, case_name, field, t):
        if case_name == 'exponential_decay':
            return -field
        elif case_name == 'exponential_growth':
            return 2 * field
        elif case_name == 'harmonic_oscillator':
            x, v = field
            return jnp.array([v, -(2*jnp.pi)**2 * x])
        elif case_name == 'damped_oscillator':
            x, v = field
            return jnp.array([v, -0.5*v - (2*jnp.pi)**2 * x])
    
    def _compute_rk4_derivatives(self, case_name, field, t, dt):
        k1 = self._compute_derivatives(case_name, field, t)
        k2 = self._compute_derivatives(case_name, field + 0.5*dt*k1, t + 0.5*dt)
        k3 = self._compute_derivatives(case_name, field + 0.5*dt*k2, t + 0.5*dt) 
        k4 = self._compute_derivatives(case_name, field + dt*k3, t + dt)
        return k1, k2, k3, k4
    
    def test_time_evolution(self):
        for case_name, case in self.test_cases.items():
            times = jnp.arange(0, self.t_final+self.dt, self.dt)
            exact_solutions = [case.exact_solution(t) for t in times]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(times, exact_solutions, 'k--', label='Exact')
            
            for scheme_name, scheme in self.schemes.items():
                numerical_solutions = [case.initial_condition]
                field = case.initial_condition
                t = 0.0
                
                while t < self.t_final:
                    if isinstance(scheme, RungeKutta4):
                        derivatives = self._compute_rk4_derivatives(
                            case_name, field, t, scheme.config.dt
                        )
                    else:
                        derivatives = self._compute_derivatives(case_name, field, t)
                    
                    field = scheme.step(derivatives, t, field)
                    numerical_solutions.append(field)
                    t += scheme.config.dt
                
                if case_name in ['harmonic_oscillator', 'damped_oscillator']:
                    ax.plot(times[:len(numerical_solutions)], 
                            [sol[0] for sol in numerical_solutions], 
                            label=scheme_name)
                else:
                    ax.plot(times[:len(numerical_solutions)],
                            numerical_solutions, 
                            label=scheme_name)
            
            ax.set_xlabel('Time')
            ax.set_ylabel('Solution')
            ax.set_title(f'{case_name} - Time Evolution')
            ax.legend()
            ax.grid(True)
            
            plt.tight_layout()
            plt.savefig(f'test_results/time_integration/{case_name}_comparison.png')
            plt.close()

    def test_all(self):
        print("\nTesting time evolution...")
        self.test_time_evolution()
        print("\nAll tests completed.")

if __name__ == '__main__':
    unittest.main()