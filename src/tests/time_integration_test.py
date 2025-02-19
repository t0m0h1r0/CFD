import os
import unittest
from typing import Dict, Optional, Callable
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
    exact_solution: Optional[Callable[[float], jnp.ndarray]]
    derivative: Callable[[jnp.ndarray, float], jnp.ndarray]

class TimeIntegrationTest(unittest.TestCase):
    def setUp(self):
        os.makedirs('test_results/time_integration', exist_ok=True)
        
        self.dt = 0.001
        self.t_final = 1.0
        
        # ここで10種類のテストケースを定義
        self.test_cases = {
            # (1) Exponential Decay
            'exponential_decay': TestCase(
                name="Exponential Decay",
                initial_condition=jnp.array(1.0),
                exact_solution=lambda t: jnp.exp(-t),
                derivative=lambda x, t: -x
            ),
            # (2) Exponential Growth
            'exponential_growth': TestCase(
                name="Exponential Growth",
                initial_condition=jnp.array(1.0),
                exact_solution=lambda t: jnp.exp(2 * t),
                derivative=lambda x, t: 2 * x
            ),
            # (3) Harmonic Oscillator
            'harmonic_oscillator': TestCase(
                name="Harmonic Oscillator",
                initial_condition=jnp.array([1.0, 0.0]),
                exact_solution=lambda t: jnp.array([
                    jnp.cos(2*jnp.pi*t),
                    -2*jnp.pi*jnp.sin(2*jnp.pi*t)
                ]),
                derivative=lambda field, t: jnp.array([
                    field[1],
                    -(2*jnp.pi)**2 * field[0]
                ])
            ),
            # (4) Damped Oscillator
            'damped_oscillator': TestCase(
                name="Damped Oscillator",
                initial_condition=jnp.array([1.0, 0.0]),
                exact_solution=lambda t: jnp.exp(-0.5*t) * jnp.array([
                    jnp.cos(2*jnp.pi*t),
                    -0.5*jnp.cos(2*jnp.pi*t) - 2*jnp.pi*jnp.sin(2*jnp.pi*t)
                ]),
                derivative=lambda field, t: jnp.array([
                    field[1],
                    -0.5*field[1] - (2*jnp.pi)**2 * field[0]
                ])
            ),
            # (5) Logistic Growth (r=1, K=1, 初期値0.1)
            'logistic_growth': TestCase(
                name="Logistic Growth",
                initial_condition=jnp.array(0.1),
                exact_solution=lambda t: 1.0 / (1.0 + 9.0 * jnp.exp(-t)),  # x(0)=0.1 => 1/(1+9e^-t)
                derivative=lambda x, t: x * (1.0 - x)
            ),
            # (6) Quadratic Motion: dx/dt = t,  x(t) = 0.5 * t^2
            'quadratic_motion': TestCase(
                name="Quadratic Motion",
                initial_condition=jnp.array(0.0),
                exact_solution=lambda t: 0.5 * t**2,
                derivative=lambda x, t: t
            ),
            # (7) Sinusoidal Decay (2次元系, exact_solutionはNoneに)
            'sinusoidal_decay': TestCase(
                name="Sinusoidal Decay",
                initial_condition=jnp.array([1.0, 0.0]),
                exact_solution=None,  # 厳密解が複雑なので None
                derivative=lambda field, t: jnp.array([
                    field[1],
                    -0.2*field[1] - (2*jnp.pi)**2 * field[0]
                ])
            ),
            # (8) Cubic Dynamics: dx/dt = x^3
            #   x(t) = 1 / sqrt(2(50 - t)) (x(0)=0.1 から導出)
            'cubic_dynamics': TestCase(
                name="Cubic Dynamics",
                initial_condition=jnp.array(0.1),
                exact_solution=lambda t: 1.0 / jnp.sqrt(2.0 * (50.0 - t)),
                derivative=lambda x, t: x**3
            ),
            # (9) Polynomial (3次) : dx/dt = 3t^2 + 2t + 1, x(0)=0 => x(t)=t^3 + t^2 + t
            'polynomial_3': TestCase(
                name="Polynomial 3rd",
                initial_condition=jnp.array(0.0),
                exact_solution=lambda t: t**3 + t**2 + t,
                derivative=lambda x, t: 3*t**2 + 2*t + 1
            ),
            # (10) Stiff Equation: dx/dt = -1000 x, x(t)= e^-1000t
            'stiff_equation': TestCase(
                name="Stiff Equation",
                initial_condition=jnp.array(1.0),
                exact_solution=lambda t: jnp.exp(-1000.0 * t),
                derivative=lambda x, t: -1000.0 * x
            ),
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
    
    def _compute_rk4_derivatives(self, derivative_fn, field, t, dt):
        k1 = derivative_fn(field, t)
        k2 = derivative_fn(field + 0.5*dt*k1, t + 0.5*dt)
        k3 = derivative_fn(field + 0.5*dt*k2, t + 0.5*dt)
        k4 = derivative_fn(field + dt*k3, t + dt)
        return k1, k2, k3, k4
    
    def test_time_evolution(self):
        for case_name, case in self.test_cases.items():
            times = jnp.arange(0, self.t_final + self.dt, self.dt)
            
            # 厳密解が定義されている場合はプロット用配列を作成
            if case.exact_solution is not None:
                exact_array = jnp.stack([case.exact_solution(t) for t in times])
            else:
                exact_array = None
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # --- 厳密解の描画 ---
            if exact_array is not None:
                # 次元数に応じてプロットを変える (1次元か2次元か)
                if exact_array.ndim == 1:
                    ax.plot(times, exact_array, 'k--', label='Exact')
                else:
                    # 2次元以上の場合、1成分目だけ可視化(必要に応じて変更)
                    ax.plot(times, exact_array[:, 0], 'k--', label='Exact')
            
            # --- 数値解の描画 ---
            for scheme_name, scheme in self.schemes.items():
                numerical_solutions = [case.initial_condition]
                field = case.initial_condition
                t = 0.0
                
                while t < self.t_final:
                    if isinstance(scheme, RungeKutta4):
                        derivatives = self._compute_rk4_derivatives(
                            case.derivative, field, t, scheme.config.dt
                        )
                    else:
                        derivatives = case.derivative(field, t)
                    
                    field = scheme.step(derivatives, t, field)
                    numerical_solutions.append(field)
                    t += scheme.config.dt
                
                numerical_array = jnp.stack(numerical_solutions)
                
                # 次元数に応じてプロットを変える
                if numerical_array.ndim == 1:
                    ax.plot(times[:len(numerical_array)],
                            numerical_array,
                            label=scheme_name)
                else:
                    ax.plot(times[:len(numerical_array)],
                            numerical_array[:, 0],
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
        print("Testing time evolution...")
        self.test_time_evolution()
        print("All tests completed.")

if __name__ == '__main__':
    unittest.main()
