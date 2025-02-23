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
        os.makedirs("test_results/time_integration", exist_ok=True)

        self.dt = 0.001
        self.t_final = 1.0

        # ここで10種類のテストケースを定義
        self.test_cases = {
            # (1) Exponential Decay
            "exponential_decay": TestCase(
                name="Exponential Decay",
                initial_condition=jnp.array(1.0),
                exact_solution=lambda t: jnp.exp(-t),
                derivative=lambda x, t: -x,
            ),
            # (2) Exponential Growth
            "exponential_growth": TestCase(
                name="Exponential Growth",
                initial_condition=jnp.array(1.0),
                exact_solution=lambda t: jnp.exp(2 * t),
                derivative=lambda x, t: 2 * x,
            ),
            # (3) Harmonic Oscillator
            "harmonic_oscillator": TestCase(
                name="Harmonic Oscillator",
                initial_condition=jnp.array([1.0, 0.0]),
                exact_solution=lambda t: jnp.array(
                    [jnp.cos(2 * jnp.pi * t), -2 * jnp.pi * jnp.sin(2 * jnp.pi * t)]
                ),
                derivative=lambda field, t: jnp.array(
                    [field[1], -((2 * jnp.pi) ** 2) * field[0]]
                ),
            ),
            # (4) Damped Oscillator
            "damped_oscillator": TestCase(
                name="Damped Oscillator",
                initial_condition=jnp.array([1.0, 0.0]),
                exact_solution=lambda t: jnp.exp(-0.5 * t)
                * jnp.array(
                    [
                        jnp.cos(2 * jnp.pi * t),
                        -0.5 * jnp.cos(2 * jnp.pi * t)
                        - 2 * jnp.pi * jnp.sin(2 * jnp.pi * t),
                    ]
                ),
                derivative=lambda field, t: jnp.array(
                    [field[1], -0.5 * field[1] - (2 * jnp.pi) ** 2 * field[0]]
                ),
            ),
            # (5) Logistic Growth
            "logistic_growth": TestCase(
                name="Logistic Growth",
                initial_condition=jnp.array(0.1),
                exact_solution=lambda t: 1.0 / (1.0 + 9.0 * jnp.exp(-t)),
                derivative=lambda x, t: x * (1.0 - x),
            ),
            # (6) Quadratic Motion
            "quadratic_motion": TestCase(
                name="Quadratic Motion",
                initial_condition=jnp.array(0.0),
                exact_solution=lambda t: 0.5 * t**2,
                derivative=lambda x, t: t,
            ),
            # (7) Sinusoidal Decay
            "sinusoidal_decay": TestCase(
                name="Sinusoidal Decay",
                initial_condition=jnp.array([1.0, 0.0]),
                exact_solution=None,
                derivative=lambda field, t: jnp.array(
                    [field[1], -0.2 * field[1] - (2 * jnp.pi) ** 2 * field[0]]
                ),
            ),
            # (8) Cubic Dynamics
            "cubic_dynamics": TestCase(
                name="Cubic Dynamics",
                initial_condition=jnp.array(0.1),
                exact_solution=lambda t: 1.0 / jnp.sqrt(2.0 * (50.0 - t)),
                derivative=lambda x, t: x**3,
            ),
            # (9) Polynomial (3次)
            "polynomial_3": TestCase(
                name="Polynomial 3rd",
                initial_condition=jnp.array(0.0),
                exact_solution=lambda t: t**3 + t**2 + t,
                derivative=lambda x, t: 3 * t**2 + 2 * t + 1,
            ),
            # (10) Stiff Equation
            "stiff_equation": TestCase(
                name="Stiff Equation",
                initial_condition=jnp.array(1.0),
                exact_solution=lambda t: jnp.exp(-1000.0 * t),
                derivative=lambda x, t: -1000.0 * x,
            ),
        }

        self.schemes = self._create_schemes()

    def _create_schemes(self) -> Dict:
        config = TimeIntegrationConfig(dt=self.dt)
        return {
            "explicit_euler": ExplicitEuler(config),
            "implicit_euler": ImplicitEuler(config),
            "rk4": RungeKutta4(config),
            "adaptive_rk4": AdaptiveRungeKutta4(
                config, relative_tolerance=1e-6, absolute_tolerance=1e-8
            ),
        }

    def _compute_error(self, numerical: jnp.ndarray, exact: jnp.ndarray) -> jnp.ndarray:
        """数値解と厳密解の相対誤差を計算"""
        if numerical.ndim > 1:
            # 多次元の場合は第1成分のみで誤差を計算
            numerical = numerical[:, 0]
            exact = exact[:, 0]
        return jnp.abs(numerical - exact) / (jnp.abs(exact) + 1e-10)

    def test_time_evolution(self):
        # カラーマップの定義
        colors = {
            "explicit_euler": "blue",
            "implicit_euler": "green",
            "rk4": "red",
            "adaptive_rk4": "purple",
        }

        for case_name, case in self.test_cases.items():
            times = jnp.arange(0, self.t_final + self.dt, self.dt)

            # メインの図とツイン軸の作成
            fig, ax1 = plt.subplots(figsize=(12, 7))
            ax2 = ax1.twinx()

            # 厳密解が定義されている場合
            if case.exact_solution is not None:
                exact_array = jnp.stack([case.exact_solution(t) for t in times])

                # 厳密解のプロット（左軸）
                if exact_array.ndim == 1:
                    ax1.plot(times, exact_array, "k--", label="Exact", linewidth=2)
                else:
                    ax1.plot(
                        times, exact_array[:, 0], "k--", label="Exact", linewidth=2
                    )

            # 数値解と誤差のプロット
            solution_lines = []
            error_lines = []

            for scheme_name, scheme in self.schemes.items():
                numerical_solutions = [case.initial_condition]
                field = case.initial_condition
                t = 0.0

                while t < self.t_final:
                    if isinstance(scheme, RungeKutta4):
                        derivatives = scheme.compute_stage_derivatives(
                            case.derivative, field, t
                        )
                    else:
                        derivatives = case.derivative(field, t)

                    field = scheme.step(derivatives, t, field)
                    numerical_solutions.append(field)
                    t += scheme.config.dt

                numerical_array = jnp.stack(numerical_solutions)

                # 数値解のプロット（左軸）
                if numerical_array.ndim == 1:
                    line1 = ax1.plot(
                        times[: len(numerical_array)],
                        numerical_array,
                        label=f"{scheme_name} (solution)",
                        color=colors[scheme_name],
                    )[0]
                else:
                    line1 = ax1.plot(
                        times[: len(numerical_array)],
                        numerical_array[:, 0],
                        label=f"{scheme_name} (solution)",
                        color=colors[scheme_name],
                    )[0]

                solution_lines.append(line1)

                # 誤差のプロット（右軸）
                if case.exact_solution is not None:
                    error = self._compute_error(
                        numerical_array, exact_array[: len(numerical_array)]
                    )
                    line2 = ax2.plot(
                        times[: len(error)],
                        error,
                        "--",
                        label=f"{scheme_name} (error)",
                        color=colors[scheme_name],
                        alpha=0.5,
                    )[0]
                    error_lines.append(line2)

            # 軸の設定
            ax1.set_xlabel("Time")
            ax1.set_ylabel("Solution")
            ax2.set_ylabel("Relative Error (log scale)")

            # 右軸を対数スケールに
            ax2.set_yscale("log")

            # タイトルの設定
            plt.title(f"{case.name} - Time Evolution and Error Analysis")

            # グリッドの設定
            ax1.grid(True, which="major", linestyle="-", alpha=0.2)
            ax1.grid(True, which="minor", linestyle=":", alpha=0.1)

            # 凡例の設定
            lines = solution_lines + error_lines
            labels = [line.get_label() for line in lines]
            ax1.legend(lines, labels, loc="center left", bbox_to_anchor=(1.15, 0.5))

            # レイアウトの調整
            plt.tight_layout()

            # 保存
            plt.savefig(
                f"test_results/time_integration/{case_name}_comparison.png",
                bbox_inches="tight",
                dpi=300,
            )
            plt.close()

    def test_all(self):
        print("Testing time evolution...")
        self.test_time_evolution()
        print("All tests completed.")


if __name__ == "__main__":
    unittest.main()
