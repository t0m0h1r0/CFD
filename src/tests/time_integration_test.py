import os
import unittest
from typing import Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.typing import ArrayLike

from src.core.time_integration.base import TimeIntegrationConfig
from src.core.time_integration.euler import ExplicitEuler, ImplicitEuler
from src.core.time_integration.runge_kutta import RungeKutta4, AdaptiveRungeKutta4

class TimeIntegrationTest(unittest.TestCase):
    """時間発展スキームのテストスイート"""
    
    def setUp(self):
        """テストの初期設定"""
        # 出力ディレクトリの作成
        os.makedirs('test_results/time_integration', exist_ok=True)
        
        # 共通の設定
        self.dt = 0.01
        self.t_final = 1.0
        self.config = TimeIntegrationConfig(
            dt=self.dt,
            check_stability=True,
            adaptive_dt=False
        )
        
        # テスト用の時間発展スキーム
        self.schemes = {
            'Explicit Euler': ExplicitEuler(self.config),
            'Implicit Euler': ImplicitEuler(self.config),
            'RK4': RungeKutta4(self.config)
        }
    
    def compute_rk4_derivatives(self,
                              field: ArrayLike,
                              omega: float = 1.0) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        """RK4の4つのステージの時間微分値を計算"""
        dt = self.dt
        
        # 調和振動子の場合の時間微分を計算
        if field.ndim == 1 and len(field) == 2:  # 調和振動子
            x, v = field[0], field[1]
            k1 = jnp.array([v, -omega**2 * x])
            
            x2, v2 = field + 0.5*dt*k1
            k2 = jnp.array([v2, -omega**2 * x2])
            
            x3, v3 = field + 0.5*dt*k2
            k3 = jnp.array([v3, -omega**2 * x3])
            
            x4, v4 = field + dt*k3
            k4 = jnp.array([v4, -omega**2 * x4])
        else:  # 線形減衰
            k1 = -field
            k2 = -(field + 0.5*dt*k1)
            k3 = -(field + 0.5*dt*k2)
            k4 = -(field + dt*k3)
        
        return k1, k2, k3, k4
    
    def test_linear_decay(self):
        """線形減衰のテスト"""
        # 初期条件
        field = jnp.array(1.0)
        t = 0.0
        
        # 各スキームでの数値解
        numerical_solutions = {}
        for name, scheme in self.schemes.items():
            current_field = field
            times = [t]
            values = [float(current_field)]
            
            while t < self.t_final:
                if isinstance(scheme, RungeKutta4):
                    # RK4の場合は4つのステージの時間微分値を計算
                    derivatives = self.compute_rk4_derivatives(current_field)
                    current_field = scheme.step(derivatives, t, current_field)
                else:
                    # オイラー法の場合は単純な時間微分
                    dfield = -current_field
                    current_field = scheme.step(dfield, t, current_field)
                
                t += self.dt
                times.append(t)
                values.append(float(current_field))
            
            numerical_solutions[name] = (times, values)
        
        # 厳密解の計算
        exact_times = jnp.linspace(0, self.t_final, len(times))
        exact_values = jnp.exp(-exact_times)
        
        # 誤差の計算と検証
        for name, (times, values) in numerical_solutions.items():
            error = jnp.max(jnp.abs(jnp.array(values) - jnp.exp(-jnp.array(times))))
            
            # スキームの次数に応じた誤差の評価
            expected_order = self.schemes[name].get_order()
            expected_error = self.dt ** expected_order
            
            self.assertLess(error, expected_error * 10,
                          f"{name}の誤差が予想より大きい: {error:.2e} > {expected_error:.2e}")
        
        # 結果のプロット
        plt.figure(figsize=(10, 6))
        plt.plot(exact_times, exact_values, 'k-', label='Exact')
        
        for name, (times, values) in numerical_solutions.items():
            plt.plot(times, values, '--', label=name)
            
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Linear Decay Test')
        plt.legend()
        plt.grid(True)
        plt.savefig('test_results/time_integration/linear_decay.png')
        plt.close()
    
    def test_harmonic_oscillator(self):
        """調和振動子のテスト"""
        omega = 2 * jnp.pi  # 角周波数
        
        # 初期条件
        field = jnp.array([1.0, 0.0])  # [位置, 速度]
        t = 0.0
        
        # 各スキームでの数値解
        numerical_solutions = {}
        for name, scheme in self.schemes.items():
            current_field = field
            times = [t]
            positions = [float(current_field[0])]
            velocities = [float(current_field[1])]
            
            while t < self.t_final:
                if isinstance(scheme, RungeKutta4):
                    # RK4の場合は4つのステージの時間微分値を計算
                    derivatives = self.compute_rk4_derivatives(current_field, omega)
                    current_field = scheme.step(derivatives, t, current_field)
                else:
                    # オイラー法の場合は単純な時間微分
                    x, v = current_field[0], current_field[1]
                    dfield = jnp.array([v, -omega**2 * x])
                    current_field = scheme.step(dfield, t, current_field)
                
                t += self.dt
                times.append(t)
                positions.append(float(current_field[0]))
                velocities.append(float(current_field[1]))
            
            numerical_solutions[name] = (times, positions, velocities)
        
        # 厳密解の計算
        exact_times = jnp.linspace(0, self.t_final, len(times))
        exact_positions = jnp.cos(omega * exact_times)
        exact_velocities = -omega * jnp.sin(omega * exact_times)
        
        # 誤差の計算と検証
        for name, (times, positions, velocities) in numerical_solutions.items():
            pos_error = jnp.max(jnp.abs(jnp.array(positions) - jnp.cos(omega * jnp.array(times))))
            vel_error = jnp.max(jnp.abs(jnp.array(velocities) - (-omega * jnp.sin(omega * jnp.array(times)))))
            
            # エネルギー保存の検証
            energies = [0.5 * (v**2 + omega**2 * x**2) 
                       for x, v in zip(positions, velocities)]
            energy_error = jnp.max(jnp.abs(jnp.array(energies) - energies[0]))
            
            # スキームの次数に応じた誤差の評価
            expected_order = self.schemes[name].get_order()
            expected_error = self.dt ** expected_order
            
            self.assertLess(pos_error, expected_error * 10,
                          f"{name}の位置の誤差が予想より大きい")
            self.assertLess(vel_error, expected_error * 10,
                          f"{name}の速度の誤差が予想より大きい")
            
            # 結果のプロット
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
            
            # 位置のプロット
            ax1.plot(exact_times, exact_positions, 'k-', label='Exact')
            ax1.plot(times, positions, '--', label=name)
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Position')
            ax1.grid(True)
            ax1.legend()
            
            # 速度のプロット
            ax2.plot(exact_times, exact_velocities, 'k-', label='Exact')
            ax2.plot(times, velocities, '--', label=name)
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Velocity')
            ax2.grid(True)
            ax2.legend()
            
            # エネルギーのプロット
            ax3.plot(times, energies, '-', label='Numerical')
            ax3.axhline(y=energies[0], color='k', linestyle='--', label='Initial')
            ax3.set_xlabel('Time')
            ax3.set_ylabel('Energy')
            ax3.grid(True)
            ax3.legend()
            
            plt.suptitle(f'Harmonic Oscillator Test - {name}')
            plt.tight_layout()
            plt.savefig(f'test_results/time_integration/harmonic_oscillator_{name}.png')
            plt.close()
    
    def test_adaptive_timestep(self):
        """適応的時間ステップ制御のテスト"""
        # 適応的RK4の設定
        adaptive_config = TimeIntegrationConfig(
            dt=0.01,
            check_stability=True,
            adaptive_dt=True
        )
        
        scheme = AdaptiveRungeKutta4(
            config=adaptive_config,
            relative_tolerance=1e-6,
            absolute_tolerance=1e-8
        )
        
        # 初期条件（調和振動子）
        omega = 2 * jnp.pi
        field = jnp.array([1.0, 0.0])
        t = 0.0
        
        # 時間発展
        times = [t]
        positions = [float(field[0])]
        velocities = [float(field[1])]
        dt_values = [float(adaptive_config.dt)]
        
        while t < self.t_final:
            # 4つのステージの時間微分値を計算
            derivatives = self.compute_rk4_derivatives(field, omega)
            
            # 時間発展と誤差推定
            error = scheme.estimate_error(derivatives, field)
            
            # 時間ステップの調整
            new_dt = scheme.adjust_timestep(error, field)
            adaptive_config.dt = new_dt
            
            # 更新された時間ステップで時間発展
            field = scheme.step(derivatives, t, field)
            t += new_dt
            
            times.append(t)
            positions.append(float(field[0]))
            velocities.append(float(field[1]))
            dt_values.append(float(new_dt))
        
        # 時間ステップの変化をプロット
        plt.figure(figsize=(10, 6))
        plt.semilogy(times[:-1], dt_values[:-1], 'b-')
        plt.xlabel('Time')
        plt.ylabel('Time Step Size')
        plt.title('Adaptive Time Step Control')
        plt.grid(True)
        plt.savefig('test_results/time_integration/adaptive_timestep.png')
        plt.close()
        
        # 最小・最大時間ステップの検証
        min_dt = min(dt_values)
        max_dt = max(dt_values)
        self.assertGreater(min_dt, 0.001, "時間ステップが小さすぎます")
        self.assertLess(max_dt, 0.1, "時間ステップが大きすぎます")

if __name__ == '__main__':
    unittest.main()