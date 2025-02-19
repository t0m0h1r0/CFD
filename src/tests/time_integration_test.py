import os
import unittest
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.typing import ArrayLike

from src.core.time_integration.base import TimeIntegrationConfig
from src.core.time_integration.euler import ExplicitEuler, ImplicitEuler
from src.core.time_integration.runge_kutta import RungeKutta4, AdaptiveRungeKutta4

@dataclass
class TestCase:
    """テストケース定義"""
    name: str
    initial_condition: ArrayLike
    exact_solution: Optional[callable] = None
    conserved_quantity: Optional[callable] = None
    stable_dt: Optional[float] = None

class TimeIntegrationTest(unittest.TestCase):
    """時間発展スキームの拡張テストスイート"""
    
    def setUp(self):
        """テストの初期設定"""
        # 出力ディレクトリの作成
        os.makedirs('test_results/time_integration', exist_ok=True)
        
        # 基本設定
        self.dt_base = 0.01
        self.t_final = 1.0
        
        # テストケースの定義
        self.test_cases = {
            'linear_decay': TestCase(
                name="Linear Decay",
                initial_condition=jnp.array(1.0),
                exact_solution=lambda t: jnp.exp(-t),
                conserved_quantity=None,
                stable_dt=2.0
            ),
            'harmonic': TestCase(
                name="Harmonic Oscillator",
                initial_condition=jnp.array([1.0, 0.0]),
                exact_solution=lambda t: jnp.array([
                    jnp.cos(2*jnp.pi*t),
                    -2*jnp.pi*jnp.sin(2*jnp.pi*t)
                ]),
                conserved_quantity=lambda x, v: 0.5*(v**2 + (2*jnp.pi)**2*x**2),
                stable_dt=0.1
            ),
            'van_der_pol': TestCase(
                name="Van der Pol Oscillator",
                initial_condition=jnp.array([2.0, 0.0]),
                exact_solution=None,  # 厳密解なし
                conserved_quantity=None,
                stable_dt=0.01
            ),
            'heat_equation': TestCase(
                name="Heat Equation",
                initial_condition=jnp.sin(jnp.linspace(0, jnp.pi, 100)),
                exact_solution=lambda t, x: jnp.sin(x)*jnp.exp(-t),
                conserved_quantity=lambda u: jnp.sum(u**2),  # L2ノルム
                stable_dt=0.5/100**2  # dx^2/2 の条件
            ),
            'wave_equation': TestCase(
                name="Wave Equation",
                initial_condition=jnp.array([
                    jnp.sin(jnp.linspace(0, 2*jnp.pi, 100)),  # 変位
                    jnp.zeros(100)  # 速度
                ]),
                exact_solution=lambda t, x: jnp.sin(x-t),
                conserved_quantity=lambda u, v: jnp.sum(v**2 + jnp.gradient(u)**2),  # エネルギー
                stable_dt=0.1
            )
        }
        
        # スキームの設定
        self.schemes = self._create_schemes()
        
    def _create_schemes(self) -> Dict:
        """テスト用のスキームを生成"""
        schemes = {}
        
        # 基本設定
        base_config = TimeIntegrationConfig(dt=self.dt_base)
        adaptive_config = TimeIntegrationConfig(dt=self.dt_base, adaptive_dt=True)
        
        # 標準的なスキーム
        schemes['explicit_euler'] = ExplicitEuler(base_config)
        schemes['implicit_euler'] = ImplicitEuler(base_config)
        schemes['rk4'] = RungeKutta4(base_config)
        
        # 適応的スキーム
        schemes['adaptive_rk4'] = AdaptiveRungeKutta4(
            adaptive_config,
            relative_tolerance=1e-6,
            absolute_tolerance=1e-8
        )
        
        return schemes
    
    def _compute_derivatives(self, case_name: str, field: ArrayLike, t: float) -> ArrayLike:
        """各テストケースの時間微分を計算"""
        if case_name == 'linear_decay':
            return -field
        elif case_name == 'harmonic':
            x, v = field
            return jnp.array([v, -(2*jnp.pi)**2 * x])
        elif case_name == 'van_der_pol':
            x, v = field
            mu = 1.0  # Van der Pol パラメータ
            return jnp.array([v, mu*(1 - x**2)*v - x])
        elif case_name == 'heat_equation':
            # 中央差分で2階微分を計算
            dx = jnp.pi/100
            d2u_dx2 = jnp.gradient(jnp.gradient(field, dx), dx)
            return d2u_dx2
        elif case_name == 'wave_equation':
            u, v = field
            # 波動方程式 u_tt = c^2 u_xx
            c = 1.0  # 波速
            dx = 2*jnp.pi/100
            d2u_dx2 = jnp.gradient(jnp.gradient(u, dx), dx)
            return jnp.array([v, c**2 * d2u_dx2])
        else:
            raise ValueError(f"Unknown test case: {case_name}")
    
    def _compute_rk4_derivatives(self,
                               case_name: str,
                               field: ArrayLike,
                               t: float,
                               dt: float) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        """RK4の4つのステージの時間微分値を計算"""
        k1 = self._compute_derivatives(case_name, field, t)
        k2 = self._compute_derivatives(case_name, field + 0.5*dt*k1, t + 0.5*dt)
        k3 = self._compute_derivatives(case_name, field + 0.5*dt*k2, t + 0.5*dt)
        k4 = self._compute_derivatives(case_name, field + dt*k3, t + dt)
        return k1, k2, k3, k4
    
    def test_convergence_rates(self):
        """収束率のテスト"""
        dt_values = jnp.logspace(-4, -1, 4)
        
        for case_name, case in self.test_cases.items():
            if case.exact_solution is None:
                continue
                
            for scheme_name, scheme in self.schemes.items():
                errors = []
                
                for dt in dt_values:
                    # スキームの設定を更新
                    scheme.config.dt = dt
                    
                    # 時間発展
                    field = case.initial_condition
                    t = 0.0
                    while t < self.t_final:
                        if isinstance(scheme, RungeKutta4):
                            k1, k2, k3, k4 = self._compute_rk4_derivatives(
                                case_name, field, t, dt
                            )
                            derivatives = (k1, k2, k3, k4)
                        else:
                            derivatives = self._compute_derivatives(case_name, field, t)
                        
                        field = scheme.step(derivatives, t, field)
                        t += dt
                    
                    # 誤差計算
                    if callable(case.exact_solution):
                        exact = case.exact_solution(self.t_final)
                        error = jnp.linalg.norm(field - exact) / jnp.linalg.norm(exact)
                        errors.append(float(error))
                
                if errors:  # エラーが計算されている場合のみ
                    # 収束率の計算と可視化
                    log_errors = jnp.log(jnp.array(errors))
                    log_dt = jnp.log(dt_values)
                    convergence_rate = -jnp.polyfit(log_dt, log_errors, 1)[0]
                    
                    plt.figure(figsize=(10, 6))
                    plt.loglog(dt_values, errors, 'o-', label=f'Error (rate={convergence_rate:.2f})')
                    plt.xlabel('dt')
                    plt.ylabel('Relative Error')
                    plt.title(f'{case.name} - {scheme_name} Convergence')
                    plt.grid(True)
                    plt.legend()
                    plt.savefig(f'test_results/time_integration/convergence_{case_name}_{scheme_name}.png')
                    plt.close()
                    
                    # 期待される収束率のチェック
                    expected_order = scheme.get_order()
                    self.assertGreater(
                        convergence_rate,
                        expected_order - 0.5,
                        f"{scheme_name} on {case.name}: " 
                        f"Convergence rate {convergence_rate:.2f} lower than expected {expected_order}"
                    )
    
    def test_stability_boundaries(self):
        """安定性境界のテスト"""
        for case_name, case in self.test_cases.items():
            if case.stable_dt is None:
                continue
                
            dt_values = jnp.linspace(0.5*case.stable_dt, 2.0*case.stable_dt, 10)
            
            for scheme_name, scheme in self.schemes.items():
                max_values = []
                
                for dt in dt_values:
                    # スキームの設定を更新
                    scheme.config.dt = dt
                    
                    # 時間発展
                    field = case.initial_condition
                    t = 0.0
                    max_value = float(jnp.max(jnp.abs(field)))
                    
                    try:
                        while t < self.t_final:
                            if isinstance(scheme, RungeKutta4):
                                k1, k2, k3, k4 = self._compute_rk4_derivatives(
                                    case_name, field, t, dt
                                )
                                derivatives = (k1, k2, k3, k4)
                            else:
                                derivatives = self._compute_derivatives(case_name, field, t)
                            
                            field = scheme.step(derivatives, t, field)
                            max_value = max(max_value, float(jnp.max(jnp.abs(field))))
                            t += dt
                            
                            # 発散チェック
                            if jnp.any(jnp.isnan(field)) or jnp.any(jnp.abs(field) > 1e10):
                                max_value = float('inf')
                                break
                                
                    except Exception as e:
                        max_value = float('inf')
                    
                    max_values.append(max_value)
                
                # 結果のプロット
                plt.figure(figsize=(10, 6))
                plt.semilogy(dt_values/case.stable_dt, max_values, 'o-')
                plt.axvline(x=1.0, color='r', linestyle='--', label='Theoretical Boundary')
                plt.xlabel('dt/dt_stable')
                plt.ylabel('Max Absolute Value')
                plt.title(f'{case.name} - {scheme_name} Stability')
                plt.grid(True)
                plt.legend()
                plt.savefig(f'test_results/time_integration/stability_{case_name}_{scheme_name}.png')
                plt.close()
    
    def test_conservation(self):
        """保存量のテスト"""
        for case_name, case in self.test_cases.items():
            if case.conserved_quantity is None:
                continue
                
            for scheme_name, scheme in self.schemes.items():
                # 時間発展
                field = case.initial_condition
                t = 0.0
                times = [t]
                conserved_values = []
                
                # 初期の保存量を計算
                if case_name == 'harmonic':
                    x, v = field
                    conserved_values.append(float(case.conserved_quantity(x, v)))
                else:
                    conserved_values.append(float(case.conserved_quantity(field)))
                
                while t < self.t_final:
                    if isinstance(scheme, RungeKutta4):
                        k1, k2, k3, k4 = self._compute_rk4_derivatives(
                            case_name, field, t, scheme.config.dt
                        )
                        derivatives = (k1, k2, k3, k4)
                    else:
                        derivatives = self._compute_derivatives(case_name, field, t)
                    
                    field = scheme.step(derivatives, t, field)
                    t += scheme.config.dt
                    
                    times.append(t)
                    if case_name == 'harmonic':
                        x, v = field
                        conserved_values.append(float(case.conserved_quantity(x, v)))
                    else:
                        conserved_values.append(float(case.conserved_quantity(field)))
                
                # 保存量の変動を計算
                variation = jnp.std(jnp.array(conserved_values)) / jnp.mean(jnp.array(conserved_values))
                
                # 結果のプロット
                plt.figure(figsize=(10, 6))
                plt.plot(times, conserved_values)
                plt.xlabel('Time')
                plt.ylabel('Conserved Quantity')
                plt.title(f'{case.name} - {scheme_name} Conservation (var={variation:.2e})')
                plt.grid(True)
                plt.savefig(f'test_results/time_integration/conservation_{case_name}_{scheme_name}.png')
                plt.close()
                
                # 保存性の検証
                if isinstance(scheme, RungeKutta4):
                    self.assertLess(
                        variation,
                        1e-3,
                        f"{scheme_name} on {case_name}: Poor conservation, variation={variation:.2e}"
                    )
    
    def test_adaptive_performance(self):
        """適応的時間ステップ制御の性能テスト"""
        adaptive_schemes = {
            name: scheme for name, scheme in self.schemes.items()
            if isinstance(scheme, AdaptiveRungeKutta4)
        }
        
        if not adaptive_schemes:
            return
            
        for case_name, case in self.test_cases.items():
            for scheme_name, scheme in adaptive_schemes.items():
                # 時間発展
                field = case.initial_condition
                t = 0.0
                times = [t]
                dt_values = [float(scheme.config.dt)]
                
                while t < self.t_final:
                    k1, k2, k3, k4 = self._compute_rk4_derivatives(
                        case_name, field, t, scheme.config.dt
                    )
                    derivatives = (k1, k2, k3, k4)
                    
                    # 誤差推定と時間ステップ調整
                    error = scheme.estimate_error(derivatives, field)
                    new_dt = float(scheme.adjust_timestep(error, field))
                    scheme.config.dt = new_dt
                    
                    field = scheme.step(derivatives, t, field)
                    t += new_dt
                    
                    times.append(t)
                    dt_values.append(new_dt)
                
                # 結果のプロット
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
                
                # 時間ステップの変化
                ax1.semilogy(times[:-1], dt_values[:-1])
                ax1.set_xlabel('Time')
                ax1.set_ylabel('dt')
                ax1.grid(True)
                ax1.set_title('Time Step Evolution')
                
                # 時間ステップの分布
                ax2.hist(dt_values[:-1], bins=50)
                ax2.set_xlabel('dt')
                ax2.set_ylabel('Frequency')
                ax2.grid(True)
                ax2.set_title('Time Step Distribution')
                
                plt.suptitle(f'{case.name} - {scheme_name} Adaptive Performance')
                plt.tight_layout()
                plt.savefig(f'test_results/time_integration/adaptive_{case_name}_{scheme_name}.png')
                plt.close()
                
                # 時間ステップの統計の検証
                min_dt = min(dt_values)
                max_dt = max(dt_values)
                mean_dt = sum(dt_values) / len(dt_values)
                
                # 最小時間ステップが小さすぎないことを確認
                self.assertGreater(
                    min_dt,
                    0.001 * self.dt_base,
                    f"{scheme_name} on {case_name}: Time step too small"
                )
                
                # 最大時間ステップが大きすぎないことを確認
                self.assertLess(
                    max_dt,
                    100 * self.dt_base,
                    f"{scheme_name} on {case_name}: Time step too large"
                )
                
                # 平均時間ステップが合理的な範囲にあることを確認
                self.assertGreater(
                    mean_dt,
                    0.1 * self.dt_base,
                    f"{scheme_name} on {case_name}: Mean time step too small"
                )
                self.assertLess(
                    mean_dt,
                    10 * self.dt_base,
                    f"{scheme_name} on {case_name}: Mean time step too large"
                )

    def test_all(self):
        """すべてのテストを実行"""
        # 収束テスト
        print("\nTesting convergence rates...")
        self.test_convergence_rates()
        
        # 安定性テスト
        print("\nTesting stability boundaries...")
        self.test_stability_boundaries()
        
        # 保存量テスト
        print("\nTesting conservation properties...")
        self.test_conservation()
        
        # 適応的時間ステップテスト
        print("\nTesting adaptive timestepping...")
        self.test_adaptive_performance()
        
        print("\nAll tests completed.")

if __name__ == '__main__':
    unittest.main()