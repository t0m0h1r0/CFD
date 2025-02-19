from typing import Optional, Dict, Any
import jax
import jax.numpy as jnp
from jax import lax

class SORCCDPoissonSolver:
    def __init__(self, config, grid_manager, discretization):
        self.config = config
        self.grid_manager = grid_manager
        self.discretization = discretization
        self.history = {'residual_history': [], 'iterations': 0, 'converged': False}

    def _compute_laplacian(self, u):
        _, u_xx = self.discretization.discretize(u, 'x')
        _, u_yy = self.discretization.discretize(u, 'y')
        laplacian = u_xx + u_yy
        
        dx = self.grid_manager.get_grid_spacing('x')
        dy = self.grid_manager.get_grid_spacing('y')
        # Use the matrix diagonal for stability
        diag_coef = 1.0  # Start with unit coefficient
        
        return laplacian, diag_coef

    def _sor_iteration(self, u, f, diag_coef, omega):
        laplacian, _ = self._compute_laplacian(u)
        residual = f - laplacian
        
        # Limit the update magnitude for stability
        update = jnp.clip(omega / diag_coef * residual, -1e3, 1e3)
        u_new = u + update
        
        residual_norm = jnp.linalg.norm(residual)
        return u_new, residual_norm

    def solve(self, A, b, x0=None):
        if x0 is None:
            x0 = jnp.zeros_like(b)

        _, diag_coef = self._compute_laplacian(x0)
        omega = getattr(self.config, 'omega', 1.5)
        tol = getattr(self.config, 'tolerance', 1e-6)
        max_iter = getattr(self.config, 'max_iterations', 1000)
        
        # Initialize history
        residual_history = []
        
        def cond_fun(state):
            iter_count, _, res_norm = state
            return jnp.logical_and(iter_count < max_iter, res_norm > tol)
        
        def body_fun(state):
            iter_count, u_old, res_norm = state
            u_new, res_norm = self._sor_iteration(u_old, b, diag_coef, omega)
            return iter_count + 1, u_new, res_norm

        init_state = (0, x0, jnp.inf)
        n_iters, x_final, final_res = lax.while_loop(cond_fun, body_fun, init_state)
        
        # Update history
        self.history['iterations'] = int(n_iters)
        self.history['converged'] = bool(final_res <= tol)
        self.history['residual_history'] = []  # Cannot update during JAX loop
        
        return x_final, self.history