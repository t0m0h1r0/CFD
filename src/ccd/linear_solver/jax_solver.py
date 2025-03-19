"""
JAX を使用した線形方程式系ソルバー

This module provides linear system solvers using JAX's numerical capabilities.
It supports direct and iterative methods with acceleration through JIT compilation
and automatic differentiation.
"""

import os
import time
import numpy as np
from .base import LinearSolver
from .cpu_solver import CPULinearSolver


class JAXLinearSolver(LinearSolver):
    """JAX を使用した線形方程式系ソルバー
    
    This solver leverages JAX's numerical capabilities to efficiently solve linear systems.
    It provides JIT-compiled implementations of direct and iterative solvers like CG,
    BiCGSTAB, and GMRES, with performance benefits on accelerator hardware.
    """
    
    def _initialize(self):
        """Initialize JAX solver resources and configuration"""
        try:
            import jax
            import jax.numpy as jnp
            from jax import lax
            
            self.jax = jax
            self.jnp = jnp
            self.lax = lax
            self.has_jax = True
            
            # Define available solvers
            self.solvers = {
                "direct": self._solve_direct,
                "cg": self._solve_cg,
                "bicgstab": self._solve_bicgstab,
                "gmres": self._solve_gmres
            }
            
            # Convert matrix to JAX format and create matvec function
            self.A, self.matvec_fn = self._to_jax_operator(self.original_A)
            
            # Setup scaling if requested
            if self.scaling_method:
                from scaling import plugin_manager
                self.scaler = plugin_manager.get_plugin(self.scaling_method)
                self._prepare_scaling()
            
        except ImportError as e:
            print(f"Warning: JAX not available: {e}")
            self.has_jax = False
            self.cpu_solver = CPULinearSolver(
                self.original_A, 
                self.enable_dirichlet, 
                self.enable_neumann, 
                self.scaling_method
            )
    
    def _to_jax_operator(self, A):
        """Convert matrix to JAX format with JIT-compiled matvec operation
        
        Args:
            A: Input matrix in any supported format
            
        Returns:
            tuple: (jax_matrix, jit_matvec_function)
        """
        try:
            # Already in JAX format
            if hasattr(A, 'shape') and str(type(A)).find('jax') >= 0:
                def matvec(x):
                    return A @ x
                return A, self.jax.jit(matvec)
                
            # Handle sparse matrices
            if hasattr(A, 'tocsr'):
                A = A.tocsr()
                
            if hasattr(A, 'data') and hasattr(A, 'indices') and hasattr(A, 'indptr'):
                from jax.experimental import sparse
                
                # Convert CSR data to JAX arrays
                data = self.jnp.array(A.data)
                indices = self.jnp.array(A.indices)
                indptr = self.jnp.array(A.indptr)
                shape = A.shape
                
                # Define matvec operation using JAX sparse API
                def matvec(x):
                    return sparse.csr_matvec(data, indices, indptr, shape[1], x)
                
                matrix_repr = {
                    'type': 'csr',
                    'data': data,
                    'indices': indices,
                    'indptr': indptr,
                    'shape': shape
                }
                
                # Include dense representation for small matrices
                if A.shape[0] * A.shape[1] < 10000:
                    matrix_repr['dense'] = self.jnp.array(A.toarray())
                
                return matrix_repr, self.jax.jit(matvec)
            else:
                # Dense matrix
                jax_matrix = self.jnp.array(A)
                
                def matvec(x):
                    return jax_matrix @ x
                
                return jax_matrix, self.jax.jit(matvec)
                
        except Exception as e:
            print(f"JAX matrix conversion error: {e}")
            return A, None
    
    def _to_jax_vector(self, b):
        """Convert vector to JAX array"""
        if str(type(b)).find('jax') >= 0:
            return b
        
        # Convert from NumPy/CuPy
        if hasattr(b, 'get'):  # CuPy
            return self.jnp.array(b.get())
        return self.jnp.array(b)
    
    def _prepare_scaling(self):
        """Initialize scaling for the linear system"""
        if not self.scaler or not self.has_jax:
            return
        
        # Create dummy vector for scaling info using NumPy
        matrix_shape = self.A['shape'] if isinstance(self.A, dict) else self.A.shape
        dummy_b = np.ones(matrix_shape[0])
        
        try:
            # Create NumPy version of matrix
            if isinstance(self.A, dict) and 'dense' in self.A:
                A_np = np.array(self.A['dense'])
            else:
                A_np = self._to_numpy_matrix(self.A)
                
            # Calculate scaling info in NumPy
            _, _, scaling_info_np = self.scaler.scale(A_np, dummy_b)
            
            # Convert scaling info to JAX
            self.scaling_info = {}
            for key, value in scaling_info_np.items():
                if isinstance(value, np.ndarray):
                    self.scaling_info[key] = self.jnp.array(value)
                else:
                    self.scaling_info[key] = value
        except Exception as e:
            print(f"Scaling preparation error: {e}")
            self.scaler = None
    
    def _to_numpy_matrix(self, A):
        """Convert JAX matrix to NumPy format"""
        if isinstance(A, dict) and 'dense' in A:
            return np.array(A['dense'])
        elif isinstance(A, dict) and A['type'] == 'csr':
            import scipy.sparse as sp
            return sp.csr_matrix(
                (np.array(A['data']), np.array(A['indices']), np.array(A['indptr'])),
                shape=A['shape']
            )
        return np.array(A)
    
    def _to_numpy_vector(self, v):
        """Convert JAX vector to NumPy array"""
        return np.array(v)
    
    def _to_numpy_scaling_info(self):
        """Convert JAX scaling info to NumPy format"""
        numpy_info = {}
        for key, value in self.scaling_info.items():
            if hasattr(value, 'shape') and str(type(value)).find('jax') >= 0:
                numpy_info[key] = np.array(value)
            else:
                numpy_info[key] = value
        return numpy_info
    
    def solve(self, b, method="direct", options=None):
        """Solve linear system Ax = b using JAX
        
        Args:
            b: Right-hand side vector
            method: Solution method ('direct', 'cg', 'bicgstab', 'gmres')
            options: Solver-specific options
            
        Returns:
            Solution vector x
        """
        if not self.has_jax:
            return self.cpu_solver.solve(b, method, options)
        
        start_time = time.time()
        options = options or {}
        
        try:
            # Convert to JAX
            b_jax = self._to_jax_vector(b)
            
            # Apply scaling if requested
            b_scaled = b_jax
            if self.scaler and self.scaling_info:
                try:
                    # Use NumPy API for scaling b
                    b_np = self._to_numpy_vector(b_jax)
                    b_np_scaled = self.scaler.scale_b_only(b_np, self._to_numpy_scaling_info())
                    b_scaled = self._to_jax_vector(b_np_scaled)
                except Exception as e:
                    print(f"Scaling error: {e}")
            
            # Choose solver method
            if method not in self.solvers:
                print(f"Unsupported solver in JAX: {method}, switching to direct")
                method = "direct"
            
            # Solve system
            solver_func = self.solvers[method]
            x_jax, iterations = solver_func(self.A, b_scaled, options)
            
            # Apply unscaling if needed
            if self.scaler and self.scaling_info:
                try:
                    # Use NumPy API for unscaling x
                    x_np = self._to_numpy_vector(x_jax)
                    x_np_unscaled = self.scaler.unscale(x_np, self._to_numpy_scaling_info())
                    x_jax = self._to_jax_vector(x_np_unscaled)
                except Exception as e:
                    print(f"Unscaling error: {e}")
            
            # Convert to NumPy for return
            x = self._to_numpy_vector(x_jax)
            
            # Record solver statistics
            self.last_iterations = iterations
            elapsed = time.time() - start_time
            print(f"JAX solver: {method}, time: {elapsed:.4f}s" + 
                  (f", iterations: {iterations}" if iterations else ""))
            
            return x
            
        except Exception as e:
            print(f"JAX solver error: {e}, falling back to CPU")
            return CPULinearSolver(
                self.original_A, 
                self.enable_dirichlet, 
                self.enable_neumann, 
                self.scaling_method
            ).solve(b, method, options)
    
    def _solve_direct(self, A, b, options=None):
        """Direct solver using JAX linear algebra"""
        try:
            # Use dense matrix if available
            if isinstance(A, dict) and 'dense' in A:
                A_dense = A['dense']
            else:
                A_dense = A
                
            x = self.jnp.linalg.solve(A_dense, b)
            return x, None
        except Exception as e:
            print(f"JAX direct solver error: {e}, falling back to CPU")
            import scipy.sparse.linalg as splinalg
            x = splinalg.spsolve(self._to_numpy_matrix(A), self._to_numpy_vector(b))
            return self._to_jax_vector(x), None
    
    def _solve_cg(self, A, b, options=None):
        """Conjugate Gradient solver using JAX"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        atol = options.get("atol", 0.0)
        maxiter = options.get("maxiter", 1000)
        
        # Initial guess
        x0 = options.get("x0", self.jnp.zeros_like(b))
        
        # Initial residual and direction
        r = b - self.matvec_fn(x0)
        p = r
        
        # Initial residual norm squared
        rsold = self.jnp.dot(r, r)
        
        # Right-hand side norm squared for convergence check
        b_norm2 = self.jnp.dot(b, b)
        tol2 = self.jnp.maximum(tol**2 * b_norm2, atol**2)
        
        # Set up monitoring variables
        residuals = []
        monitor = options.get("monitor_convergence", False)
        
        # Define CG step function
        def cg_step(state):
            x, r, p, rs_old, k = state
            
            # Standard CG iteration
            Ap = self.matvec_fn(p)
            alpha = rs_old / self.jnp.dot(p, Ap)
            x_new = x + alpha * p
            r_new = r - alpha * Ap
            rs_new = self.jnp.dot(r_new, r_new)
            beta = rs_new / rs_old
            p_new = r_new + beta * p
            
            # Monitor convergence if requested
            if monitor:
                rel_res = self.jnp.sqrt(rs_new / b_norm2)
                self.jax.experimental.host_callback.id_tap(
                    lambda r, _: residuals.append(float(r)), rel_res, result=None
                )
            
            return x_new, r_new, p_new, rs_new, k + 1
        
        # Define stopping condition
        def cg_condition(state):
            _, _, _, rs, k = state
            return (rs > tol2) & (k < maxiter)
        
        # JIT compile the CG loop
        cg_loop = self.jax.jit(
            lambda state: self.lax.while_loop(cg_condition, cg_step, state)
        )
        
        # Run the algorithm
        x_final, _, _, _, iterations = cg_loop((x0, r, p, rsold, 0))
        
        # Visualize convergence history if requested
        if monitor and residuals:
            self._visualize_convergence(residuals, "cg", options)
        
        return x_final, int(iterations)
    
    def _solve_bicgstab(self, A, b, options=None):
        """BiCGSTAB solver using JAX"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        atol = options.get("atol", 0.0)
        maxiter = options.get("maxiter", 1000)
        
        # Initial guess
        x0 = options.get("x0", self.jnp.zeros_like(b))
        
        # Initial residual and shadow residual
        r0 = b - self.matvec_fn(x0)
        r_hat = r0
        
        # Initialize parameters
        rho_prev = alpha = omega = 1.0
        p = v = self.jnp.zeros_like(b)
        
        # Squared norms for convergence check
        b_norm2 = self.jnp.dot(b, b)
        tol2 = self.jnp.maximum(tol**2 * b_norm2, atol**2)
        
        # Set up monitoring variables
        residuals = []
        monitor = options.get("monitor_convergence", False)
        
        # Define BiCGSTAB step
        def bicgstab_step(state):
            x, r, p, v, r_hat, rho_prev, omega, k = state
            
            # BiCGSTAB iteration
            rho = self.jnp.dot(r_hat, r)
            beta = (rho / self.jnp.maximum(rho_prev, 1e-15)) * (alpha / self.jnp.maximum(omega, 1e-15))
            p_new = r + beta * (p - omega * v)
            
            v_new = self.matvec_fn(p_new)
            alpha = rho / self.jnp.maximum(self.jnp.dot(r_hat, v_new), 1e-15)
            
            s = r - alpha * v_new
            t = self.matvec_fn(s)
            
            omega_new = self.jnp.dot(t, s) / self.jnp.maximum(self.jnp.dot(t, t), 1e-15)
            x_new = x + alpha * p_new + omega_new * s
            r_new = s - omega_new * t
            
            # Monitor convergence if requested
            if monitor:
                rel_res = self.jnp.linalg.norm(r_new) / self.jnp.sqrt(b_norm2)
                self.jax.experimental.host_callback.id_tap(
                    lambda r, _: residuals.append(float(r)), rel_res, result=None
                )
            
            # Detect stagnation or divergence
            valid_step = (omega_new != 0.0) & (alpha != 0.0) & (rho != 0.0)
            k_new = self.jnp.where(valid_step, k + 1, maxiter)
            
            return x_new, r_new, p_new, v_new, r_hat, rho, omega_new, k_new
        
        # Define stopping condition
        def bicgstab_condition(state):
            _, r, _, _, _, _, _, k = state
            return (self.jnp.dot(r, r) > tol2) & (k < maxiter)
        
        # JIT compile and run the BiCGSTAB loop
        try:
            bicgstab_loop = self.jax.jit(
                lambda state: self.lax.while_loop(bicgstab_condition, bicgstab_step, state)
            )
            
            # Run the algorithm
            init_state = (x0, r0, p, v, r_hat, rho_prev, omega, 0)
            x_final, _, _, _, _, _, _, iterations = bicgstab_loop(init_state)
            
            # Visualize convergence history if requested
            if monitor and residuals:
                self._visualize_convergence(residuals, "bicgstab", options)
            
            return x_final, int(iterations)
        except Exception as e:
            print(f"JAX BiCGSTAB error: {e}, falling back to CPU implementation")
            import scipy.sparse.linalg as splinalg
            x = splinalg.bicgstab(
                self._to_numpy_matrix(A), 
                self._to_numpy_vector(b),
                tol=tol,
                maxiter=maxiter
            )[0]
            return self._to_jax_vector(x), None
    
    def _solve_gmres(self, A, b, options=None):
        """GMRES solver using JAX"""
        options = options or {}
        tol = options.get("tol", 1e-10)
        atol = options.get("atol", 0.0)
        maxiter = options.get("maxiter", 1000)
        restart = options.get("restart", 20)
        
        # Initial guess
        x0 = options.get("x0", self.jnp.zeros_like(b))
        
        # Compute initial residual
        r0 = b - self.matvec_fn(x0)
        b_norm = self.jnp.linalg.norm(b)
        r_norm = self.jnp.linalg.norm(r0)
        
        # Set target tolerance
        target_norm = self.jnp.maximum(tol * b_norm, atol)
        
        # Set up monitoring variables
        residuals = []
        monitor = options.get("monitor_convergence", False)
        
        # Run single GMRES iteration with restart
        def gmres_single_restart(x0, r0, r_norm):
            """Run one restart of GMRES"""
            # Normalize the first Krylov vector
            v1 = r0 / r_norm
            
            # Initialize Arnoldi storage with padded size
            krylov_size = min(restart, b.size)
            V = self.jnp.zeros((b.size, krylov_size + 1))
            V = V.at[:, 0].set(v1)
            
            # Hessenberg matrix for Arnoldi
            H = self.jnp.zeros((krylov_size + 1, krylov_size))
            
            # Givens rotation storage
            s = self.jnp.zeros(krylov_size)
            c = self.jnp.zeros(krylov_size)
            
            # Residual vector
            beta = self.jnp.zeros(krylov_size + 1)
            beta = beta.at[0].set(r_norm)
            
            # Arnoldi iteration in each step
            def arnoldi_step(j, state):
                V, H, s, c, beta = state
                
                # Apply matrix-vector product
                w = self.matvec_fn(V[:, j])
                
                # Modified Gram-Schmidt orthogonalization
                for i in range(j + 1):
                    H = H.at[i, j].set(self.jnp.dot(V[:, i], w))
                    w = w - H[i, j] * V[:, i]
                
                # Set H[j+1,j] and V[:,j+1]
                H_j1_j = self.jnp.linalg.norm(w)
                V = V.at[:, j + 1].set(w / self.jnp.maximum(H_j1_j, 1e-14))
                H = H.at[j + 1, j].set(H_j1_j)
                
                # Apply previous Givens rotations to new column of H
                for i in range(j):
                    # Apply rotation to column j
                    temp = c[i] * H[i, j] + s[i] * H[i + 1, j]
                    H = H.at[i + 1, j].set(-s[i] * H[i, j] + c[i] * H[i + 1, j])
                    H = H.at[i, j].set(temp)
                
                # Compute new Givens rotation for column j
                if H[j + 1, j] != 0.0:
                    # Compute rotation parameters
                    t = self.jnp.sqrt(H[j, j]**2 + H[j + 1, j]**2)
                    c = c.at[j].set(H[j, j] / t)
                    s = s.at[j].set(H[j + 1, j] / t)
                    
                    # Apply rotation to H
                    H = H.at[j, j].set(t)
                    H = H.at[j + 1, j].set(0.0)
                    
                    # Apply rotation to beta
                    temp = c[j] * beta[j]
                    beta = beta.at[j + 1].set(-s[j] * beta[j])
                    beta = beta.at[j].set(temp)
                
                # Monitor convergence if requested
                if monitor and j % 4 == 0:
                    res_approx = abs(beta[j + 1])
                    rel_res = res_approx / b_norm
                    self.jax.experimental.host_callback.id_tap(
                        lambda r, _: residuals.append(float(r)), rel_res, result=None
                    )
                
                return V, H, s, c, beta
            
            # Run Arnoldi process for krylov_size steps
            V, H, s, c, beta = self.jax.lax.fori_loop(
                0, krylov_size, arnoldi_step, (V, H, s, c, beta)
            )
            
            # Solve upper triangular system H y = beta to get Krylov coefficients
            y = self.jnp.zeros(krylov_size)
            for i in range(krylov_size - 1, -1, -1):
                y_i = beta[i]
                for j in range(i + 1, krylov_size):
                    y_i = y_i - H[i, j] * y[j]
                y = y.at[i].set(y_i / H[i, i])
            
            # Compute new solution and residual
            dx = V[:, :krylov_size] @ y
            x_new = x0 + dx
            r_new = b - self.matvec_fn(x_new)
            r_norm_new = self.jnp.linalg.norm(r_new)
            
            return x_new, r_new, r_norm_new
        
        # GMRES outer loop with restarts
        def gmres_with_restarts():
            """GMRES with restarts"""
            # Initialize
            x = x0
            r = r0
            r_current_norm = r_norm
            iteration = 0
            
            # Continue until convergence or max iterations
            while r_current_norm > target_norm and iteration < maxiter:
                x, r, r_current_norm = gmres_single_restart(x, r, r_current_norm)
                iteration += 1
                
                # Monitor between restarts
                if monitor:
                    rel_res = r_current_norm / b_norm
                    residuals.append(float(rel_res))
                    if iteration % 1 == 0:
                        print(f"  Restart {iteration}: residual = {rel_res:.6e}")
            
            return x, iteration
        
        # Execute GMRES
        x_final, iterations = gmres_with_restarts()
        
        # Visualize convergence history if requested
        if monitor and residuals:
            self._visualize_convergence(residuals, "gmres", options)
        
        return x_final, iterations
    
    def _visualize_convergence(self, residuals, method_name, options):
        """Visualize convergence history"""
        output_dir = options.get("output_dir", "results")
        os.makedirs(output_dir, exist_ok=True)
        
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.semilogy(range(1, len(residuals) + 1), residuals, 'b-')
        plt.grid(True, which="both", ls="--")
        plt.xlabel('Iteration')
        plt.ylabel('Residual (log scale)')
        plt.title(f'{method_name.upper()} Solver Convergence (JAX)')
        
        prefix = options.get("prefix", "")
        filename = os.path.join(output_dir, f"{prefix}_convergence_jax_{method_name}.png")
        plt.savefig(filename, dpi=150)
        plt.close()