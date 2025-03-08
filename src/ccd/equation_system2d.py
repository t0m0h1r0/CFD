import cupy as cp
import cupyx.scipy.sparse as sp
import numpy as np
from scipy import sparse

class EquationSystem2D:
    """Manages the system of equations for 2D CCD method"""
    
    def __init__(self, grid):
        """
        Initialize with a 2D grid
        
        Args:
            grid: Grid2D object
        """
        self.grid = grid
        self.interior_equations = []
        self.boundary_equations = []
        self.corner_equations = []
    
    def add_interior_equation(self, equation):
        """Add an equation for interior points"""
        self.interior_equations.append(equation)
    
    def add_boundary_equation(self, equation):
        """Add an equation for boundary points (not corners)"""
        self.boundary_equations.append(equation)
    
    def add_corner_equation(self, equation):
        """Add an equation for corner points"""
        self.corner_equations.append(equation)
    
    def build_matrix_system(self):
        """Build the sparse matrix system"""
        nx, ny = self.grid.nx_points, self.grid.ny_points
        
        # 7 unknowns per grid point: ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy
        n_unknowns_per_point = 7
        system_size = n_unknowns_per_point * nx * ny
        
        data = []
        row_indices = []
        col_indices = []
        b = cp.zeros(system_size)
        
        # Iterate over all grid points
        for j in range(ny):
            for i in range(nx):
                # Determine point type (interior, boundary, corner)
                is_corner = self.grid.is_corner_point(i, j)
                is_boundary = self.grid.is_boundary_point(i, j) and not is_corner
                
                # Select appropriate equations
                if is_corner:
                    equations = self.corner_equations
                elif is_boundary:
                    equations = self.boundary_equations
                else:
                    equations = self.interior_equations
                
                # Apply equations at this grid point
                for k, eq in enumerate(equations):
                    if eq.is_valid_at(self.grid, i, j):
                        # Current equation row index
                        row = (j * nx + i) * n_unknowns_per_point + k
                        
                        # Get stencil coefficients
                        stencil_coeffs = eq.get_stencil_coefficients(self.grid, i, j)
                        
                        # Add coefficients to the matrix
                        for (di, dj), coeffs in stencil_coeffs.items():
                            ni, nj = i + di, j + dj
                            if 0 <= ni < nx and 0 <= nj < ny:
                                col_base = (nj * nx + ni) * n_unknowns_per_point
                                for m, coeff in enumerate(coeffs):
                                    if coeff != 0.0:
                                        row_indices.append(row)
                                        col_indices.append(col_base + m)
                                        data.append(float(coeff))
                        
                        # Add right-hand side value
                        b[row] = eq.get_rhs(self.grid, i, j)
        
        # Create the sparse matrix
        A = sp.csr_matrix(
            (cp.array(data), (cp.array(row_indices), cp.array(col_indices))),
            shape=(system_size, system_size)
        )
        
        return A, b
    
    def build_kronecker_matrix_system(self):
        """
        Alternative method to build the matrix system using explicit Kronecker products
        This is more suitable for theoretical analysis and understanding
        """
        nx, ny = self.grid.nx_points, self.grid.ny_points
        hx, hy = self.grid.get_spacing()
        
        # First, build the 1D operators for x and y directions
        # (This is a simplified version for demonstration)
        
        # Matrix sizes: 4 unknowns per point in 1D
        x_size = 4 * nx
        y_size = 4 * ny
        
        # Build sample 1D operators (actual implementation would use the original 1D method)
        Lx = sp.eye(x_size)  # Placeholder
        Ly = sp.eye(y_size)  # Placeholder
        
        # Create identity matrices
        Ix = sp.eye(x_size)
        Iy = sp.eye(y_size)
        
        # Build 2D operator using Kronecker products
        # L_2D = L_x ⊗ I_y + I_x ⊗ L_y
        L2D = sparse.kron(Lx.get().tocsr(), Iy.get().tocsr()) + sparse.kron(Ix.get().tocsr(), Ly.get().tocsr())
        
        # Convert back to CuPy sparse matrix
        L2D_cupy = sp.csr_matrix(L2D)
        
        # Build right-hand side (simplified)
        b2D = cp.zeros(L2D_cupy.shape[0])
        
        return L2D_cupy, b2D
    
    def analyze_sparsity(self):
        """Analyze the sparsity of the matrix system"""
        A, _ = self.build_matrix_system()
        
        total_size = A.shape[0]
        nnz = A.nnz
        max_possible_nnz = total_size * total_size
        sparsity = 1.0 - (nnz / max_possible_nnz)
        
        memory_dense_MB = (total_size * total_size * 8) / (1024 * 1024)  # 8 bytes per double
        memory_sparse_MB = (nnz * 12) / (1024 * 1024)  # 8 bytes for value + 4 bytes for indices (approximate)
        
        results = {
            "matrix_size": total_size,
            "non_zeros": nnz,
            "sparsity": sparsity,
            "memory_dense_MB": memory_dense_MB,
            "memory_sparse_MB": memory_sparse_MB
        }
        
        return results