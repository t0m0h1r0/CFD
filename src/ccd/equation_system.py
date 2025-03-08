import cupy as cp
import cupyx.scipy.sparse as sp

class EquationSystem:
    """方程式システムを管理するクラス"""

    def __init__(self, grid):
        self.grid = grid
        self.left_boundary_equations = []
        self.interior_equations = []
        self.right_boundary_equations = []

    def add_left_boundary_equation(self, equation):
        self.left_boundary_equations.append(equation)

    def add_interior_equation(self, equation):
        self.interior_equations.append(equation)

    def add_right_boundary_equation(self, equation):
        self.right_boundary_equations.append(equation)

    def add_equation(self, equation):
        self.left_boundary_equations.append(equation)
        self.interior_equations.append(equation)
        self.right_boundary_equations.append(equation)

    def build_matrix_system(self):
        """スパース行列システムを構築"""
        n = self.grid.n_points
        size = 4 * n
        
        data = []
        row_indices = []
        col_indices = []
        b = cp.zeros(size)

        for i in range(n):
            if i == 0:
                equations = self.left_boundary_equations
            elif i == n - 1:
                equations = self.right_boundary_equations
            else:
                equations = self.interior_equations

            if len(equations) != 4:
                raise ValueError(f"点 {i} に対する方程式が4つではありません")

            for k, eq in enumerate(equations):
                stencil_coeffs = eq.get_stencil_coefficients(self.grid, i)
                rhs_value = eq.get_rhs(self.grid, i)

                for offset, coeffs in stencil_coeffs.items():
                    j = i + offset
                    if 0 <= j < n:
                        for m, coeff in enumerate(coeffs):
                            if coeff != 0.0:
                                row_indices.append(i * 4 + k)
                                col_indices.append(j * 4 + m)
                                data.append(coeff)

                b[i * 4 + k] = rhs_value

        A = sp.csr_matrix(
            (cp.array(data), (cp.array(row_indices), cp.array(col_indices))), 
            shape=(size, size)
        )

        return A, b