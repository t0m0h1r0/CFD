import cupy as cp
from grid1d import Grid
from solver1d import CCDSolver
from equation_system1d import EquationSystem
from equation_sets1d import EquationSet

class CCDTester:
    """CCDメソッドのテストを行うクラス"""

    def __init__(self, grid):
        self.grid = grid
        self.system = None
        self.solver = None
        self.solver_method = "direct"
        self.solver_options = None
        self.analyze_matrix = False
        self.equation_set = None

    def set_solver_options(self, method, options, analyze_matrix=False):
        self.solver_method = method
        self.solver_options = options
        self.analyze_matrix = analyze_matrix

    def set_equation_set(self, equation_set_name):
        self.equation_set = EquationSet.create(equation_set_name)

    def setup_equation_system(self, test_func, use_dirichlet=True, use_neumann=True):
        self.system = EquationSystem(self.grid)

        if self.equation_set is None:
            self.equation_set = EquationSet.create("poisson")

        self.equation_set.setup_equations(
            self.system, 
            self.grid, 
            test_func, 
            use_dirichlet, 
            use_neumann
        )

        if self.solver is None:
            self.solver = CCDSolver(self.system, self.grid)
        else:
            self.solver.system = self.system

        if self.solver_method != "direct" or self.solver_options:
            self.solver.set_solver(method=self.solver_method, options=self.solver_options)

    def run_test_with_options(self, test_func, use_dirichlet=True, use_neumann=True):
        self.setup_equation_system(test_func, use_dirichlet, use_neumann)

        if self.analyze_matrix:
            self.solver.analyze_system()

        psi, psi_prime, psi_second, psi_third = self.solver.solve(analyze_before_solve=False)

        x = self.grid.get_points()
        exact_psi = cp.array([test_func.f(xi) for xi in x])
        exact_psi_prime = cp.array([test_func.df(xi) for xi in x])
        exact_psi_second = cp.array([test_func.d2f(xi) for xi in x])
        exact_psi_third = cp.array([test_func.d3f(xi) for xi in x])

        err_psi = float(cp.max(cp.abs(psi - exact_psi)))
        err_psi_prime = float(cp.max(cp.abs(psi_prime - exact_psi_prime)))
        err_psi_second = float(cp.max(cp.abs(psi_second - exact_psi_second)))
        err_psi_third = float(cp.max(cp.abs(psi_third - exact_psi_third)))

        return {
            "function": test_func.name,
            "numerical": [psi, psi_prime, psi_second, psi_third],
            "exact": [exact_psi, exact_psi_prime, exact_psi_second, exact_psi_third],
            "errors": [err_psi, err_psi_prime, err_psi_second, err_psi_third],
        }

    def run_grid_convergence_test(self, test_func, grid_sizes, x_range, use_dirichlet=True, use_neumann=True):
        results = {}
        original_method = self.solver_method
        original_options = self.solver_options
        original_analyze = self.analyze_matrix
        original_equation_set = self.equation_set

        for n in grid_sizes:
            grid = Grid(n, x_range)
            tester = CCDTester(grid)
            tester.set_solver_options(original_method, original_options, original_analyze)
            tester.equation_set = original_equation_set
            result = tester.run_test_with_options(test_func, use_dirichlet, use_neumann)
            results[n] = result["errors"]

        return results