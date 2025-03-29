"""
高精度コンパクト差分法 (CCD) の3次元テスターモジュール

このモジュールは、3次元のCCDソルバーのテスト機能を提供します。
"""

import numpy as np
from core.base.base_tester import BaseTester
from core.solver.solver3d import CCDSolver3D
from equation_set.equation_sets import DerivativeEquationSet3D


class CCDTester3D(BaseTester):
    """3次元CCDテスタークラス"""
    
    def __init__(self, grid):
        """
        3Dテスターを初期化
        
        Args:
            grid: 3Dグリッドオブジェクト
        """
        if not hasattr(grid, 'is_3d') or not grid.is_3d:
            raise ValueError("3Dテスターは非3Dグリッドでは使用できません")
        super().__init__(grid)
    
    def _create_solver(self):
        """3D用ソルバー作成"""
        self.solver = CCDSolver3D(self.equation_set, self.grid, backend=self.backend)
        
        # ソルバーオプションを適用
        if self.solver_method != "direct" or self.solver_options or self.scaling_method:
            self.solver.set_solver(
                method=self.solver_method, 
                options=self.solver_options, 
                scaling_method=self.scaling_method
            )
        
        # システム行列への参照を保持
        self.matrix_A = self.solver.matrix_A
            
    def get_dimension(self):
        """
        次元を返す
        
        Returns:
            int: 3 (3次元)
        """
        return 3
    
    def get_test_function(self, func_name):
        """
        3Dテスト関数取得
        
        Args:
            func_name: テスト関数名
            
        Returns:
            TestFunction3D: 取得したテスト関数
        """
        from test_function.test_function3d import TestFunction3DFactory
        
        # 3D関数を検索
        func_3d = TestFunction3DFactory.get_function_by_name(func_name)
        if func_3d:
            return func_3d
                
        # 2D関数から変換を試みる
        from test_function.test_function2d import TestFunction2DFactory
        func_2d = TestFunction2DFactory.get_function_by_name(func_name)
        if func_2d:
            from test_function.test_function3d import TestFunction3D
            return TestFunction3D.from_2d(func_2d, method='extrude')
            
        # 1D関数から変換を試みる
        from test_function.test_function1d import TestFunction1DFactory
        func_1d = TestFunction1DFactory.get_function_by_name(func_name)
        if func_1d:
            from test_function.test_function3d import TestFunction3D
            return TestFunction3D.from_1d(func_1d, method='triple_product')
        
        # デフォルト関数を返す
        print(f"警告: 3D関数 '{func_name}' が見つかりません。デフォルト関数を使用します。")
        funcs = TestFunction3DFactory.create_standard_functions()
        return funcs[0]
    
    def _build_rhs(self, test_func):
        """
        3D右辺ベクトル構築（簡略化）
        
        Args:
            test_func: テスト関数
            
        Returns:
            np.ndarray: 右辺ベクトル
        """
        # ソース項の計算
        is_derivative = isinstance(self.equation_set, DerivativeEquationSet3D)
        f_values = self._compute_source_term(test_func, is_derivative)
        
        # 境界条件辞書を作成
        boundary = self._create_3d_boundary_conditions(test_func)
        
        self._init_solver()
        return self.solver.rhs_builder.build_rhs_vector(f_values=f_values, **boundary)
    
    def _compute_source_term(self, test_func, is_derivative):
        """
        ソース項を計算（ヘルパーメソッド）
        
        Args:
            test_func: テスト関数
            is_derivative: 微分方程式かどうか
            
        Returns:
            ソース項の値
        """
        nx, ny, nz = self.grid.nx_points, self.grid.ny_points, self.grid.nz_points
        f_values = np.zeros((nx, ny, nz))
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    x, y, z = self.grid.get_point(i, j, k)
                    f_values[i, j, k] = test_func.f(x, y, z) if is_derivative else (
                        test_func.d2f_dx2(x, y, z) + test_func.d2f_dy2(x, y, z) + test_func.d2f_dz2(x, y, z)
                    )
        
        return f_values
    
    def _create_3d_boundary_conditions(self, test_func):
        """
        3D境界条件を作成（ヘルパーメソッド）
        
        Args:
            test_func: テスト関数
            
        Returns:
            境界条件の辞書
        """
        nx, ny, nz = self.grid.nx_points, self.grid.ny_points, self.grid.nz_points
        x_min, x_max = self.grid.x_min, self.grid.x_max
        y_min, y_max = self.grid.y_min, self.grid.y_max
        z_min, z_max = self.grid.z_min, self.grid.z_max
        
        boundary = {}
        
        # 面のディリクレ境界条件
        boundary['face_x_min_dirichlet'] = np.array([[test_func.f(x_min, y, z) 
                                                   for z in self.grid.z] for y in self.grid.y])
        boundary['face_x_max_dirichlet'] = np.array([[test_func.f(x_max, y, z) 
                                                   for z in self.grid.z] for y in self.grid.y])
        boundary['face_y_min_dirichlet'] = np.array([[test_func.f(x, y_min, z) 
                                                   for z in self.grid.z] for x in self.grid.x])
        boundary['face_y_max_dirichlet'] = np.array([[test_func.f(x, y_max, z) 
                                                   for z in self.grid.z] for x in self.grid.x])
        boundary['face_z_min_dirichlet'] = np.array([[test_func.f(x, y, z_min) 
                                                   for y in self.grid.y] for x in self.grid.x])
        boundary['face_z_max_dirichlet'] = np.array([[test_func.f(x, y, z_max) 
                                                   for y in self.grid.y] for x in self.grid.x])
        
        # ノイマン境界条件も必要に応じて追加
        if self.equation_set.enable_neumann:
            boundary['face_x_min_neumann_x'] = np.array([[test_func.df_dx(x_min, y, z) 
                                                      for z in self.grid.z] for y in self.grid.y])
            boundary['face_x_max_neumann_x'] = np.array([[test_func.df_dx(x_max, y, z) 
                                                      for z in self.grid.z] for y in self.grid.y])
            boundary['face_y_min_neumann_y'] = np.array([[test_func.df_dy(x, y_min, z) 
                                                      for z in self.grid.z] for x in self.grid.x])
            boundary['face_y_max_neumann_y'] = np.array([[test_func.df_dy(x, y_max, z) 
                                                      for z in self.grid.z] for x in self.grid.x])
            boundary['face_z_min_neumann_z'] = np.array([[test_func.df_dz(x, y, z_min) 
                                                      for y in self.grid.y] for x in self.grid.x])
            boundary['face_z_max_neumann_z'] = np.array([[test_func.df_dz(x, y, z_max) 
                                                      for y in self.grid.y] for x in self.grid.x])
        
        return boundary
    
    def _compute_exact(self, test_func):
        """
        3D厳密解計算（簡略化）
        
        Args:
            test_func: テスト関数
            
        Returns:
            np.ndarray: 厳密解ベクトル
        """
        nx, ny, nz = self.grid.nx_points, self.grid.ny_points, self.grid.nz_points
        exact = np.zeros(nx * ny * nz * 10)  # [ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy, ψ_z, ψ_zz, ψ_zzz]
        
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    idx = (k * ny * nx + j * nx + i) * 10
                    x, y, z = self.grid.get_point(i, j, k)
                    
                    exact[idx] = test_func.f(x, y, z)          # ψ
                    exact[idx+1] = test_func.df_dx(x, y, z)     # ψ_x
                    exact[idx+2] = test_func.d2f_dx2(x, y, z)   # ψ_xx
                    exact[idx+3] = test_func.d3f_dx3(x, y, z)   # ψ_xxx
                    exact[idx+4] = test_func.df_dy(x, y, z)     # ψ_y
                    exact[idx+5] = test_func.d2f_dy2(x, y, z)   # ψ_yy
                    exact[idx+6] = test_func.d3f_dy3(x, y, z)   # ψ_yyy
                    exact[idx+7] = test_func.df_dz(x, y, z)     # ψ_z
                    exact[idx+8] = test_func.d2f_dz2(x, y, z)   # ψ_zz
                    exact[idx+9] = test_func.d3f_dz3(x, y, z)   # ψ_zzz
                
        return exact
    
    def _compute_exact_components(self, test_func):
        """
        3D厳密解のコンポーネントを計算（ヘルパーメソッド）
        
        Args:
            test_func: テスト関数
            
        Returns:
            厳密解の各コンポーネント
        """
        nx, ny, nz = self.grid.nx_points, self.grid.ny_points, self.grid.nz_points
        
        # 厳密解準備
        exact = {
            'psi': np.zeros((nx, ny, nz)),
            'psi_x': np.zeros((nx, ny, nz)),
            'psi_y': np.zeros((nx, ny, nz)),
            'psi_z': np.zeros((nx, ny, nz)),
            'psi_xx': np.zeros((nx, ny, nz)),
            'psi_yy': np.zeros((nx, ny, nz)),
            'psi_zz': np.zeros((nx, ny, nz)),
            'psi_xxx': np.zeros((nx, ny, nz)),
            'psi_yyy': np.zeros((nx, ny, nz)),
            'psi_zzz': np.zeros((nx, ny, nz))
        }
        
        # 各点で厳密値計算
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    x, y, z = self.grid.get_point(i, j, k)
                    exact['psi'][i, j, k] = test_func.f(x, y, z)
                    exact['psi_x'][i, j, k] = test_func.df_dx(x, y, z)
                    exact['psi_y'][i, j, k] = test_func.df_dy(x, y, z)
                    exact['psi_z'][i, j, k] = test_func.df_dz(x, y, z)
                    exact['psi_xx'][i, j, k] = test_func.d2f_dx2(x, y, z)
                    exact['psi_yy'][i, j, k] = test_func.d2f_dy2(x, y, z)
                    exact['psi_zz'][i, j, k] = test_func.d2f_dz2(x, y, z)
                    exact['psi_xxx'][i, j, k] = test_func.d3f_dx3(x, y, z)
                    exact['psi_yyy'][i, j, k] = test_func.d3f_dy3(x, y, z)
                    exact['psi_zzz'][i, j, k] = test_func.d3f_dz3(x, y, z)
        
        return exact
    
    def _process_solution(self, test_func):
        """
        3D解処理（簡略化）
        
        Args:
            test_func: テスト関数
            
        Returns:
            Dict: 処理結果を含む辞書
        """
        nx, ny, nz = self.grid.nx_points, self.grid.ny_points, self.grid.nz_points
        is_derivative = isinstance(self.equation_set, DerivativeEquationSet3D)
        
        # 厳密解の計算
        exact_components = self._compute_exact_components(test_func)
        
        # ソース項準備
        f_values = self._compute_source_term(test_func, is_derivative)
        
        # 境界条件
        boundary = self._create_3d_boundary_conditions(test_func)
        
        # ソルバーオプション設定と実行
        if self.perturbation_level is not None and self.solver_method != 'direct' and self.exact_solution is not None:
            solve_options = {
                'tol': self.solver_options.get('tol', 1e-10),
                'maxiter': self.solver_options.get('maxiter', 1000),
                'x0': self.exact_solution
            }
            sol = self.solver.solve_with_options(
                analyze_before_solve=False, 
                f_values=f_values, 
                solve_options=solve_options, 
                **boundary
            )
        else:
            sol = self.solver.solve(
                analyze_before_solve=False, 
                f_values=f_values, 
                **boundary
            )
            
        # 数値解のコンポーネント抽出と変換
        numerical_vals = [self._to_numpy(comp) for comp in sol]
        exact_vals = [comp for comp in exact_components.values()]
        
        # 誤差計算
        errors = [float(np.max(np.abs(n - e))) for n, e in zip(numerical_vals, exact_vals)]
        
        return {
            "function": test_func.name,
            "numerical": numerical_vals,
            "exact": exact_vals,
            "errors": errors
        }