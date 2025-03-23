"""
高精度コンパクト差分法 (CCD) の3次元テスターモジュール

このモジュールは、3次元のCCDソルバーのテスト機能を提供します。
"""

import numpy as np

from core.base.base_tester import CCDTester
from core.solver.solver3d import CCDSolver3D
from equation_set.equation_sets import DerivativeEquationSet3D


class CCDTester3D(CCDTester):
    """3D CCDテスター"""
    
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
        3D右辺構築
        
        Args:
            test_func: テスト関数
            
        Returns:
            np.ndarray: 右辺ベクトル
        """
        nx, ny, nz = self.grid.nx_points, self.grid.ny_points, self.grid.nz_points
        x_min, x_max = self.grid.x_min, self.grid.x_max
        y_min, y_max = self.grid.y_min, self.grid.y_max
        z_min, z_max = self.grid.z_min, self.grid.z_max
        
        # ソース項 (方程式タイプに依存)
        is_derivative = isinstance(self.equation_set, DerivativeEquationSet3D)
        f_values = np.zeros((nx, ny, nz))
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    x, y, z = self.grid.get_point(i, j, k)
                    f_values[i, j, k] = test_func.f(x, y, z) if is_derivative else (
                        test_func.d2f_dx2(x, y, z) + test_func.d2f_dy2(x, y, z) + test_func.d2f_dz2(x, y, z)
                    )
        
        # 境界値（面、辺、頂点）
        boundary = {}
        
        # 面（x方向）
        x_min_values = np.array([[test_func.f(x_min, y, z) for z in self.grid.z] for y in self.grid.y])
        x_max_values = np.array([[test_func.f(x_max, y, z) for z in self.grid.z] for y in self.grid.y])
        boundary['face_x_min_dirichlet'] = x_min_values
        boundary['face_x_max_dirichlet'] = x_max_values
        
        # 面（y方向）
        y_min_values = np.array([[test_func.f(x, y_min, z) for z in self.grid.z] for x in self.grid.x])
        y_max_values = np.array([[test_func.f(x, y_max, z) for z in self.grid.z] for x in self.grid.x])
        boundary['face_y_min_dirichlet'] = y_min_values
        boundary['face_y_max_dirichlet'] = y_max_values
        
        # 面（z方向）
        z_min_values = np.array([[test_func.f(x, y, z_min) for y in self.grid.y] for x in self.grid.x])
        z_max_values = np.array([[test_func.f(x, y, z_max) for y in self.grid.y] for x in self.grid.x])
        boundary['face_z_min_dirichlet'] = z_min_values
        boundary['face_z_max_dirichlet'] = z_max_values
        
        # ノイマン境界条件（必要に応じて）
        if self.equation_set.enable_neumann:
            # x方向ノイマン
            x_min_neumann = np.array([[test_func.df_dx(x_min, y, z) for z in self.grid.z] for y in self.grid.y])
            x_max_neumann = np.array([[test_func.df_dx(x_max, y, z) for z in self.grid.z] for y in self.grid.y])
            boundary['face_x_min_neumann_x'] = x_min_neumann
            boundary['face_x_max_neumann_x'] = x_max_neumann
            
            # y方向ノイマン
            y_min_neumann = np.array([[test_func.df_dy(x, y_min, z) for z in self.grid.z] for x in self.grid.x])
            y_max_neumann = np.array([[test_func.df_dy(x, y_max, z) for z in self.grid.z] for x in self.grid.x])
            boundary['face_y_min_neumann_y'] = y_min_neumann
            boundary['face_y_max_neumann_y'] = y_max_neumann
            
            # z方向ノイマン
            z_min_neumann = np.array([[test_func.df_dz(x, y, z_min) for y in self.grid.y] for x in self.grid.x])
            z_max_neumann = np.array([[test_func.df_dz(x, y, z_max) for y in self.grid.y] for x in self.grid.x])
            boundary['face_z_min_neumann_z'] = z_min_neumann
            boundary['face_z_max_neumann_z'] = z_max_neumann
        
        self._init_solver()
        return self.solver.rhs_builder.build_rhs_vector(f_values=f_values, **boundary)
    
    def _compute_exact(self, test_func):
        """
        3D厳密解計算
        
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
    
    def _process_solution(self, test_func):
        """
        3D解処理（NumPy/CuPy互換性を考慮）
        
        Args:
            test_func: テスト関数
            
        Returns:
            Dict: 処理結果を含む辞書
        """
        nx, ny, nz = self.grid.nx_points, self.grid.ny_points, self.grid.nz_points
        X, Y, Z = self.grid.get_points()
        x_min, x_max = self.grid.x_min, self.grid.x_max
        y_min, y_max = self.grid.y_min, self.grid.y_max
        z_min, z_max = self.grid.z_min, self.grid.z_max
        is_derivative = isinstance(self.equation_set, DerivativeEquationSet3D)

        # 厳密解準備（NumPy配列として）
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

        # 右辺値準備
        f_values = np.zeros((nx, ny, nz))
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    x, y, z = self.grid.get_point(i, j, k)
                    f_values[i, j, k] = test_func.f(x, y, z) if is_derivative else (
                        test_func.d2f_dx2(x, y, z) + test_func.d2f_dy2(x, y, z) + test_func.d2f_dz2(x, y, z)
                    )

        # 境界条件
        boundary = {}
        
        # 面（x方向）
        boundary['face_x_min_dirichlet'] = np.array([[test_func.f(x_min, y, z) for z in self.grid.z] for y in self.grid.y])
        boundary['face_x_max_dirichlet'] = np.array([[test_func.f(x_max, y, z) for z in self.grid.z] for y in self.grid.y])
        
        # 面（y方向）
        boundary['face_y_min_dirichlet'] = np.array([[test_func.f(x, y_min, z) for z in self.grid.z] for x in self.grid.x])
        boundary['face_y_max_dirichlet'] = np.array([[test_func.f(x, y_max, z) for z in self.grid.z] for x in self.grid.x])
        
        # 面（z方向）
        boundary['face_z_min_dirichlet'] = np.array([[test_func.f(x, y, z_min) for y in self.grid.y] for x in self.grid.x])
        boundary['face_z_max_dirichlet'] = np.array([[test_func.f(x, y, z_max) for y in self.grid.y] for x in self.grid.x])
        
        # ノイマン境界条件（必要に応じて）
        if self.equation_set.enable_neumann:
            # x方向ノイマン
            boundary['face_x_min_neumann_x'] = np.array([[test_func.df_dx(x_min, y, z) for z in self.grid.z] for y in self.grid.y])
            boundary['face_x_max_neumann_x'] = np.array([[test_func.df_dx(x_max, y, z) for z in self.grid.z] for y in self.grid.y])
            
            # y方向ノイマン
            boundary['face_y_min_neumann_y'] = np.array([[test_func.df_dy(x, y_min, z) for z in self.grid.z] for x in self.grid.x])
            boundary['face_y_max_neumann_y'] = np.array([[test_func.df_dy(x, y_max, z) for z in self.grid.z] for x in self.grid.x])
            
            # z方向ノイマン
            boundary['face_z_min_neumann_z'] = np.array([[test_func.df_dz(x, y, z_min) for y in self.grid.y] for x in self.grid.x])
            boundary['face_z_max_neumann_z'] = np.array([[test_func.df_dz(x, y, z_max) for y in self.grid.y] for x in self.grid.x])

        # 厳密解を初期値として使用する場合
        if self.perturbation_level is not None and self.solver_method != 'direct' and self.exact_solution is not None:
            print("摂動が加えられた初期値を使用してsolve()を呼び出します")
            # オプションを作成して厳密解（摂動あり）を含める
            solve_options = {
                'tol': self.solver_options.get('tol', 1e-10),
                'maxiter': self.solver_options.get('maxiter', 1000),
                'x0': self.exact_solution  # 摂動が加えられた初期値を使用
            }
            
            # x0のサイズを確認
            x0_shape = solve_options['x0'].shape
            expected_shape = (nx * ny * nz * 10,)  # [ψ, ψ_x, ψ_xx, ψ_xxx, ψ_y, ψ_yy, ψ_yyy, ψ_z, ψ_zz, ψ_zzz] for each point
            if x0_shape != expected_shape:
                print(f"警告: x0のサイズが期待値と異なります: {x0_shape} vs {expected_shape}")
            
            # 解計算（カスタムオプション付き）
            sol = self.solver.solve_with_options(
                analyze_before_solve=False, 
                f_values=f_values, 
                solve_options=solve_options, 
                **boundary
            )
        else:
            # 通常の解計算
            sol = self.solver.solve(
                analyze_before_solve=False, 
                f_values=f_values, 
                **boundary
            )
            
        psi, psi_x, psi_xx, psi_xxx, psi_y, psi_yy, psi_yyy, psi_z, psi_zz, psi_zzz = sol

        # NumPy配列に変換（CuPyの場合）
        psi = self._to_numpy(psi)
        psi_x = self._to_numpy(psi_x)
        psi_y = self._to_numpy(psi_y)
        psi_z = self._to_numpy(psi_z)
        psi_xx = self._to_numpy(psi_xx)
        psi_yy = self._to_numpy(psi_yy)
        psi_zz = self._to_numpy(psi_zz)
        psi_xxx = self._to_numpy(psi_xxx)
        psi_yyy = self._to_numpy(psi_yyy)
        psi_zzz = self._to_numpy(psi_zzz)

        # 誤差計算（NumPy配列として）
        errors = [
            float(np.max(np.abs(psi - exact['psi']))),
            float(np.max(np.abs(psi_x - exact['psi_x']))),
            float(np.max(np.abs(psi_y - exact['psi_y']))),
            float(np.max(np.abs(psi_z - exact['psi_z']))),
            float(np.max(np.abs(psi_xx - exact['psi_xx']))),
            float(np.max(np.abs(psi_yy - exact['psi_yy']))),
            float(np.max(np.abs(psi_zz - exact['psi_zz']))),
            float(np.max(np.abs(psi_xxx - exact['psi_xxx']))),
            float(np.max(np.abs(psi_yyy - exact['psi_yyy']))),
            float(np.max(np.abs(psi_zzz - exact['psi_zzz'])))
        ]

        return {
            "function": test_func.name,
            "numerical": [psi, psi_x, psi_y, psi_z, psi_xx, psi_yy, psi_zz, psi_xxx, psi_yyy, psi_zzz],
            "exact": [v for v in exact.values()],
            "errors": errors
        }