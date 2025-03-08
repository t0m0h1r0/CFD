import cupy as cp

class Grid:
    """計算格子を管理するクラス"""

    def __init__(self, n_points, x_range):
        self.n_points = n_points
        self.x_min, self.x_max = x_range
        self.h = (self.x_max - self.x_min) / (self.n_points - 1)
        self.x = cp.linspace(self.x_min, self.x_max, self.n_points)

    def get_point(self, i):
        return self.x[i]

    def get_points(self):
        return self.x

    def get_spacing(self):
        return self.h