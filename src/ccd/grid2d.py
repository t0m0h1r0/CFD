import cupy as cp

class Grid2D:
    """2D computation grid for CCD method"""
    
    def __init__(self, nx_points, ny_points, x_range, y_range):
        """
        Initialize a 2D grid
        
        Args:
            nx_points: Number of grid points in x direction
            ny_points: Number of grid points in y direction
            x_range: Tuple (x_min, x_max)
            y_range: Tuple (y_min, y_max)
        """
        self.nx_points = nx_points
        self.ny_points = ny_points
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        
        self.hx = (self.x_max - self.x_min) / (self.nx_points - 1)
        self.hy = (self.y_max - self.y_min) / (self.ny_points - 1)
        
        self.x = cp.linspace(self.x_min, self.x_max, self.nx_points)
        self.y = cp.linspace(self.y_min, self.y_max, self.ny_points)
        
        # Create mesh grid (useful for vectorized computations)
        self.X, self.Y = cp.meshgrid(self.x, self.y, indexing='ij')
    
    def get_point(self, i, j):
        """Get coordinates of grid point (i,j)"""
        return self.x[i], self.y[j]
    
    def get_points(self):
        """Get all grid points as meshgrid"""
        return self.X, self.Y
    
    def get_spacing(self):
        """Get grid spacing in both directions"""
        return self.hx, self.hy
    
    def get_index(self, i, j):
        """Convert 2D indices to 1D index in the flattened array"""
        return i + j * self.nx_points
    
    def get_indices(self, flat_index):
        """Convert flattened 1D index back to 2D indices"""
        j = flat_index // self.nx_points
        i = flat_index % self.nx_points
        return i, j
    
    def is_boundary_point(self, i, j):
        """Check if (i,j) is a boundary point"""
        return (i == 0 or i == self.nx_points - 1 or 
                j == 0 or j == self.ny_points - 1)
    
    def is_corner_point(self, i, j):
        """Check if (i,j) is a corner point"""
        return ((i == 0 or i == self.nx_points - 1) and 
                (j == 0 or j == self.ny_points - 1))
    
    def is_interior_point(self, i, j):
        """Check if (i,j) is an interior point"""
        return (0 < i < self.nx_points - 1 and 
                0 < j < self.ny_points - 1)