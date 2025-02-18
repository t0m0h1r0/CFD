from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from .types import Grid, GridType, Vector3D

@dataclass
class GridConfig:
    """Configuration for grid generation."""
    dimensions: Vector3D  # Physical dimensions (Lx, Ly, Lz)
    points: Vector3D  # Number of points (nx, ny, nz)
    grid_type: GridType = GridType.UNIFORM
    stretching_parameters: Optional[Dict[str, float]] = None

class GridManager:
    """Manager for computational grid operations."""
    
    def __init__(self, config: GridConfig):
        """
        Initialize grid manager.
        
        Args:
            config: Grid configuration
        """
        self.config = config
        self._initialize_grid()
        
    def _initialize_grid(self) -> None:
        """Initialize grid based on configuration."""
        if self.config.grid_type == GridType.UNIFORM:
            self.grid = self._create_uniform_grid()
        elif self.config.grid_type == GridType.STRETCHED:
            self.grid = self._create_stretched_grid()
        else:
            raise ValueError(f"Unsupported grid type: {self.config.grid_type}")
            
        # Calculate grid metrics
        self._calculate_metrics()
        
    def _create_uniform_grid(self) -> Grid:
        """Create uniform grid."""
        Lx, Ly, Lz = self.config.dimensions
        nx, ny, nz = self.config.points
        
        x = jnp.linspace(0, Lx, nx)
        y = jnp.linspace(0, Ly, ny)
        z = jnp.linspace(0, Lz, nz)
        
        # Create 3D grid
        self.x_3d, self.y_3d, self.z_3d = jnp.meshgrid(x, y, z, indexing='ij')
        
        return x, y, z
        
    def _create_stretched_grid(self) -> Grid:
        """Create stretched grid using hyperbolic tangent stretching."""
        Lx, Ly, Lz = self.config.dimensions
        nx, ny, nz = self.config.points
        
        params = self.config.stretching_parameters or {}
        beta = params.get('beta', 1.0)  # Stretching parameter
        
        # Create stretched coordinates
        def stretch_coordinate(N: int, L: float) -> ArrayLike:
            xi = jnp.linspace(-1, 1, N)
            return L/2 * (1 + jnp.tanh(beta * xi)/jnp.tanh(beta))
            
        x = stretch_coordinate(nx, Lx)
        y = stretch_coordinate(ny, Ly)
        z = stretch_coordinate(nz, Lz)
        
        # Create 3D grid
        self.x_3d, self.y_3d, self.z_3d = jnp.meshgrid(x, y, z, indexing='ij')
        
        return x, y, z
        
    def _calculate_metrics(self) -> None:
        """Calculate grid metrics (spacing, jacobian, etc.)."""
        x, y, z = self.grid
        
        # Calculate grid spacing
        self.dx = jnp.diff(x)
        self.dy = jnp.diff(y)
        self.dz = jnp.diff(z)
        
        # Calculate metrics at cell centers
        self.dx_center = jnp.concatenate([self.dx, self.dx[-1:]])
        self.dy_center = jnp.concatenate([self.dy, self.dy[-1:]])
        self.dz_center = jnp.concatenate([self.dz, self.dz[-1:]])
        
        # Calculate cell volumes
        self.volume = (self.dx_center[:, None, None] * 
                      self.dy_center[None, :, None] * 
                      self.dz_center[None, None, :])
        
    @partial(jax.jit, static_argnums=(0,))
    def interpolate(self,
                   field: ArrayLike,
                   location: Tuple[float, float, float]
                   ) -> float:
        """
        Interpolate field value at given location.
        
        Args:
            field: Field to interpolate
            location: Physical coordinates (x, y, z)
            
        Returns:
            Interpolated value
        """
        x, y, z = location
        
        # Find nearest grid points
        ix = jnp.searchsorted(self.grid[0], x)
        iy = jnp.searchsorted(self.grid[1], y)
        iz = jnp.searchsorted(self.grid[2], z)
        
        # Calculate interpolation weights
        wx = (x - self.grid[0][ix-1]) / (self.grid[0][ix] - self.grid[0][ix-1])
        wy = (y - self.grid[1][iy-1]) / (self.grid[1][iy] - self.grid[1][iy-1])
        wz = (z - self.grid[2][iz-1]) / (self.grid[2][iz] - self.grid[2][iz-1])
        
        # Perform trilinear interpolation
        c000 = field[ix-1, iy-1, iz-1]
        c001 = field[ix-1, iy-1, iz]
        c010 = field[ix-1, iy, iz-1]
        c011 = field[ix-1, iy, iz]
        c100 = field[ix, iy-1, iz-1]
        c101 = field[ix, iy-1, iz]
        c110 = field[ix, iy, iz-1]
        c111 = field[ix, iy, iz]
        
        return ((1-wx)*((1-wy)*((1-wz)*c000 + wz*c001) + 
                         wy*((1-wz)*c010 + wz*c011)) +
                wx*((1-wy)*((1-wz)*c100 + wz*c101) +
                     wy*((1-wz)*c110 + wz*c111)))
    
    def get_grid_spacing(self, direction: str) -> ArrayLike:
        """Get grid spacing for given direction."""
        if direction == 'x':
            return self.dx_center
        elif direction == 'y':
            return self.dy_center
        elif direction == 'z':
            return self.dz_center
        else:
            raise ValueError(f"Invalid direction: {direction}")
            
    def get_grid_points(self, direction: str) -> int:
        """Get number of grid points in given direction."""
        if direction == 'x':
            return self.config.points[0]
        elif direction == 'y':
            return self.config.points[1]
        elif direction == 'z':
            return self.config.points[2]
        else:
            raise ValueError(f"Invalid direction: {direction}")
            
    def get_coordinates(self) -> Grid:
        """Get grid coordinates."""
        return self.grid
        
    def get_volume(self) -> ArrayLike:
        """Get cell volumes."""
        return self.volume
        
    def get_face_areas(self) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """Get face areas for each direction."""
        # Calculate face areas
        Ax = self.dy_center[None, :, None] * self.dz_center[None, None, :]
        Ay = self.dx_center[:, None, None] * self.dz_center[None, None, :]
        Az = self.dx_center[:, None, None] * self.dy_center[None, :, None]
        
        return Ax, Ay, Az