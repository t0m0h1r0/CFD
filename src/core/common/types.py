from typing import Tuple, Union, Literal, TypeVar, Protocol, Callable
from dataclasses import dataclass
from enum import Enum

from jax.typing import ArrayLike

# Type aliases
Scalar = Union[float, int]
Vector3D = Tuple[Scalar, Scalar, Scalar]
Grid = Tuple[ArrayLike, ArrayLike, ArrayLike]  # (x, y, z) grid coordinates

# Grid type definitions
class GridType(Enum):
    """Supported grid types."""
    UNIFORM = "uniform"
    NONUNIFORM = "nonuniform"
    STRETCHED = "stretched"

# Boundary condition type definitions
class BCType(Enum):
    """Supported boundary condition types."""
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    PERIODIC = "periodic"
    
@dataclass
class BoundaryCondition:
    """Container for boundary condition information."""
    type: BCType
    value: Union[Scalar, Callable]  # Can be constant or function
    location: Literal["left", "right", "bottom", "top", "front", "back"]

# Solver type definitions
class SolverType(Enum):
    """Supported solver types."""
    CG = "conjugate_gradient"
    SOR = "successive_over_relaxation"
    DIRECT = "direct"

# Discretization type definitions
class DiscretizationType(Enum):
    """Supported spatial discretization types."""
    CCD = "combined_compact"
    COMPACT = "compact"
    CENTRAL = "central"

# Protocol for functions that can be differentiated
class DifferentiableFunction(Protocol):
    """Protocol for functions that support differentiation."""
    def __call__(self, x: ArrayLike, *args, **kwargs) -> ArrayLike:
        ...

# Type variable for generic grid operations
T = TypeVar('T', bound=ArrayLike)

# Error types
class NumericalError(Exception):
    """Base class for numerical errors."""
    pass

class ConvergenceError(NumericalError):
    """Error raised when solver fails to converge."""
    pass

class StabilityError(NumericalError):
    """Error raised when numerical instability is detected."""
    pass

class GridError(NumericalError):
    """Error raised for grid-related issues."""
    pass