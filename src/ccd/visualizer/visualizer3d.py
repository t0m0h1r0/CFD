"""
High-Precision Compact Difference (CCD) 3D Visualization Module

This module provides visualization tools for 3D CCD solver results,
with specialized techniques for visualizing volumetric data through
slices, projections, and statistical analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from typing import List

from core.base.base_visualizer import BaseVisualizer


class CCDVisualizer3D(BaseVisualizer):
    """Class for visualizing 3D CCD solver results"""
    
    def __init__(self, output_dir="results_3d"):
        """Initialize with dedicated output directory"""
        super().__init__(output_dir)
        # Additional colormap settings
        self.cmap_gradient = 'coolwarm'  # For gradients
    
    def get_dimension_label(self) -> str:
        """Return dimension label"""
        return "3D"
    
    def get_error_types(self) -> List[str]:
        """Return list of error types"""
        return ["ψ", "ψ_x", "ψ_y", "ψ_z", "ψ_xx", "ψ_yy", "ψ_zz", "ψ_xxx", "ψ_yyy", "ψ_zzz"]
    
    def visualize_solution(self, grid, function_name, numerical, exact, errors, prefix="",
                          save=True, show=False, dpi=180):
        """
        Visualize 3D solution with a simplified dashboard approach
        
        Args:
            grid: Grid object
            function_name: Test function name
            numerical: List of numerical solutions
            exact: List of exact solutions
            errors: List of error values
            prefix: Filename prefix
            save: Whether to save the figure
            show: Whether to display the figure
            dpi: Resolution
            
        Returns:
            Output file path
        """
        # Grid size and center indices
        nx, ny, nz = grid.nx_points, grid.ny_points, grid.nz_points
        mid_x, mid_y, mid_z = nx // 2, ny // 2, nz // 2
        
        # Extract main components (convert to NumPy arrays)
        psi = self._to_numpy(numerical[0])       # Solution
        psi_ex = self._to_numpy(exact[0])        # Exact solution
        error = np.abs(psi - psi_ex)             # Error
        
        # Component names
        component_names = self.get_error_types()
        
        # Create 3x2 dashboard grid (simplified)
        fig = plt.figure(figsize=(16, 14))
        gs = GridSpec(3, 2)
        
        # 1. Solution slices (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_solution_slices(ax1, grid, psi, mid_x, mid_y, mid_z)
        
        # 2. Error heatmap (top-right)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_error_heatmap(ax2, grid, error, mid_z)
        
        # 3. Error histogram (middle-left)
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_error_histogram(ax3, error)
        
        # 4. Gradient slice (middle-right)
        ax4 = fig.add_subplot(gs[1, 1])
        psi_x = self._to_numpy(numerical[1])
        psi_y = self._to_numpy(numerical[2])
        psi_z = self._to_numpy(numerical[3])
        self._plot_gradient_slice(ax4, grid, psi_x, psi_y, psi_z, mid_z)
        
        # 5. Error summary (bottom-left)
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_error_summary(ax5, component_names, errors)
        
        # 6. Statistics (bottom-right)
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_statistics(ax6, numerical, exact, errors)
        
        # Overall settings
        plt.suptitle(f"{function_name} Function Analysis ({nx}x{ny}x{nz} points)", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save and display
        filepath = ""
        if save:
            filepath = self.generate_filename(function_name, (nx, ny, nz), prefix)
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return filepath
    
    def _plot_solution_slices(self, ax, grid, data, mid_x, mid_y, mid_z):
        """Display solution slices"""
        # Extract the middle slice
        xy_slice = data[:, :, mid_z]
        
        # Grid coordinates
        x = self._to_numpy(grid.x)
        y = self._to_numpy(grid.y)
        
        # Display middle slice (simplified)
        im = ax.imshow(xy_slice.T, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', 
                     cmap=self.cmap_solution, interpolation='bilinear')
        
        # Settings
        ax.set_title(f"Solution Slice at z={grid.z[mid_z]:.2f}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax)
    
    def _plot_error_heatmap(self, ax, grid, error, mid_z):
        """Display error distribution heatmap"""
        # Extract middle Z slice
        error_slice = error[:, :, mid_z]
        
        # Grid coordinates
        x = self._to_numpy(grid.x)
        y = self._to_numpy(grid.y)
        
        # Display on log scale
        err_min = max(np.min(error_slice[error_slice > 0]) if np.any(error_slice > 0) else 1e-10, 1e-15)
        im = ax.imshow(
            error_slice.T, 
            extent=[x[0], x[-1], y[0], y[-1]],
            norm=LogNorm(vmin=err_min, vmax=np.max(error_slice)),
            cmap=self.cmap_error, origin='lower'
        )
        
        # Settings
        ax.set_title(f"Error Distribution at z={grid.z[mid_z]:.2f}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax)
        
        # Mark maximum error position
        max_pos = np.unravel_index(np.argmax(error_slice), error_slice.shape)
        ax.plot(x[max_pos[0]], y[max_pos[1]], 'rx', markersize=8)
        ax.text(x[max_pos[0]], y[max_pos[1]], f" Max: {np.max(error_slice):.2e}", color='r')
    
    def _plot_error_histogram(self, ax, error):
        """Display error distribution histogram"""
        # Only include non-zero errors
        nonzero_errors = error[error > 0].flatten()
        
        if len(nonzero_errors) > 0:
            # Draw histogram
            counts, bins, _ = ax.hist(nonzero_errors, bins=30, alpha=0.7, color='skyblue',
                                    log=True, density=True)
            
            # Set x-axis to log scale
            ax.set_xscale('log')
            
            # Display statistics
            ax.text(0.05, 0.95, f"Max Error: {np.max(error):.2e}", 
                   transform=ax.transAxes, va='top')
            ax.text(0.05, 0.90, f"Mean Error: {np.mean(error):.2e}", 
                   transform=ax.transAxes, va='top')
            ax.text(0.05, 0.85, f"Median Error: {np.median(nonzero_errors):.2e}", 
                   transform=ax.transAxes, va='top')
        else:
            ax.text(0.5, 0.5, "No non-zero errors", ha='center', va='center', 
                  transform=ax.transAxes)
        
        # Settings
        ax.set_title("Error Distribution Histogram")
        ax.set_xlabel("Error (log scale)")
        ax.set_ylabel("Frequency (log scale)")
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
    
    def _plot_gradient_slice(self, ax, grid, grad_x, grad_y, grad_z, mid_z):
        """Display gradient vector slice"""
        # Extract middle Z slice
        x_slice = grad_x[:, :, mid_z]
        y_slice = grad_y[:, :, mid_z]
        
        # Grid coordinates
        x = self._to_numpy(grid.x)
        y = self._to_numpy(grid.y)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Calculate gradient magnitude
        magnitude = np.sqrt(x_slice**2 + y_slice**2)
        
        # Display gradient magnitude with colormap
        im = ax.imshow(
            magnitude.T, extent=[x[0], x[-1], y[0], y[-1]],
            cmap=self.cmap_gradient, origin='lower'
        )
        
        # Display gradient vectors (subsampled)
        stride = max(1, min(len(x), len(y)) // 20)
        ax.quiver(
            X[::stride, ::stride], Y[::stride, ::stride],
            x_slice[::stride, ::stride].T, y_slice[::stride, ::stride].T,
            angles='xy', scale_units='xy', scale=5, color='k', alpha=0.7
        )
        
        # Settings
        ax.set_title(f"Gradient Field at z={grid.z[mid_z]:.2f}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax, label="Gradient Magnitude")
    
    def _plot_statistics(self, ax, numerical, exact, errors):
        """Display solution statistics"""
        ax.axis('off')
        
        # Statistics for main components
        stats = []
        
        # ψ component statistics
        psi = self._to_numpy(numerical[0])
        psi_ex = self._to_numpy(exact[0])
        stats.extend([
            "Solution Statistics:",
            f"Max Value: {np.max(psi):.2e}",
            f"Min Value: {np.min(psi):.2e}",
            f"Mean Value: {np.mean(psi):.2e}",
            f"Max Error: {errors[0]:.2e}",
            f"Error Ratio: {errors[0]/np.max(np.abs(psi_ex)):.2e}",
            ""
        ])
        
        # Gradient statistics
        grad_x = self._to_numpy(numerical[1])
        grad_y = self._to_numpy(numerical[2])
        grad_z = self._to_numpy(numerical[3])
        grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        
        stats.extend([
            "Gradient Statistics:",
            f"Max Magnitude: {np.max(grad_mag):.2e}",
            f"Mean Magnitude: {np.mean(grad_mag):.2e}",
            f"Max Gradient Error: {max(errors[1:4]):.2e}"
        ])
        
        # Display
        ax.text(0.05, 0.95, '\n'.join(stats), transform=ax.transAxes, va='top')
        ax.set_title("Statistics")