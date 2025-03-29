"""
High-Precision Compact Difference (CCD) 2D Visualization Module

This module provides visualization tools for 2D CCD solver results,
displaying solution components and error analysis in a comprehensive dashboard.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from typing import List

from core.base.base_visualizer import BaseVisualizer


class CCDVisualizer2D(BaseVisualizer):
    """Class for visualizing 2D CCD solver results"""
    
    def __init__(self, output_dir="results_2d"):
        """Initialize with dedicated output directory"""
        super().__init__(output_dir)
    
    def get_dimension_label(self) -> str:
        """Return dimension label"""
        return "2D"
    
    def get_error_types(self) -> List[str]:
        """Return list of error types"""
        return ["ψ", "ψ_x", "ψ_y", "ψ_xx", "ψ_yy", "ψ_xxx", "ψ_yyy"]
    
    def visualize_solution(self, grid, function_name, numerical, exact, errors, prefix="",
                          save=True, show=False, dpi=150):
        """
        Visualize 2D solution in a comprehensive dashboard
        
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
        # Grid information
        X, Y = grid.get_points()
        nx_points, ny_points = grid.nx_points, grid.ny_points
        
        # Convert to NumPy arrays
        X_np = self._to_numpy(X)
        Y_np = self._to_numpy(Y)
        
        # Component names
        solution_names = self.get_error_types()
        
        # Create 3x3 dashboard grid
        fig = plt.figure(figsize=(16, 14))
        gs = GridSpec(3, 3, height_ratios=[1, 1, 0.6])
        
        # Visualize main components (first 6)
        display_count = min(6, len(solution_names))
        for i in range(display_count):
            row, col = divmod(i, 3)
            ax = fig.add_subplot(gs[row, col])
            
            # Get and convert data
            num_np = self._to_numpy(numerical[i])
            err = errors[i]
            
            # Display contour map
            im = ax.contourf(X_np, Y_np, num_np, 20, cmap=self.cmap_solution)
            ax.set_title(f"{solution_names[i]} (Error: {err:.2e})")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.colorbar(im, ax=ax)
            
            # Add contour lines
            ax.contour(X_np, Y_np, num_np, 8, colors='k', linewidths=0.5, alpha=0.5)
        
        # Error distribution heatmap (bottom-left)
        ax_err = fig.add_subplot(gs[2, 0])
        
        # Main component error
        num_np = self._to_numpy(numerical[0])
        ex_np = self._to_numpy(exact[0])
        error_np = np.abs(num_np - ex_np)
        
        # Display error on log scale
        err_min = max(np.min(error_np[error_np > 0]) if np.any(error_np > 0) else 1e-10, 1e-15)
        im_err = ax_err.pcolormesh(
            X_np, Y_np, error_np, 
            norm=LogNorm(vmin=err_min, vmax=np.max(error_np)),
            cmap=self.cmap_error, shading='auto'
        )
        
        ax_err.set_title("Error Distribution (ψ)")
        ax_err.set_xlabel('X')
        ax_err.set_ylabel('Y')
        plt.colorbar(im_err, ax=ax_err)
        
        # Error summary graph (bottom-middle and bottom-right combined)
        ax_summary = fig.add_subplot(gs[2, 1:])
        self._plot_error_summary(ax_summary, solution_names, errors)
        
        # Overall settings
        plt.suptitle(
            f"{function_name} Function Analysis ({nx_points}x{ny_points} points)", 
            fontsize=16
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save and display
        filepath = ""
        if save:
            filepath = self.generate_filename(function_name, (nx_points, ny_points), prefix)
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return filepath