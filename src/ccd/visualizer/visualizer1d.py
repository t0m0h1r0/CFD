"""
High-Precision Compact Difference (CCD) 1D Visualization Module

This module provides visualization tools for 1D CCD solver results,
displaying numerical and exact solutions along with error analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import List

from core.base.base_visualizer import BaseVisualizer


class CCDVisualizer1D(BaseVisualizer):
    """Class for visualizing 1D CCD solver results"""

    def __init__(self, output_dir="results_1d"):
        """Initialize with dedicated output directory"""
        super().__init__(output_dir)

    def get_dimension_label(self) -> str:
        """Return dimension label"""
        return "1D"
    
    def get_error_types(self) -> List[str]:
        """Return list of error types"""
        return ["ψ", "ψ'", "ψ''", "ψ'''"]
    
    # 互換性のために旧メソッド名も維持
    def visualize_derivatives(self, grid, function_name, numerical, exact, errors, prefix="",
                             save=True, show=False, dpi=150):
        """
        Backward compatibility method - redirects to visualize_solution
        """
        return self.visualize_solution(grid, function_name, numerical, exact, errors, 
                                      prefix, save, show, dpi)

    def visualize_solution(self, grid, function_name, numerical, exact, errors, prefix="",
                          save=True, show=False, dpi=150):
        """
        Visualize 1D solution in a comprehensive dashboard
        
        Args:
            grid: Grid object
            function_name: Test function name
            numerical: List of numerical solutions [psi, psi', psi'', psi''']
            exact: List of exact solutions
            errors: List of error values
            prefix: Filename prefix
            save: Whether to save the figure
            show: Whether to display the figure
            dpi: Resolution
            
        Returns:
            Output file path
        """
        # Grid data
        x_np = self._to_numpy(grid.get_points())
        n_points = grid.n_points
        
        # Component names
        component_names = self.get_error_types()
        
        # Create 3x2 dashboard grid
        fig = plt.figure(figsize=(12, 10))
        gs = GridSpec(3, 2, height_ratios=[1, 1, 0.7])
        
        # Visualize each derivative
        for i in range(4):
            row, col = divmod(i, 2)
            ax = fig.add_subplot(gs[row, col])
            
            # Convert data
            exact_data = self._to_numpy(exact[i])
            num_data = self._to_numpy(numerical[i])
            
            # Visualize solution and error simultaneously
            self._plot_solution_with_error(ax, x_np, exact_data, num_data, component_names[i], errors[i])
        
        # Error summary
        ax_summary = fig.add_subplot(gs[2, :])
        self._plot_error_summary(ax_summary, component_names, errors)
        
        # Overall settings
        plt.suptitle(f"{function_name} Function Analysis ({n_points} points)", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Margin for title

        # Save and display
        filepath = ""
        if save:
            filepath = self.generate_filename(function_name, n_points, prefix)
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close(fig)

        return filepath
    
    def _plot_solution_with_error(self, ax, x, exact, numerical, title, max_error):
        """
        Plot solution and error in a single graph
        
        Args:
            ax: matplotlib Axes
            x: x coordinate values
            exact: Exact solution data
            numerical: Numerical solution data
            title: Plot title
            max_error: Maximum error value
        """
        # Left Y-axis: Solution values
        ax.plot(x, exact, "b-", label="Exact", linewidth=1.5)
        ax.plot(x, numerical, "r--", label="Numerical", linewidth=1.5)
        ax.set_xlabel("x")
        ax.set_ylabel("Value")
        
        # Right Y-axis: Error (log scale)
        ax2 = ax.twinx()
        error = np.abs(numerical - exact)
        
        # Handle zero errors
        min_error = max(np.min(error[error > 0]) if np.any(error > 0) else 1e-15, 1e-15)
        plot_error = np.maximum(error, min_error)
        
        # Error plot
        ax2.semilogy(x, plot_error, "g-", alpha=0.3, label="Error")
        ax2.fill_between(x, min_error, plot_error, color='green', alpha=0.1)
        ax2.set_ylabel("Error (log)", color="g")
        ax2.tick_params(axis="y", labelcolor="g")
        
        # Mark maximum error point
        max_err_idx = np.argmax(error)
        max_err_x = x[max_err_idx]
        max_err_y = error[max_err_idx]
        
        # Only show marker if max error is non-zero
        if max_err_y > 0:
            ax2.plot(max_err_x, max_err_y, "go", ms=4)
            ax2.annotate("Max", 
                      xy=(max_err_x, max_err_y),
                      xytext=(5, 5),
                      textcoords="offset points",
                      fontsize=8,
                      color="g")
        
        # Title
        ax.set_title(f"{title} (Error: {max_error:.2e})")
        
        # Legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)
        
        # Grid
        ax.grid(True, alpha=0.3)