"""
Base Visualizer for High-Precision Compact Difference (CCD) Method Results

This module provides the base class and common functionality for visualizing
CCD solver calculation results across different dimensions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import List


class BaseVisualizer(ABC):
    """Base class for visualizing CCD solver results"""

    def __init__(self, output_dir="results"):
        """
        Initialize visualizer

        Args:
            output_dir: Output directory path for saving visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.min_log_value = 1e-16  # Minimum value for log-scale plotting

    def generate_filename(self, func_name, n_points, prefix=""):
        """
        Generate filename for output visualization

        Args:
            func_name: Name of the test function
            n_points: Number of grid points (string or numeric)
            prefix: Optional prefix for filename

        Returns:
            Generated file path
        """
        if prefix:
            return f"{self.output_dir}/{prefix}_{func_name.lower()}_{n_points}_points.png"
        else:
            return f"{self.output_dir}/{func_name.lower()}_{n_points}_points.png"

    def compare_all_functions_errors(self, results_summary, grid_size=None, prefix="", dpi=150, show=False):
        """
        Generate error comparison graph for all test functions

        Args:
            results_summary: Summary dictionary of results for all functions
            grid_size: Grid size
            prefix: Output file prefix
            dpi: Image resolution
            show: Whether to display the figure

        Returns:
            Output file path
        """
        func_names = list(results_summary.keys())
        
        # Set labels based on dimension
        self.get_dimension_label()
        error_types = self.get_error_types()
        
        # Create figure and subplots
        fig, axes = plt.subplots(2, 4, figsize=(15, 10))
        
        for i, (ax, error_type) in enumerate(zip(axes.flat, error_types)):
            if i < len(error_types):
                # Get original errors for the specific error type
                original_errors = [results_summary[name][i] for name in func_names]
                
                # Replace zero values with a small value for log scale
                errors = []
                for err in original_errors:
                    if err == 0.0:
                        errors.append(self.min_log_value)
                    else:
                        errors.append(err)
                
                x_positions = np.arange(len(func_names))
                bars = ax.bar(x_positions, errors)
                ax.set_yscale("log")
                ax.set_title(f"{error_type} Error Comparison")
                ax.set_xlabel("Test Function")
                ax.set_ylabel("Error (log scale)")
                ax.grid(True, which="both", linestyle="--", alpha=0.5)
                ax.set_xticks(x_positions)
                ax.set_xticklabels(func_names, rotation=45, ha="right")
                
                # Annotate bars with values
                for j, (bar, orig_err) in enumerate(zip(bars, original_errors)):
                    height = bar.get_height()
                    label_text = "0.0" if orig_err == 0.0 else f"{orig_err:.2e}"
                    y_pos = height * 1.1
                    ax.annotate(
                        label_text,
                        xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
        
        plt.suptitle(f"Error Comparison for All Functions ({grid_size} points)")
        plt.tight_layout()
        
        # Generate output filename
        if prefix:
            filename = f"{self.output_dir}/{prefix}_all_functions_comparison"
        else:
            filename = f"{self.output_dir}/all_functions_comparison"
            
        if grid_size:
            filename += f"_{grid_size}"
        
        filename += ".png"
            
        plt.savefig(filename, dpi=dpi)
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return filename
        
    @abstractmethod
    def visualize_grid_convergence(self, function_name, grid_sizes, results, prefix="", save=True, show=False, dpi=150):
        """
        Generate grid convergence graph

        Args:
            function_name: Name of the test function
            grid_sizes: List of grid sizes
            results: Result data
            prefix: Output file prefix
            save: Whether to save the figure
            show: Whether to display the figure
            dpi: Image resolution

        Returns:
            Boolean indicating success
        """
        pass
    
    @abstractmethod
    def get_dimension_label(self) -> str:
        """
        Return dimension label

        Returns:
            "1D" or "2D"
        """
        pass
    
    @abstractmethod
    def get_error_types(self) -> List[str]:
        """
        Return list of error types

        Returns:
            List of error type names depending on dimension
        """
        pass
    
    def _to_numpy(self, arr):
        """
        Convert array to NumPy format (if necessary)

        Args:
            arr: Input array (CuPy or NumPy)

        Returns:
            NumPy array
        """
        return arr.get() if hasattr(arr, 'get') else arr