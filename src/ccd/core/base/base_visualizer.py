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
        self.cmap_solution = 'viridis'  # Solution colormap
        self.cmap_error = 'hot'  # Error colormap

    def _ensure_directory(self):
        """Ensure output directory exists"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    def _to_numpy(self, arr):
        """Convert array to NumPy format"""
        return arr.get() if hasattr(arr, 'get') else arr

    def _prepare_for_log_scale(self, data, min_value=None):
        """
        Preprocess data for log-scale visualization
        
        Args:
            data: Input data
            min_value: Minimum value (auto-detected if not specified)
            
        Returns:
            Preprocessed data
        """
        min_val = min_value or self.min_log_value
        
        # Replace values <= 0
        if isinstance(data, np.ndarray):
            result = data.copy()
            result[result <= 0] = min_val
            return result
        elif isinstance(data, (list, tuple)):
            return [max(x, min_val) for x in data]
        else:
            return max(data, min_val)

    def generate_filename(self, func_name, grid_info, prefix=""):
        """
        Generate filename for output visualization
        
        Args:
            func_name: Name of the test function
            grid_info: Grid information (point count or tuple)
            prefix: Optional prefix for filename
            
        Returns:
            Generated file path
        """
        # Process grid information
        if isinstance(grid_info, (list, tuple)):
            grid_info_str = "x".join(str(x) for x in grid_info)
        else:
            grid_info_str = f"{grid_info}"
        
        # Process prefix
        prefix_str = f"{prefix}_" if prefix else ""
        
        return f"{self.output_dir}/{prefix_str}{func_name.lower()}_{grid_info_str}_points.png"

    def _plot_error_summary(self, ax, component_names, errors, title="Error Summary"):
        """
        Plot error summary bar chart (common implementation)
        
        Args:
            ax: matplotlib Axes
            component_names: List of component names
            errors: List of error values
            title: Plot title
        """
        # Process for log scale
        plot_errors = np.array(errors).copy()
        plot_errors[plot_errors <= 0] = self.min_log_value
        
        # Create horizontal bar chart for better readability
        y_pos = np.arange(len(component_names))
        bars = ax.barh(y_pos, plot_errors, color='skyblue')
        
        # Settings
        ax.set_xscale('log')
        ax.set_title(title)
        ax.set_xlabel('Maximum Error (log scale)')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(component_names)
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        
        # Add value labels to bars
        for i, (bar, err) in enumerate(zip(bars, errors)):
            ax.text(
                max(err, self.min_log_value) * 1.1, 
                i, 
                f'{err:.2e}',
                va='center'
            )

    def visualize_grid_convergence(self, function_name, grid_sizes, results, prefix="", 
                                  save=True, show=False, dpi=150):
        """
        Visualize grid convergence test results (dimension-independent implementation)
        
        Args:
            function_name: Test function name
            grid_sizes: List of grid sizes
            results: Result data
            prefix: Output file prefix
            save: Whether to save the figure
            show: Whether to display the figure
            dpi: Image resolution
            
        Returns:
            Boolean indicating success
        """
        component_names = self.get_error_types()
        dimension_label = self.get_dimension_label()
        
        # Calculate layout based on component count
        n_components = len(component_names)
        rows = (n_components + 1) // 2
        
        # Create grid
        fig, axes = plt.subplots(rows, 2, figsize=(12, 3 * rows))
        axes = axes.flatten() if rows > 1 else [axes]
        
        # Calculate grid spacings
        grid_spacings = [2.0 / (n - 1) for n in grid_sizes]
        
        # Store convergence order estimates
        convergence_orders = []
        
        for i, (ax, name) in enumerate(zip(axes, component_names)):
            if i < n_components:
                # Get errors for each grid size
                errors = [results[n][i] for n in grid_sizes]
                
                # Skip if all errors are zero
                if all(err == 0 for err in errors):
                    ax.text(0.5, 0.5, "All errors are zero", 
                            ha='center', va='center', transform=ax.transAxes)
                    convergence_orders.append(float('inf'))
                    continue
                
                # Process for log scale
                plot_errors = [max(err, self.min_log_value) for err in errors]
                
                # Plot
                ax.loglog(grid_spacings, plot_errors, 'o-', label=name)
                
                # Estimate convergence order
                if len(grid_spacings) >= 2 and all(err > 0 for err in plot_errors):
                    # Simple slope calculation using last two points
                    order = (np.log(plot_errors[-2]) - np.log(plot_errors[-1])) / \
                           (np.log(grid_spacings[-2]) - np.log(grid_spacings[-1]))
                    convergence_orders.append(order)
                    
                    # Display estimated order
                    ax.text(0.05, 0.05, f"Order: {order:.2f}", 
                           transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.7))
                else:
                    convergence_orders.append(0)
                
                # Reference lines for 2nd and 4th order
                if min(plot_errors) > 0:
                    x_ref = np.array([min(grid_spacings), max(grid_spacings)])
                    for order, style, color in zip([2, 4], ['--', '-.'], ['r', 'g']):
                        # Scale to match the last point
                        scale = plot_errors[-1] / (grid_spacings[-1] ** order)
                        y_ref = scale * x_ref ** order
                        ax.loglog(x_ref, y_ref, style, color=color, label=f"O(h^{order})")
                
                # Settings
                ax.set_title(f"{name} Convergence")
                ax.set_xlabel("Grid Spacing (h)")
                ax.set_ylabel("Error")
                ax.grid(True, which='both', alpha=0.3)
                ax.legend(fontsize=8)
        
        # Calculate average convergence order
        valid_orders = [o for o in convergence_orders if o != float('inf') and o > 0]
        avg_order = np.mean(valid_orders) if valid_orders else 0
        
        # Title and settings
        plt.suptitle(f"{dimension_label} Grid Convergence: {function_name} (Avg. Order: {avg_order:.2f})")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save and display
        if save:
            filepath = f"{self.output_dir}/{prefix}_{function_name.lower()}_grid_convergence.png"
            if not prefix:
                filepath = f"{self.output_dir}/{function_name.lower()}_grid_convergence.png"
            plt.savefig(filepath, dpi=dpi)
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return True

    def compare_all_functions_errors(self, results_summary, grid_size=None, prefix="", 
                                    save=True, dpi=150, show=False):
        """
        Generate error comparison graph for all test functions (dimension-independent)
        
        Args:
            results_summary: Dictionary with function names as keys and error lists as values
            grid_size: Grid size
            prefix: Output file prefix
            save: Whether to save the figure
            dpi: Image resolution
            show: Whether to display the figure
            
        Returns:
            Output file path
        """
        func_names = list(results_summary.keys())
        component_names = self.get_error_types()
        dimension_label = self.get_dimension_label()
        
        # Calculate layout based on component count
        n_components = len(component_names)
        rows = (n_components + 1) // 2
        
        # Create grid
        fig, axes = plt.subplots(rows, 2, figsize=(14, 3 * rows))
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        
        for i, (ax, error_type) in enumerate(zip(axes, component_names)):
            if i < n_components:
                # Get corresponding errors for each function
                orig_errors = [results_summary[name][i] for name in func_names]
                
                # Process for log scale
                plot_errors = [max(err, self.min_log_value) for err in orig_errors]
                
                # Bar chart
                x_pos = np.arange(len(func_names))
                ax.bar(x_pos, plot_errors)
                ax.set_yscale('log')
                ax.set_title(f"{error_type}")
                ax.set_xticks(x_pos)
                ax.set_xticklabels(func_names, rotation=45, ha="right", fontsize=8)
                ax.grid(True, which='both', linestyle='--', alpha=0.5)
                
                # Add value annotations
                for j, (err, pos) in enumerate(zip(orig_errors, x_pos)):
                    label = "0.0" if err == 0.0 else f"{err:.1e}"
                    ax.annotate(
                        label,
                        xy=(pos, plot_errors[j] * 1.1),
                        ha='center', va='bottom',
                        fontsize=7, rotation=90
                    )
        
        # Overall settings
        plt.suptitle(f"{dimension_label} Error Comparison ({grid_size} points)")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save and display
        filename = f"{self.output_dir}/{prefix}_function_comparison"
        if not prefix:
            filename = f"{self.output_dir}/function_comparison"
            
        if grid_size:
            filename += f"_{grid_size}"
        
        filename += ".png"
            
        if save:
            plt.savefig(filename, dpi=dpi)
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return filename

    # Abstract methods (to be implemented by subclasses)
    @abstractmethod
    def visualize_solution(self, grid, function_name, numerical, exact, errors, prefix="", 
                         save=True, show=False, dpi=150):
        """Visualize solution"""
        pass
    
    @abstractmethod
    def get_dimension_label(self) -> str:
        """Return dimension label"""
        pass
    
    @abstractmethod
    def get_error_types(self) -> List[str]:
        """Return list of error types"""
        pass