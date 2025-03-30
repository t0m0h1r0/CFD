"""
High-Precision Compact Difference (CCD) 3D Visualization Module

This module provides visualization tools for 3D CCD solver results,
with specialized techniques for visualizing volumetric data through
slices, projections, isosurfaces and statistical analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure
from typing import List

from core.base.base_visualizer import BaseVisualizer


class CCDVisualizer3D(BaseVisualizer):
    """Class for visualizing 3D CCD solver results"""
    
    def __init__(self, output_dir="results_3d"):
        """Initialize with dedicated output directory"""
        super().__init__(output_dir)
        # Additional colormap settings
        self.cmap_gradient = 'coolwarm'  # For gradients
        self.cmap_isosurface = 'viridis'  # For isosurfaces
    
    def get_dimension_label(self) -> str:
        """Return dimension label"""
        return "3D"
    
    def get_error_types(self) -> List[str]:
        """Return list of error types"""
        return ["ψ", "ψ_x", "ψ_y", "ψ_z", "ψ_xx", "ψ_yy", "ψ_zz", "ψ_xxx", "ψ_yyy", "ψ_zzz"]
    
    def visualize_solution(self, grid, function_name, numerical, exact, errors, prefix="",
                          save=True, show=False, dpi=180, isosurface_levels=None, num_isosurfaces=2):
        """
        Visualize 3D solution with isosurface display and analysis dashboard
        
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
            isosurface_levels: Optional custom levels for isosurfaces
            num_isosurfaces: Number of isosurface levels to display (default: 2)
            
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
        
        # Create 3x2 dashboard grid
        fig = plt.figure(figsize=(18, 16))
        gs = GridSpec(3, 2)
        
        # 1. 3D Isosurface visualization (top-left)
        ax1 = fig.add_subplot(gs[0, 0], projection='3d')
        self._plot_3d_isosurface(ax1, grid, psi, num_levels=num_isosurfaces, user_levels=isosurface_levels)
        
        # 2. Solution slices (top-right)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_solution_slices(ax2, grid, psi, mid_x, mid_y, mid_z)
        
        # 3. Error heatmap (middle-left)
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_error_heatmap(ax3, grid, error, mid_z)
        
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
    
    def _plot_3d_isosurface(self, ax, grid, data, num_levels=2, user_levels=None):
        """
        Visualize 3D isosurfaces of the solution
        
        Args:
            ax: matplotlib 3D Axes
            grid: Grid object
            data: 3D solution data
            num_levels: Number of isosurface levels to display (default: 2)
            user_levels: Optional custom levels provided by user
        """
        # Get grid coordinates
        x = self._to_numpy(grid.x)
        y = self._to_numpy(grid.y)
        z = self._to_numpy(grid.z)
        
        # Use user-provided levels if available
        if user_levels is not None:
            levels = user_levels
        else:
            # Compute value range for isosurface levels
            vmin, vmax = np.min(data), np.max(data)
            vrange = vmax - vmin
            
            # Define isosurface levels using percentiles to ensure meaningful values
            if abs(vrange) < 1e-10:  # Handle near-constant data
                levels = [vmin]
            else:
                # Use 25% and 75% percentiles for default 2 levels
                if num_levels == 2:
                    p_values = [25, 75]
                    percentiles = [np.percentile(data, p) for p in p_values]
                    
                    # Check if percentiles give distinct values
                    if len(set(percentiles)) < num_levels:
                        # Fallback to evenly spaced levels
                        levels = [vmin + vrange * 0.25, vmin + vrange * 0.75]
                    else:
                        levels = percentiles
                else:
                    # For other numbers of levels, distribute evenly across percentiles
                    p_values = np.linspace(25, 75, num_levels)
                    percentiles = [np.percentile(data, p) for p in p_values]
                    
                    # Check if percentiles give distinct values
                    if len(set(percentiles)) < num_levels:
                        # Fallback to evenly spaced levels
                        levels = [vmin + vrange * (i+1)/(num_levels+1) for i in range(num_levels)]
                    else:
                        levels = percentiles
                
                # Make sure levels have significant values (not too close to zero if overall data is large)
                if vmax > 1e-3:
                    levels = [max(level, vmin + vrange * 0.01) for level in levels]
                
                # Ensure levels are distinct enough and avoid exactly 0.0 as it often creates complex surfaces
                levels = [level if abs(level) > 1e-6 else 1e-6 * (1 if level >= 0 else -1) for level in levels]
                levels = sorted(set([round(level, 6) for level in levels]))
        
        # Setup 3D coordinate scaling for proper display
        x_min, x_max = x[0], x[-1]
        y_min, y_max = y[0], y[-1]
        z_min, z_max = z[0], z[-1]
        
        # Generate isosurfaces using scikit-image's marching cubes
        for i, level in enumerate(levels):
            try:
                # Generate isosurface using marching cubes
                verts, faces, _, _ = measure.marching_cubes(data, level)
                
                # Scale vertices to match the actual coordinate system
                verts_scaled = np.zeros_like(verts)
                verts_scaled[:, 0] = x_min + verts[:, 0] * (x_max - x_min) / (len(x) - 1)
                verts_scaled[:, 1] = y_min + verts[:, 1] * (y_max - y_min) / (len(y) - 1)
                verts_scaled[:, 2] = z_min + verts[:, 2] * (z_max - z_min) / (len(z) - 1)
                
                # Create color based on isosurface level
                color_val = (level - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                color = plt.cm.viridis(color_val)
                
                # Plot the isosurface
                ax.plot_trisurf(
                    verts_scaled[:, 0], verts_scaled[:, 1], verts_scaled[:, 2],
                    triangles=faces,
                    color=color,
                    alpha=0.3 + 0.2 * i,  # Increasing opacity for higher levels
                    shade=True
                )
            except Exception as e:
                # Handle error if isosurface generation fails
                print(f"Warning: Could not generate isosurface at level {level}: {e}")
                continue
        
        # Plot coordinate system reference
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
        mid_x = (x_max + x_min) / 2
        mid_y = (y_max + y_min) / 2
        mid_z = (z_max + z_min) / 2
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
        
        # Add reference plane at z = 0
        if z_min <= 0 <= z_max:
            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, 2),
                np.linspace(y_min, y_max, 2)
            )
            z_plane = np.zeros_like(xx)
            ax.plot_surface(xx, yy, z_plane, alpha=0.1, color='gray')
        
        # Settings
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"3D Isosurfaces")
        
        # Add a legend for isosurface levels
        import matplotlib.patches as mpatches
        handles = []
        for i, level in enumerate(levels):
            color_val = (level - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            color = plt.cm.viridis(color_val)
            patch = mpatches.Patch(color=color, alpha=0.5, label=f"Level: {level:.2f}")
            handles.append(patch)
        
        # Keep the legend outside the plot area for clarity
        ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.2, 1))
        
        # Set a good viewing angle
        ax.view_init(elev=30, azim=45)
    
    def _plot_solution_slices(self, ax, grid, data, mid_x, mid_y, mid_z):
        """Display solution slices"""
        # Extract the middle slice
        xy_slice = data[:, :, mid_z]
        
        # Grid coordinates
        x = self._to_numpy(grid.x)
        y = self._to_numpy(grid.y)
        
        # Display middle slice
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