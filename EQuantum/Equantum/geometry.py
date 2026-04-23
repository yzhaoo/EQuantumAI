import numpy as np
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting


class Geometry3D:
    def __init__(self, lattice_type, box_size, sampling_density_function, quantum_center=(0.0, 0.0, 0.0)):
        """
        Initialize the 3D geometry.

        Parameters:
        - lattice_type: callable that generates an x-y lattice.
          It should have the signature: lattice_type(xy_box, spacing) -> np.ndarray of shape (N,2)
        - box_size: tuple of tuples ((xmin, xmax), (ymin, ymax), (zmin, zmax))
        - sampling_density_function: callable f(r) that returns the desired local spacing
          as a function of distance (r) from the quantum system center.
        - quantum_center: tuple (x_center, y_center, z_center) indicating the center of the quantum system.
        """

        set_lattice(self,lattice_type)
        self.box_size = box_size
        self.sampling_density_function = sampling_density_function
        self.quantum_center = np.array(quantum_center)
        self.points = None
        self.voronoi = None

    def discretize(self):
        """
        Discretize the 3D simulation box by generating a 2D lattice in the x-y plane for each z-layer.
        For each layer, the effective x-y spacing is computed via the sampling_density_function,
        and then the provided lattice_type callable generates the lattice points.
        
        Returns:
        - points: np.ndarray of shape (N, 3) with the coordinates of the sites.
        """
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = self.box_size
        
        # Determine the base spacing for the z-direction.
        # Here, we use the sampling density at z = quantum_center[2] as the minimum spacing.
        base_spacing = self.sampling_density_function(0.0)
        z_candidates = np.concatenate((np.arange(zmin, 0, base_spacing),np.arange(0,zmax,base_spacing)))
        
        all_points = []
        xy_box = ((xmin, xmax), (ymin, ymax))
        for z in z_candidates:
            # Compute the effective spacing for the x-y lattice at this z level.
            # Here we compute the offset in z from the quantum center.
            dz = abs(z - self.quantum_center[2])
            # Optionally, one could combine dz with an x-y offset;
            # here we simply use dz to modulate the spacing.
            effective_spacing = self.sampling_density_function(dz)
            # Generate the 2D lattice for the current layer.
            xy_points = self.lattice_type(xy_box, effective_spacing)
            # Add the z coordinate to each x-y point.
            for pt in xy_points:
                all_points.append([pt[0], pt[1], z])
        
        self.points = np.array(all_points)
        return self.points

    def compute_voronoi(self):
        if self.points is None:
            raise ValueError("No points generated. Call discretize() first.")

        if self.voronoi is None:
            self.voronoi = Voronoi(self.points)

        return self.voronoi

    def plot_geometry(self):
        """
        Plot the discretized 3D points.
        """
        if self.points is None:
            raise ValueError("No points generated. Call discretize() first.")
        
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2], c='b', s=10, label='Sites')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('3D Discretized Geometry')
        ax.legend()
        plt.show()

def set_lattice(self,lattice_type):
    if type(lattice_type)==str:
        if lattice_type=="square":
            self.lattice_type = square_lattice
        elif lattice_type=="honeycomb":
            self.lattice_type = honeycomb_lattice
        else:
            print("the input lattice type is not pre-defined, please define it first.")
    elif callable(lattice_type):
        self.lattice_type=lattice_type
    else:
        print("please define the lattice type.")


def square_lattice(xy_box, spacing):
    """
    Generate a square lattice in the x-y plane.

    Parameters:
    - xy_box: tuple ((xmin, xmax), (ymin, ymax))
    - spacing: desired spacing between points

    Returns:
    - points: np.ndarray of shape (N, 2) containing the lattice points.
    """
    (xmin, xmax), (ymin, ymax) = xy_box
    xs = np.arange(xmin, xmax, spacing)
    ys = np.arange(ymin, ymax, spacing)
    xv, yv = np.meshgrid(xs, ys, indexing='ij')
    points = np.vstack([xv.ravel(), yv.ravel()]).T
    return points

def honeycomb_lattice(xy_box, spacing):
    """
    Generate a honeycomb lattice (2D hexagonal lattice with two sublattices)
    inside the given x-y box.

    Parameters:
    - xy_box: tuple of tuples ((xmin, xmax), (ymin, ymax))
    - spacing: float, the nearest-neighbor distance between points

    Returns:
    - points: np.ndarray of shape (N, 2) containing the lattice points in the x-y plane.
    """
    (xmin, xmax), (ymin, ymax) = xy_box
    
    # Define the underlying triangular lattice vectors.
    # Here, we choose:
    # a1 = (1.5 * spacing,  sqrt(3)/2 * spacing)
    # a2 = (1.5 * spacing, -sqrt(3)/2 * spacing)
    a1 = np.array([1.5 * spacing, np.sqrt(3)/2 * spacing])
    a2 = np.array([1.5 * spacing, -np.sqrt(3)/2 * spacing])
    
    # Define the two basis vectors for the honeycomb structure.
    # The two sublattices (A and B) are given by:
    r_A = np.array([0.0, 0.0])
    r_B = np.array([spacing, 0.0])
    
    # Estimate the required range of indices m and n.
    # These are chosen to cover the xy_box.
    N_m = int((xmax - xmin) / (1.5 * spacing)) + 3
    N_n = int((ymax - ymin) / ((np.sqrt(3)/2) * spacing)) + 3
    
    points = []
    for m in range(-N_m, N_m + 1):
        for n in range(-N_n, N_n + 1):
            # Compute the lattice point for sublattice A.
            R_A = m * a1 + n * a2 + r_A
            # Compute the lattice point for sublattice B.
            R_B = m * a1 + n * a2 + r_B
            # Add points that lie inside the xy_box.
            if (xmin <= R_A[0] <= xmax) and (ymin <= R_A[1] <= ymax):
                points.append(R_A)
            if (xmin <= R_B[0] <= xmax) and (ymin <= R_B[1] <= ymax):
                points.append(R_B)
                
    points = np.array(points)
    ### recenter the points
    points = points-(np.mean(points[:,0]),np.mean(points[:,1]))
    return points