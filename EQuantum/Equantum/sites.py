import numpy as np

class Site:
    def __init__(self, site_id, coordinates, charge=0.0, potential=0.0, dielectric_constant=1.0,boundary=False,BCtype=1):
        """
        Initialize a Site in 3D.
        
        Parameters:
        - site_id: int, unique identifier for the site.
        - coordinates: array-like, [x, y, z] coordinates.
        - charge: float, initial charge at the site.
        - potential: float, initial potential at the site.
        - dielectric_constant: float, local dielectric constant.
        """
        self.id = site_id
        self.coordinates = np.array(coordinates)
        self.charge = charge
        self.potential = potential
        self.dielectric_constant = dielectric_constant
        self.ifboundary=boundary
        self.BCtype=BCtype
        # Dictionary of neighbor site IDs mapped to the common face area (or interface information).
        self.neighbors = {}
        self.material="vacuum"
    def add_neighbor(self, neighbor_id, face_area=None):
        """
        Add a neighbor relationship.
        
        Parameters:
        - neighbor_id: int, ID of the neighbor site.
        - face_area: float or None, the area of the common face with the neighbor.
        """
        self.neighbors[neighbor_id] = face_area
    def compute_delta_contribution(self, neighbor_potential, face_area, distance):
        """
        Compute a contribution to the finite-volume Laplacian (or capacitance) matrix.
        
        This is an example formula; adjust as needed for your discretization.
        
        Parameters:
        - neighbor_potential: float, potential of the neighbor.
        - face_area: float, area of the interface between the sites.
        - distance: float, distance between the two site centers.
        
        Returns:
        - contribution: float.
        """
        # Example: proportional to the difference in potential scaled by interface area and inversely by distance.
        return face_area * (self.potential - neighbor_potential) / distance

    def __repr__(self):
        return (f"Site(id={self.id}, coord={self.coordinates}, "
                f"potential={self.potential}, charge={self.charge}, "
                f"material={self.material}, BCtype={self.BCtype}.)")
