class Surface:
    def __init__(self, site_id1, site_id2, area, distance, dielectric_constant, material_interface=False):
        """
        Initialize a Surface object representing the interface between two sites.

        Parameters:
        - site_id1: int, identifier for the first site.
        - site_id2: int, identifier for the second site.
        - area: float, the area of the interface.
        - distance: float, the distance between the two sites.
        - dielectric_constant: float, the averaged dielectric constant across the interface.
        - material_interface: bool, True if the surface separates different materials,
                              False if both sites belong to the same material.
        """
        self.site_ids = (site_id1, site_id2)
        self.area = area
        self.distance = distance
        self.dielectric_constant = dielectric_constant
        self.material_interface = material_interface

    def __repr__(self):
        return (f"Surface(sites={self.site_ids}, area={self.area:.3f}, "
                f"distance={self.distance:.3f}, dielectric={self.dielectric_constant:.3f}, "
                f"material_interface={self.material_interface})")