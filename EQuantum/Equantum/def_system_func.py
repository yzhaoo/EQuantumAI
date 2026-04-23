import numpy as np
from sites import Site
from surfaces import Surface
import matplotlib.path as mpltPath

def polygon_area(vertices):
    """
    Compute the area of a planar convex polygon in 3D.
    
    This function triangulates the polygon using the first vertex as a reference.
    
    Parameters:
    - vertices: np.ndarray of shape (N,3) representing the polygon vertices in order.
    
    Returns:
    - area: float, the computed area of the polygon.
    """
    if len(vertices) < 3:
        return 0.0
    v0 = vertices[0]
    area = 0.0
    for i in range(1, len(vertices) - 1):
        v1 = vertices[i]
        v2 = vertices[i + 1]
        triangle_area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        area += triangle_area
    return area

def create_sites_from_points_3d(geometry3d):
    """
    Create bare Site objects from geometry points only.
    No Voronoi tessellation, no neighbors.
    """
    points = geometry3d.points
    sites = {}

    for i, p in enumerate(points):
        sites[i] = Site(
            site_id=i,
            coordinates=np.array(p, dtype=float),
            charge=0.0,
            potential=0.0,
            dielectric_constant=1.0
        )

    return sites

def create_sites_from_geometry_3d(geometry3d):
    """
    Create Site objects from a Geometry3D instance.
    
    This function assumes that geometry3d.points is an array of shape (N, 3) and that
    geometry3d.compute_voronoi() has been called to obtain a 3D Voronoi tessellation.
    It then builds a dictionary of Site objects and populates neighbor relationships
    based on the Voronoi ridges.
    
    Parameters:
    - geometry3d: an instance of your Geometry3D class.
    
    Returns:
    - sites: dict mapping site IDs to Site objects.
    """
    # Get the points and the 3D Voronoi object.
    points = geometry3d.points  # shape (N, 3)
    vor = geometry3d.compute_voronoi()
    
    sites = {}
    num_points = len(points)
    for i in range(num_points):
        # Initialize each site with default values (modify as needed).
        sites[i] = Site(site_id=i, coordinates=points[i], charge=0.0, potential=0.0, dielectric_constant=1.0)
    
    # Use vor.ridge_points and vor.ridge_vertices to assign neighbor relationships.
    # Each ridge corresponds to a common face between two Voronoi cells.
    for ridge, ridge_vertices in zip(vor.ridge_points, vor.ridge_vertices):
        i, j = ridge
        # Skip infinite ridges (if any vertex is -1, it means the ridge is unbounded).
        if -1 in ridge_vertices:
            continue
        # Extract the vertices of the common face.
        face_vertices = vor.vertices[ridge_vertices]
        # Compute the area of the face.
        face_area = polygon_area(face_vertices)
        # # Compute the distance between the two sites.
        # distance = np.linalg.norm(sites[i].coordinates - sites[j].coordinates)
        # Add neighbor relationship for both sites.
        sites[i].add_neighbor(j, face_area)
        sites[j].add_neighbor(i, face_area)
        # if the site at the boundary
        if sites[i].dielectric_constant!=sites[j].dielectric_constant:
            sites[i].boundary=True
            sites[j].boundary=True

    
    return sites


def create_surfaces_from_geom3d(geom3d, sites):
    """
    Create a list of Surface objects from a Geometry3D instance and a dictionary of Site objects.
    
    Parameters:
    - geom3d: An instance of Geometry3D that has already been discretized and for which
              geom3d.compute_voronoi() has been called.
    - sites: A dictionary mapping site IDs to Site objects.
    
    Returns:
    - surfaces: A list of Surface objects representing interfaces between neighboring sites.
    """
    # Ensure that the Voronoi tessellation has been computed.
    vor = geom3d.voronoi
    surfaces = []
    
    # Loop over each ridge in the Voronoi tessellation.
    for ridge, ridge_vertices in zip(vor.ridge_points, vor.ridge_vertices):
        # Skip ridges that extend to infinity.
        if -1 in ridge_vertices:
            continue
        # Extract the vertices for this ridge.
        face_vertices = vor.vertices[ridge_vertices]
        # Compute the area of the common face using the helper polygon_area function.
        face_area = polygon_area(face_vertices)
        
        # Get the two site IDs.
        i, j = ridge
        # Compute the distance between the two site centers.
        distance = np.linalg.norm(sites[i].coordinates - sites[j].coordinates)
        # Average the dielectric constant between the two sites.
        avg_dielectric = 2/(1/sites[i].dielectric_constant + 1/ sites[j].dielectric_constant)
        # Label as a material interface if the dielectric constants differ.
        material_interface = (sites[i].dielectric_constant != sites[j].dielectric_constant)
        
        # Create a Surface object.
        surface = Surface(site_id1=i, site_id2=j, area=face_area, distance=distance,
                          dielectric_constant=avg_dielectric, material_interface=material_interface)
        surfaces.append(surface)
        
    return surfaces

# Example usage:
# Assume geom3d is an instance of Geometry3D (already discretized and with its Voronoi tessellation computed)
# and sites_dict is a dictionary of Site objects created via create_sites_from_geometry_3d(geom3d).
# surfaces_list = create_surfaces_from_geom3d(geom3d, sites_dict)
# for s in surfaces_list:
#     print(s)

def assign_point_to_dot(sites):
    coords=[[np.linalg.norm((site.coordinates[0],site.coordinates[1])),site.coordinates[2]]for site in list(sites.values())]
    polygon=[[0,0],[100,0],[110,30],[120,40],[130,40],[130,60],[0,60]]
    polygon=np.array(polygon)/1000
    
    path=mpltPath.Path(polygon)
    point_in=np.where(True==path.contains_points(coords))[0]
    for idx in point_in:
        sites[idx].material='gate'
        sites[idx].potential=0.3
        sites[idx].BCtype=0
        sites[idx].dielectric_constant=10

geo_params={"r_system":1,
"r_qsystem":0.99,
"top_dielectric":0.05,
"bot_dielectric":0.05,
"backgate":0.02}
def assign_point_to_material(site,params=geo_params):
    r_system=params['r_system']
    r_qsystem=params['r_qsystem']
    t_top_dielectric=params['top_dielectric']
    t_bot_dielectric=params['bot_dielectric']
    t_backgate=params['backgate']

    r=np.linalg.norm((site.coordinates[0],site.coordinates[1]))
    z=site.coordinates[2]

    if site.material != 'gate':
        if np.abs(z)<1e-4 and r<=r_qsystem:
            site.material='Qsystem'
            site.potential=0
            site.BCtype=0
            site.dielectric_constant=3/30
        elif -t_bot_dielectric-t_backgate<=z<-t_bot_dielectric and r<=r_system:
            site.material='backgate'
            site.potential=-2
            site.BCtype=0
            site.dielectric_constant=4
        elif (-t_bot_dielectric<=z<0 or 0<z<=t_top_dielectric) and r<r_system:
            site.material='dielectric'
            site.potential=0
            site.BCtype=1
            site.dielectric_constant=3.3



    



