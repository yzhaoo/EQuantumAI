import json
import numpy as np
import scipy.constants as sc
#import mathutils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import qbuilder as qbuilder
from geometry import Geometry3D
from sites import Site
import def_system_func as systemfunc
from collections import deque

# Assume Geometry3D and Site are defined elsewhere and imported.
# For example:
# from geometry3d_module import Geometry3D
# from site_module import Site

class System:
    def __init__(self, 
                 geometry_params, 
                 assign_mat=True, 
                 config_file=None,
                 ifqsystem=False,
                 t=1,
                 quantum_builder="default",
                 prune_vacuum=True,
                 vacuum_shells_to_keep=2):
        """
        Initialize the System by building the 3D geometry and assigning site properties.

        Parameters:
          - geometry_params: dict with parameters for Geometry3D, for example:
              {
                "lattice_type": square_lattice,   # or honeycomb_lattice, etc.
                "box_size": ((xmin, xmax), (ymin, ymax), (zmin, zmax)),
                "sampling_density_function": sampling_function,
                "quantum_center": (x0, y0, z0)     # optional, defaults to (0,0,0)
              }
          - config_file: path to a configuration file (e.g., JSON) exported from a 3D builder
                        that defines regions and their physical properties.
        """
        # Initialize Geometry3D
        self.geometry_params = geometry_params
        self.lat_spacing=geometry_params['sampling_density_function'](0.0)
        self.geometry = None
        self.build_geometry()
        # Discretize the simulation box
        self.geometry.discretize()
        print("Generated", len(self.geometry.points), "points in 3D.")
        
        need_topology = (config_file is not None) or assign_mat or ifqsystem

        if need_topology:
            self.geometry.compute_voronoi()
            print("Voronoi cells have been created.")
            self.sites = systemfunc.create_sites_from_geometry_3d(self.geometry)
            
        else:
            self.sites = systemfunc.create_sites_from_points_3d(self.geometry)
            
        self.num_sites=len(self.sites)
        self.material_indices={}
        self.Qsites=None
        self.N_indices=None
        self.D_indices=None
        if assign_mat is True:
            if config_file is not None:
                # Load configuration file defining regions and material properties.
                self.update_sites_from_blender(filename=config_file)
                self.remove_dangling_site()
            else:
                self.update_sites_from_func()
        
        # prune vacuum sites (safe: Blender guarantees naming)
        kept_ids, removed_ids = self.prune_vacuum_sites(shells_to_keep=2)
        self.apply_site_pruning(kept_ids)
        self.reindex_sites()
        self.num_sites = len(self.sites)
        self.update_mat_indices()

        print(f"Vacuum pruning: removed {len(removed_ids)} sites")

        #initialize quantum system
        #scale the hopping amplitude according to the discretization level
        if self.geometry_params["lattice_type"]=="square":
            #use m*=0.065m0
            #unit as eV
            self.t=sc.hbar**2/(2*(self.lat_spacing*1e-6)**2 *0.067*sc.m_e)/sc.elementary_charge
            self.lat_spacing_real=0.565 #nm
            self.unit_cell_area=(self.lat_spacing*1e-6)**2
            self.unit_cell_area_real=(self.lat_spacing_real*1e-9)**2
            self.max_fill=2/self.unit_cell_area/1e16 #maximal spinless carrier density, unit: # 10^12/cm^-2          
        elif self.geometry_params["lattice_type"]=="honeycomb":
            # use v_F=1e6 m/s from graphene
            #self.t=2*sc.hbar*1e6/(3*(self.lat_spacing*1e-6))/sc.elementary_charge
            self.t=0.4388/(self.lat_spacing*1000)
            self.lat_spacing_real=0.142 #nm
            self.unit_cell_area=(3*np.sqrt(3)*(self.lat_spacing*1e-6)**2 /2)
            self.unit_cell_area_real=(3*np.sqrt(3)*(self.lat_spacing_real*1e-9)**2 /2)
            self.max_fill=2/self.unit_cell_area/1e16 #maximal spinless carrier density, unit: # 10^12 /cm^-2
            
        else:
            self.t=t
            self.max_fill=1 #unitless calculation
            self.unit_cell_area=1
        self.qsystem=None
        self.qsite_map=None
        self.quantum_builder=quantum_builder
        if ifqsystem:
            if self.Qsites.size ==0:
                print("No site has been assigned to the quantum system.")
            else:
                self.build_qsystem(quantum_builder)
                print("Quantum system is generated using "+quantum_builder+".")
        print("EQsystem is successfully initialized.")

    def prune_vacuum_sites(self, shells_to_keep=2):
        all_ids = set(self.sites.keys())

        vacuum_ids = {
            sid for sid, site in self.sites.items()
            if site.material == "vacuum"
        }
        matter_ids = all_ids - vacuum_ids

        if len(vacuum_ids) == 0 or len(matter_ids) == 0:
            return all_ids, set()

        vacuum_dist = {}
        q = deque()

        # first shell
        for mid in matter_ids:
            for nid in self.sites[mid].neighbors:
                if nid in vacuum_ids and nid not in vacuum_dist:
                    vacuum_dist[nid] = 1
                    q.append(nid)

        # BFS outward
        while q:
            sid = q.popleft()
            d = vacuum_dist[sid]

            if d >= shells_to_keep:
                continue

            for nid in self.sites[sid].neighbors:
                if nid in vacuum_ids and nid not in vacuum_dist:
                    vacuum_dist[nid] = d + 1
                    q.append(nid)

        kept_vacuum = {
            sid for sid, d in vacuum_dist.items()
            if d <= shells_to_keep
        }

        kept_ids = set(matter_ids) | kept_vacuum
        removed_ids = all_ids - kept_ids

        return kept_ids, removed_ids

    def apply_site_pruning(self, kept_ids):
        """
        Remove sites not in kept_ids and clean neighbor dictionaries.
        """
        kept_ids = set(kept_ids)

        new_sites = {}
        for sid, site in self.sites.items():
            if sid not in kept_ids:
                continue
            new_sites[sid] = site

        # prune neighbor links
        for sid, site in new_sites.items():
            site.neighbors = {
                nid: val for nid, val in site.neighbors.items()
                if nid in kept_ids
            }

        self.sites = new_sites

    def reindex_sites(self):
        old_ids = sorted(self.sites.keys())
        old_to_new_map = {old_id: new_id for new_id, old_id in enumerate(old_ids)}

        new_site_dict = {}

        for old_id in old_ids:
            site = self.sites[old_id]
            new_id = old_to_new_map[old_id]

            site.id = new_id

            new_neighbors = {}
            for nn, val in site.neighbors.items():
                if nn in old_to_new_map:
                    new_neighbors[old_to_new_map[nn]] = val
            site.neighbors = new_neighbors

            new_site_dict[new_id] = site

        self.sites = new_site_dict

    def build_geometry(self):
        self.geometry= Geometry3D(lattice_type=self.geometry_params["lattice_type"],
                                   box_size=self.geometry_params["box_size"],
                                   sampling_density_function=self.geometry_params["sampling_density_function"],
                                   quantum_center=self.geometry_params.get("quantum_center", (0.0, 0.0, 0.0)))

    def remove_dangling_site(self):
        """
        Remove dangling sites (sites with no neighbors).
        """
        num_sites = len(self.sites)

        # find dangling site ids
        no_neighbor = [idx for idx, site in self.sites.items()
                    if site.neighbors == {}]

        if len(no_neighbor) == 0:
            print("0 sites have been removed from the system.")
            return

        # keep everything else
        kept_ids = set(self.sites.keys()) - set(no_neighbor)

        # apply pruning
        self.apply_site_pruning(kept_ids)

        # 🔥 now reindex in ONE place
        self.reindex_sites()

        print(f"{len(no_neighbor)} sites have been removed from the system.")

    def load_config(self, config_file):
        """
        Load the configuration file (assumed to be in JSON format) that defines regions.

        Parameters:
          - config_file: string, path to the configuration file.
        
        Returns:
          - config_data: the parsed JSON data.
        """
        if config_file==None:
            print("please provide configuration of the setup.")
            return None
        else:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            return config_data
    def update_mat_indices(self):
        # --- rebuild material classification ---
        self.initialize_material_indices()

        # --- Qsites: enforce deterministic, integer ordering ---
        self.Qsites = np.array(
            sorted(self.material_indices.get('Qsystem', [])),
            dtype=int
        )

        # --- mapping: global site_id → Q-index (VERY IMPORTANT) ---
        self.qsite_id_to_idx = {
            site_id: i for i, site_id in enumerate(self.Qsites)
        }

        # --- boundary condition partition ---
        self.N_indices = np.array(
            [i for i, site in self.sites.items() if site.BCtype == 1],
            dtype=int
        )

        self.D_indices = np.array(
            [i for i, site in self.sites.items() if site.BCtype == 0],
            dtype=int
        )

        # --- sanity checks (highly recommended) ---
        assert len(set(self.N_indices).intersection(self.D_indices)) == 0, \
            "N_indices and D_indices overlap!"

        assert len(self.N_indices) + len(self.D_indices) == len(self.sites), \
            "N_indices + D_indices != total number of sites"

        # optional: ensure Qsites are valid
        assert all(site_id in self.sites for site_id in self.Qsites), \
            "Qsites contain invalid site IDs"

    def initialize_material_indices(self):
        """
        Initialize indices for sites based on the material attribute.
        
        This method populates self.material_indices as a dictionary mapping material types
        (e.g., "gate", "dopants", "dielectric", "Qsystem") to a list of site indices.
        
        Returns:
        - material_indices: dict, keys are material names and values are lists of site indices.
        """
        # You can predefine the expected material types if desired.
        expected_materials = ["gate", "dopants", "dielectric", "Qsystem"]
        self.material_indices = {m: [] for m in expected_materials}

        for i, site in self.sites.items():
            mat = site.material if hasattr(site, "material") else None
            if mat is None:
                # Optionally, handle sites with no material attribute.
                continue
            if mat not in self.material_indices:
                self.material_indices[mat] = []
            self.material_indices[mat].append(i)
        return self.material_indices
    
    def build_qsystem(self,quantum_builder,**kwarg):
        self.qsystem=qbuilder.build_system(self,builder=quantum_builder,**kwarg)

    def plot_geometry(self, material=None, prop=None, ax=None, show=True):
        """
        Plot the discretized sites in 3D space.
        
        If 'prop' is None, sites are colored according to their material (discrete colors).
        Otherwise, 'prop' is expected to be a property name (e.g., "charge") and sites
        will be colored using a continuous colormap based on that property's value.
        """
        created_figure = ax is None
        if created_figure:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure
            ax.clear()
        if material is not None:
            coords=[]
            prop_values=[]
            for site in self.sites.values():
                if site.material==material:
                    value = getattr(site, prop, None)
                    coords.append(site.coordinates)
                    prop_values.append(value)
            coords=np.array(coords)
            prop_values=np.array(prop_values)
            cmap = cm.viridis
            sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                            c=prop_values, cmap=cmap, s=10)
            # Add a colorbar to indicate the property values.
            cbar = fig.colorbar(sc, ax=ax, pad=0.1)
            cbar.set_label(prop)
            
        elif prop is None:
            # Group sites by material.
            color_map = {
            'dielectric': (135/255, 213/255, 216/255, 0.1),
            'dopants':    (200/255, 151/255, 206/255, 0.6),
            'gate':       (200/255, 216/255, 14/255, 0.9),
            'Qsystem':    (44/255, 94/255, 84/255, 0.8),
            'vacuum':     (1, 1, 1, 0)}

        # Loop over each material key stored in self.material_indices.
            for mat in list(self.material_indices.keys()):
                # First, try an exact match:
                if mat in color_map:
                    my_color = color_map[mat]
                else:
                    # Convert the material name to lower case for case-insensitive matching.
                    lower_mat = mat.lower()
                    if 'dielectric' in lower_mat:
                        my_color = tuple(np.array(color_map['dielectric'])+0.03*np.random.rand(4))
                    elif 'dopants' in lower_mat:
                        my_color = tuple(np.array(color_map['dopants'])+0.03*np.random.rand(4))
                    elif 'gate' in lower_mat:
                        my_color = tuple(np.array(color_map['gate'])+0.03*np.random.rand(4))
                    else:
                        # Default color if none of the keywords match
                        my_color = (0, 0, 0, 1)
                
                coords = np.array([self.sites[idx].coordinates for idx in self.material_indices[mat]])
                if len(coords) > 0:
                    try:
                        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                                color=my_color, label=mat, s=10)
                    except ValueError:
                        print(my_color)
        else:
            # Color sites based on a continuous property, e.g., "charge".
            prop_values = []
            coords = []
            for site in self.sites.values():
                # Retrieve the value of the property.
                value = getattr(site, prop, None)
                if value is not None:
                    prop_values.append(value)
                    coords.append(site.coordinates)
            
            if len(coords) == 0:
                raise ValueError(f"No sites have the property '{prop}'.")
            
            coords = np.array(coords)
            prop_values = np.array(prop_values)
            
            # Create a normalization and a ScalarMappable for the colormap.
            norm = mcolors.Normalize(vmin=np.min(prop_values), vmax=np.max(prop_values))
            cmap = cm.viridis
            alphamap=np.abs(prop_values/(1.5*(np.max(prop_values)-np.min(prop_values))))
            sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                            c=prop_values, cmap=cmap, s=10)
            # Add a colorbar to indicate the property values.
            cbar = fig.colorbar(sc, ax=ax, pad=0.1)
            cbar.set_label(prop)
        box_size=self.geometry_params['box_size']
        ax.set_box_aspect((box_size[0][1]-box_size[0][0], box_size[1][1]-box_size[1][0], box_size[2][1]-box_size[2][0]))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("System Sites")
        ax.legend()
        if show:
            plt.show()
        return fig, ax

    def export_sites(self, filename="sites.json"):
        """
        Export the sites from the System as a JSON file.
        Each site is represented by its properties (id, coordinates, material, etc.).
        """
        data = []
        for site in self.sites.values():
            data.append({
                "id": site.id,
                "coordinates": site.coordinates.tolist(),  # convert np.array to list
                "material": site.material,
                "charge": site.charge,
                "potential": site.potential,
                "dielectric_constant": site.dielectric_constant,
                "BCtype": site.BCtype
            })
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

    def update_sites_from_blender(self, filename="updated_sites.json"):
        """
        Load updated site information from a JSON file and update the corresponding Site objects.
        
        The JSON file is expected to be a list of dictionaries where each dictionary represents
        a site with keys like "id", "coordinates", "material", "charge", "potential", "dielectric_constant", and "BCtype".
        """
        with open(filename, "r") as f:
            updated_data = json.load(f)
        
        if len(updated_data) != self.num_sites:
            print("The total number of sites of the current system doesn't match with the provided config file.")
        
        for entry in updated_data:
            site_id = entry.get("id")
            if site_id is None:
                continue  # Skip entries without an id.
            # Check if this site exists in the system.
            if site_id in self.sites:
                site = self.sites[site_id]
                # Update the site's properties. You can also update coordinates if needed.
                # Note: if the coordinates are stored as a mathutils.Vector, convert the list back.
                coords = entry.get("coordinates")
                if coords is not None:
                    site.coordinates = coords
                site.material = entry.get("material", site.material)
                site.charge = entry.get("charge", site.charge)
                site.potential = entry.get("potential", site.potential)
                site.dielectric_constant = entry.get("dielectric_constant", site.dielectric_constant)
                site.BCtype = entry.get("BCtype", site.BCtype)
            else:
                print(f"Warning: Site with id {site_id} not found in the system.")
        self.update_mat_indices()

    def update_sites_from_func(self):
        systemfunc.assign_point_to_dot(self.sites)
        for site in list(self.sites.values()):
            systemfunc.assign_point_to_material(site)

        self.update_mat_indices()

