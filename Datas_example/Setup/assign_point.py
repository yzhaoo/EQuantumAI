import bpy
import argparse
import json
import mathutils
from mathutils.bvhtree import BVHTree
import bmesh
import os
import sys
# Priority logic prevents material intersections.
# Higher number = higher priority. If bounding volumes physically overlap, the highest priority wins.
MATERIAL_PRIORITY = {
    "qsystem": 100,
    "gate": 150,
    "dielectric": 10,
    "default": 0
}
def get_material_priority(mat_name):
    """Helper function to determine the priority of a material."""
    if not mat_name:
        return MATERIAL_PRIORITY["default"]
    mat_lower = mat_name.lower()
    if 'gate' in mat_lower:
        return MATERIAL_PRIORITY["gate"]
    if 'dielectric' in mat_lower:
        return MATERIAL_PRIORITY["dielectric"]
    if 'qsystem' in mat_lower:
        return MATERIAL_PRIORITY["qsystem"]
    return MATERIAL_PRIORITY.get(mat_name, MATERIAL_PRIORITY["default"])
def build_bvh_and_bbox(obj, depsgraph):
    """
    Builds a BVH tree and gets the bounding box for an object in world coordinates.
    Building this ONCE per object drastically optimizes the script compared to per-point building.
    """
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    for vert in bm.verts:
        vert.co = obj.matrix_world @ vert.co
        
    bvhtree = BVHTree.FromBMesh(bm)
    bm.free()
    eval_obj.to_mesh_clear()
    
    # Calculate exact world bounding box 
    coords = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    xs = [p.x for p in coords]
    ys = [p.y for p in coords]
    zs = [p.z for p in coords]
    bbox = {
        "xmin": min(xs), "xmax": max(xs),
        "ymin": min(ys), "ymax": max(ys),
        "zmin": min(zs), "zmax": max(zs)
    }
    return bvhtree, bbox
def point_in_bbox(point, bbox):
    """Adds a minor epsilon to account for numerical boundary edges."""
    eps = 1e-6
    return (bbox["xmin"] - eps <= point.x <= bbox["xmax"] + eps and
            bbox["ymin"] - eps <= point.y <= bbox["ymax"] + eps and
            bbox["zmin"] - eps <= point.z <= bbox["zmax"] + eps)
def is_point_inside_bvhtree(point, bvhtree):
    """
    Test if a point is inside the BVH Tree. 
    It includes an immediate geometric boundary check to fix ray-casting anomalies on flat surfaces.
    """
    # 1. First, check if the point is explicitly ON the mesh boundary
    # This prevents parity errors if the point sits perfectly on a flat face (e.g., z=0)
    nearest = bvhtree.find_nearest(point)
    if nearest and nearest[3] is not None and nearest[3] < 1e-6:
        return True
        
    # 2. Raycast from point to infinity (+z) checking intersection parity (odd = inside)
    direction = mathutils.Vector((0, 0, 1))
    count = 0
    test_point = point.copy()
    epsilon = 1e-6
    
    while True:
        result = bvhtree.ray_cast(test_point, direction)
        if result[0] is None:
            break
        hit_location, hit_normal, face_index, distance = result
        count += 1
        # Advance slightly past the intersection to trace the next volume exit
        test_point = hit_location + direction * epsilon
    return (count % 2) == 1
def update_site_properties(site, mat_name, mat_props, force_priority=False):
    """Updates the site only if the new material has a higher priority than the current one."""
    current_mat = site.get("material", "default")
    
    # In some logic flows, we might want to forcefully assign the property (like the 2D plane)
    if force_priority or get_material_priority(mat_name) > get_material_priority(current_mat):
        site["material"] = mat_name
        site["charge"] = mat_props.get("charge", site.get("charge", 0.0))
        site["potential"] = mat_props.get("potential", site.get("potential", 0.0))
        site["dielectric_constant"] = mat_props.get("dielectric_constant", site.get("dielectric_constant", 1))
        site["BCtype"] = mat_props.get("BCtype", site.get("BCtype", 0))
def import_sites(filename="sites.json"):
    with open(filename, "r") as f:
        sites_data = json.load(f)
    for site in sites_data:
        site["coordinates"] = mathutils.Vector(site["coordinates"])
    return sites_data
def export_sites(sites_data, filename="updated_sites.json"):
    export_data = []
    for site in sites_data:
        site_copy = site.copy()
        if isinstance(site_copy.get("coordinates"), mathutils.Vector):
            site_copy["coordinates"] = list(site_copy["coordinates"])
        export_data.append(site_copy)
    with open(filename, "w") as f:
        json.dump(export_data, f, indent=4)


def parse_cli_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Assign Blender material properties to EQuantum sites.")
    parser.add_argument("--sites", default=None, help="Path to input sites.json")
    parser.add_argument("--output", default=None, help="Path to output updated_sites.json")
    parser.add_argument("--setup-dir", default=None, help="Setup directory containing sites/config files")
    parser.add_argument("--profile", default=None, help="Optional profile name")
    parser.add_argument("--device-shape", default=None, help="Optional device shape name")
    parser.add_argument("--spacing0", default=None, help="Optional spacing0 metadata")
    parser.add_argument("--density-k", default=None, help="Optional density_k metadata")
    return parser.parse_args(argv)


def resolve_io_paths():
    args = parse_cli_args()
    filepath = bpy.data.filepath
    blend_directory = os.path.dirname(filepath) if filepath else ""

    if args.setup_dir:
        setup_dir = os.path.abspath(args.setup_dir)
    elif args.sites:
        setup_dir = os.path.dirname(os.path.abspath(args.sites))
    elif args.output:
        setup_dir = os.path.dirname(os.path.abspath(args.output))
    else:
        setup_dir = blend_directory

    if not setup_dir:
        print("Please save the blend file first or pass --setup-dir / --sites / --output.")
        return None, None

    sites_path = os.path.abspath(args.sites) if args.sites else os.path.join(setup_dir, "sites.json")
    output_path = os.path.abspath(args.output) if args.output else os.path.join(setup_dir, "updated_sites_dot.json")
    return sites_path, output_path


def main():
    sitefile, outputfile = resolve_io_paths()
    if not sitefile or not outputfile:
        return
    if not os.path.exists(sitefile):
        print(f"Cannot find {sitefile}")
        return
    depsgraph = bpy.context.evaluated_depsgraph_get()
    
    sites = import_sites(sitefile)
    print("Imported {} sites".format(len(sites)))
    
    # Ensure all objects data are constructed outside the loop for speed 
    # Store objects with their bvh trees and bboxes 
    parsed_objects = []
    qsystem_obj = None
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            if obj.name == "2dplane":
                qsystem_obj = obj
            else:
                bvhtree, bbox = build_bvh_and_bbox(obj, depsgraph)
                parsed_objects.append({
                    "obj": obj,
                    "bvhtree": bvhtree,
                    "bbox": bbox
                })
    # Pre-parse 2dPlane (Qsystem)
    qsystem_bvhtree = None
    qsystem_bbox = None
    if qsystem_obj is not None:
        qsystem_bvhtree, qsystem_bbox = build_bvh_and_bbox(qsystem_obj, depsgraph)
    # Main iterative loop over sites
    for site in sites:
        point = site["coordinates"]
        
        # 1. Evaluate Qsystem exclusively at strictly z ~ 0
        is_z_zero = abs(point.z) < 1e-6
        if is_z_zero and qsystem_obj is not None:
            if point_in_bbox(point, qsystem_bbox) and is_point_inside_bvhtree(point, qsystem_bvhtree):
                # The 2dplane is always Qsystem
                qsystem_props = {"charge": 0.0,"potential": 0.0,"dielectric_constant": 3.0,"BCtype": 0}
                update_site_properties(site, "Qsystem", qsystem_props, force_priority=True)
                
        # 2. Evaluate other meshes 
        for parsed in parsed_objects:
            obj = parsed["obj"]
            bbox = parsed["bbox"]
            bvhtree = parsed["bvhtree"]
            
            # Use bounded box filtering first (extremely fast compared to BVH)
            if point_in_bbox(point, bbox):
                if is_point_inside_bvhtree(point, bvhtree):
                    if obj.data.materials:
                        mat = obj.data.materials[0]
                        update_site_properties(site, mat.name, mat)
    os.makedirs(os.path.dirname(outputfile), exist_ok=True)
    export_sites(sites, outputfile)
    print(f"Updated site information exported to {outputfile}")
if __name__ == "__main__":
    main()
