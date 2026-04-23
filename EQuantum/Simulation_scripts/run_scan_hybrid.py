import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, "Equantum"))

from fsc import FSC
from EQsystem import System

from scan_helpers_hybrid import write_scan_manifest, run_scan

pid = os.getpid()

print(f"[MAIN] PID = {pid}", flush=True)

    
def density_function(z):
    spacing0 = 0.011
    k = 0.2
    if abs(z) < 3 * spacing0:
        return spacing0
    return spacing0 + k * z


def main():
    geoparams={"lattice_type": "square",   # or honeycomb_lattice, etc.
    "box_size": ((-0.6, 0.6), (-0.6, 0.6), (-0.08, 0.08)),
    "sampling_density_function": density_function,
    "quantum_center": (0,0,0)     # optional, defaults to (0,0,0)
                }

    setuppathupdate = "/scratch/zhaoyuha/Datas/EQuantum_data/dotgate_center/setup/setup_7652d749b55d933a16606b6f95fbda03"
    config_file = setuppathupdate + "/updated_sites_dot.json"
    out_folder = setuppathupdate + "/fsc_logs_scan"
    os.makedirs(out_folder, exist_ok=True)

    phis = np.linspace(0.08, 0.12, 20)
    Vbgs = np.linspace(1, 1.5, 20)

    fixed_params = {
        # Number of scan chunks / concurrent workers.
        "Ncore": 20,
        # Cores used inside one FSC.solve(). Usually 1 for chunked parameter scans.
        "solver_Ncore": 1,
        "gate_potential": -1,
        "dielectric_constant": 4,
        "convergence_tol": [1e-3,1e-1],
        "ldos_method": "ED",
        "eta": 0.00015,
        "M": 256,
        "eps": 0.05,
        "kernel": "jackson",
        "save_ildos": True,
    }

    write_scan_manifest(
        out_folder=out_folder,
        density_function=density_function,
        geoparams=geoparams,
        phis=phis,
        Vbgs=Vbgs,
        config_file=config_file,
        fixed_params=fixed_params,
        script_path=__file__,
    )

    run_scan(
        FSC_cls=FSC,
        System_cls=System,
        geoparams=geoparams,
        config_file=config_file,
        phis=phis,
        Vbgs=Vbgs,
        out_folder=out_folder,
        fixed_params=fixed_params,
    )


if __name__ == "__main__":
    main()
