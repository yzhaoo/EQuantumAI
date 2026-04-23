import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, "Equantum"))

from fsc import FSC
from EQsystem import System

from scan_helpers import write_scan_manifest, run_scan


def density_function(z):
    spacing0 = 0.011
    k = 0.2
    if abs(z) < 3 * spacing0:
        return spacing0
    return spacing0 + k * z


def main():
    geoparams = {
        "lattice_type": "square",
        "box_size": ((-0.6, 0.6), (-0.45, 0.45), (-0.08, 0.08)),
        "sampling_density_function": density_function,
        "quantum_center": (0, 0, 0),
    }

    setuppathupdate = "/scratch/zhaoyuha/Datas/EQuantum_data/squaregate_center/setup/setup_ae7d9fc59f6721d485f0e595c14ca989"
    config_file = setuppathupdate + "/updated_sites_square.json"
    out_folder = setuppathupdate + "/fsc_logs_scan_2"
    os.makedirs(out_folder, exist_ok=True)

    phis = np.linspace(0.08, 0.12, 50)
    Vbgs = np.linspace(1, 2.5, 50)

    fixed_params = {
        "gate_potential": -1,
        "dielectric_constant": 4,
        "Ncore": 30,
        "convergence_tol": 3e-2,
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

    syst = System(
        geoparams,
        config_file=config_file,
        ifqsystem=True,
        quantum_builder="default",
    )

    run_scan(
        FSC_cls=FSC,
        syst=syst,
        phis=phis,
        Vbgs=Vbgs,
        out_folder=out_folder,
        fixed_params=fixed_params,
    )


if __name__ == "__main__":
    main()