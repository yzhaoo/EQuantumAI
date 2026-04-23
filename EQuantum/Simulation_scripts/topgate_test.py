import os ; import sys
sys.path.append(os.getcwd()+"/Equantum")
#path to data
datapath="/scratch/zhaoyuha/Datas/EQuantum_data/topgate/"
setuppath="setup/dotgate/"

from sites import Site
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from fsc import FSC
import scipy.linalg as sl
import importlib
import kwant
from EQsystem import System
def density_function(r):
        spacing0 = 0.008 # spacing at r=0
        k = 0.2    # spacing increases by 0.05 per unit distance
        return spacing0 + k * r

    # Define a 3D simulation box: ((xmin, xmax), (ymin, ymax), (zmin, zmax))

geoparams={"lattice_type": "honeycomb",   # or honeycomb_lattice, etc.
"box_size": ((-1, 1), (-1, 1), (-0.08, 0.08)),
"sampling_density_function": density_function,
"quantum_center": (0,0,0)     # optional, defaults to (0,0,0)
              }

config_file=setuppath+"0008_02_hc.json"
syst=System(geoparams,config_file=config_file,ifqsystem=True,quantum_builder="default")

qparams={'Ufunc': lambda x:0,'phi':0.025}
fsc=FSC(syst,ifinitial=False,params=qparams)

fsc.update_BC(syst,'gate','potential',0.9)
fsc.update_BC(syst,'backgate','potential',0.7,ifinitial=True)


fsc.solve(syst,save=datapath+"test0008_02")