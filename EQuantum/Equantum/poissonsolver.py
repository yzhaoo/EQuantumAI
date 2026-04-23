import numpy as np
from scipy.sparse import lil_matrix, identity, hstack, vstack
from scipy.sparse.linalg import spsolve
#import mumps

# class PoissonSolver:
#     def __init__(fsc, sites):
#         """
#         Initialize the Poisson solver.

#         Parameters:
#         - sites: dict, mapping site IDs to Site objects.
#                  Each Site must have attributes:
#                    - coordinates (np.array),
#                    - charge,
#                    - potential (for Dirichlet sites, the prescribed value),
#                    - dielectric_constant,
#                    - BCtype ("n" for Neumann, "d" for Dirichlet),
#                    - neighbors (dict mapping neighbor site IDs -> face area)
#         """
#         fsc.sites = sites
#         fsc.num_sites = len(sites)
#         fsc.A_mixed = None  # The assembled mixed matrix.
#         fsc.F_input = None  # The input (RHS) vector in Eq. (20).
#         fsc.N_indices = []  # Indices for Neumann sites.
#         fsc.D_indices = []  # Indices for Dirichlet sites.


#         fsc.Delta_matrix= None # Matrices for discrete laplacian
#         fsc.calculate_delta() # Initialize Delta_matrix

    

def calculate_delta(fsc):
    unit_cell_area=fsc.unit_cell_area

    #change density unit to #/site
    Cunit=55.2634936/unit_cell_area/1e16 #1V voltage difference through 1mum**2 area seprated by distance 1mum will indice Cunit number of charge 10^12 cm^-2


    total = fsc.num_sites

    # Partition sites by boundary condition type.
    

    # Assemble the full finite-volume matrix Δ and full source vector b.
    A_full = lil_matrix((total, total))
    b_full = np.zeros(total)
    for i, site in fsc.sites.items():
        diag_val = 0.0
        for j, face_area in site.neighbors.items():
            neighbor = fsc.sites[j]
            d = np.linalg.norm(np.array(site.coordinates) - np.array(neighbor.coordinates))
            avg_dielectric = -2.0 /(1/site.dielectric_constant + 1/neighbor.dielectric_constant)
            coeff = avg_dielectric * face_area * Cunit / d
            diag_val += coeff
            A_full[i, j] = -coeff
        A_full[i, i] = diag_val

    A_full = A_full.tocsr()

    # Partition A_full into blocks corresponding to Neumann (N) and Dirichlet (D) sites.
    A_NN = A_full[fsc.N_indices, :][:, fsc.N_indices]
    A_ND = A_full[fsc.N_indices, :][:, fsc.D_indices]
    A_DN = A_full[fsc.D_indices, :][:, fsc.N_indices]
    A_DD = A_full[fsc.D_indices, :][:, fsc.D_indices]
    #update the delta matrix
    fsc.Delta_matrix=[A_NN,A_ND,A_DN,A_DD]

    # define A_mixed matrix
    zero_block = lil_matrix((len(fsc.N_indices), len(fsc.D_indices)))
    top_block = hstack([A_NN, zero_block])
    # For Dirichlet sites: row = [ Δ_DN   -I ]
    I_D = identity(len(fsc.D_indices), format='lil')
    bottom_block = hstack([A_DN, -I_D])
    #update the A_mixied
    fsc.A_mixed = vstack([top_block, bottom_block]).tocsr()

def assemble_input(fsc,n_N, U_D):
    [_,A_ND,_,A_DD]=fsc.Delta_matrix
    # Now, construct the input vector F_input.
    # For Neumann sites: F_top = n_N - Δ_ND * U_D,
    # where n_N are the source terms for Neumann sites and U_D are the prescribed potentials.
    A_ND = A_ND.tocsr()
    F_top = n_N - A_ND.dot(U_D)
    # For Dirichlet sites: F_bottom = -Δ_DD * U_D.
    A_DD = A_DD.tocsr()
    F_bottom = -A_DD.dot(U_D)

    fsc.F_input = np.concatenate([F_top, F_bottom])
    return fsc.F_input

def solve(A,F, solver="scipy"):
    if solver=="scipy":
        solver_func=scipy_solver
    elif solver=="mumps":
        solver_func=mumps_solver
    elif callable(solver):
        solver_func=solver
    else:
        print("please provide the solver_function")
    """
    Solve the system A_mixed * X = F_input using the provided solver function.

    Parameters:
    - solver_func: a function that accepts (A, F) and returns the solution X.

    Returns:
    - X: the solution vector containing [U_N; n_D].
    """
    return solver_func(A, F)

# def solve_capacitance(fsc,**kwargs):
#     N_sites_num=len(fsc.N_indices)
#     D_sites_num=len(fsc.D_indices)
#     n_N = np.zeros(N_sites_num)
#     U_D = np.zeros(fsc.num_sites)
    
#     common_indices_D = list(set(fsc.Qprime).intersection(fsc.D_indices))
#     where_Qp_in_D=[list(fsc.D_indices).index(x) for x in fsc.Qprime]

#     # for Nidx,idx in enumerate(common_indices_N):
#     #     n_N[Nidx]=fsc.sites[idx].charge
#     for idx in common_indices_D:
#         U_D[idx]= 1.
#     U_D=U_D[fsc.D_indices]
#     sol= solve(fsc.A_mixed,assemble_input(fsc,n_N,U_D),**kwargs)
#     Cij=sol[-D_sites_num:][np.array(where_Qp_in_D)]

#     return Cij
def solve_capacitance(fsc, **kwargs):
    N_sites_num = len(fsc.N_indices)
    D_sites_num = len(fsc.D_indices)

    if len(fsc.Qprime) == 0:
        raise RuntimeError("Qprime is empty: the active quantum region vanished.")

    missing = [x for x in fsc.Qprime if x not in fsc.D_indices]
    if missing:
        raise RuntimeError(
            "Qprime is not a subset of D_indices. "
            f"{len(missing)} sites missing, first few: {missing[:10]}"
        )

    n_N = np.zeros(N_sites_num)
    U_D = np.zeros(fsc.num_sites)

    D_list = list(fsc.D_indices)
    where_Qp_in_D = np.array([D_list.index(x) for x in fsc.Qprime], dtype=int)

    for idx in fsc.Qprime:
        U_D[idx] = 1.0

    U_D = U_D[fsc.D_indices]
    sol = solve(fsc.A_mixed, assemble_input(fsc, n_N, U_D), **kwargs)

    Cij = sol[-D_sites_num:][where_Qp_in_D]
    return Cij

def solve_NDpoisson(fsc,**kwargs):
    N_sites_num=len(fsc.N_indices)
    n_N = np.zeros(N_sites_num)
    for nNidx,idx in enumerate(fsc.N_indices):
        n_N[nNidx]=fsc.ni[idx]
    U_D = np.array([fsc.Ui[i] for i in fsc.D_indices])

    sol= solve(fsc.A_mixed,assemble_input(fsc,n_N,U_D),**kwargs)
    return sol




def mumps_solver(A, F):
    """
    Solve the sparse system A * X = F using MUMPS via the pymumps package.
    
    Parameters:
    - A: a scipy.sparse matrix in CSR (or converted to CSC) format.
    - F: the right-hand side vector.
    
    Returns:
    - X: the solution vector.
    """
    # Convert the matrix to CSC format, which is generally expected by MUMPS.
    
    # Create a MUMPS context for a real unsymmetric system.
    ctx = mumps.Context()
    if hasattr(ctx, "set_silent"):
        ctx.set_silent(True)
    ctx.set_centralized_system(A, rhs=F)
    
    # Job 6 corresponds to analysis, factorization, and solve.
    ctx.run(job=6)
    
    # Retrieve the solution.
    X = ctx.get_solution()
    
    # Clean up the MUMPS context.
    ctx.destroy()
    
    return X


def scipy_solver(A, F):
    """
    Solve the sparse system A * X = F using SciPy's spsolve.

    Parameters:
    - A: sparse matrix in CSR or CSC format.
    - F: right-hand side vector.

    Returns:
    - X: solution vector.
    """
    X = spsolve(A, F)
    return X