# self_consistent_solver.py
import poissonsolver as psolver
import qbuilder as qbuilder
import numpy as np
import solvers as solvers
import scipy.constants as sc
import scipy.sparse.linalg as spla
import time
from IPython.display import clear_output
import os

import time

import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from IPython.display import clear_output

class FSC:
    def __init__(self, system, ifinitial=True,qparams=None,convergence_tol=[1e-8,1e-6], max_iter=50,FL_pinning=True,Ncore=1, approx="ED"):
        """
        Initialize the self-consistent solver.

        Parameters:
          - system: an instance of your System class (which includes Geometry3D, Site objects, etc.)
          - quantum_solver: an instance (or module) that performs the quantum calculation (QAA update)
          - poisson_solver: an instance (or module) that performs the electrostatic/Poisson calculation (PAA update)
          - convergence_tol: tolerance for convergence.
          - max_iter: maximum number of iterations allowed.
        """
        #quantities
        #initialized from System
        self.geometry_params=system.geometry_params
        self.quantum_builder=system.quantum_builder
        self.sites = (system.sites).copy()
        self.num_sites=system.num_sites
        self.material_indices=system.material_indices
        self.ni=np.array([site.charge for i, site in self.sites.items()])
        self.Ui=np.array([site.potential for i, site in self.sites.items()])
        self.Ui_quantum = self.Ui.copy()#separat the active region and the pinning region
        self.N_indices = (system.N_indices).copy()
        self.D_indices = (system.D_indices).copy()
        self.t=system.t
        self.lattice_type=self.geometry_params['lattice_type']
        self.lat_spacing=system.lat_spacing
        self.unit_cell_area=system.unit_cell_area
        self.unit_cell_area_real=system.unit_cell_area_real
        self.max_fill=system.max_fill
        self.system = system
        self.energy_bounds = None
        # fixed global storage grid for LDOS
        # choose a range wide enough for physical charge integration,
        # but this grid is only for storage / integration, not KPM scaling
        self.E_global = np.linspace(-6 * self.t, 6 * self.t, 1024)
        
        #initialize with Poisson solver parameters
        self.Ci=None
        self.A_mixed=None
        self.F_input=None
        self.Delta_matrix= None

        #initialize with quantum solver parameters
        # initialize with quantum solver parameters
        self.qsystem = system.qsystem
        self.qparams = dict(self.qsystem.qparams)   # keep FSC and qsystem aligned
        self.ildos = None

        self.Qsites = np.array(system.Qsites, dtype=int)
        self.Qprime = self.Qsites.copy()
        self.Qsites_map={}
        self.qsite_id_to_idx = {
            site_id: i for i, site_id in enumerate(self.Qsites)
        }

        self.Qp_in_Q = {
            ii: self.qsite_id_to_idx[qpidx]
            for ii, qpidx in enumerate(self.Qprime)
        }
        #solver properties
        self.local_update_count=0
        self.Ncore=Ncore
        self.convergence_tol = convergence_tol
        self.max_iter = max_iter
        self.log = {
            "Qprime_len": [len(self.Qprime)],
            "Ui_maxdiff": [],
            "Ui_l2diff": [],
            "ni_maxdiff": [],
            "ni_l2diff": [],
            "ni_reldiff": [],
            "ildos_maxdiff": [],
            "ildos_l2diff":[],
            "ildos_meanreldiff":[],
            "timing_poisson": [],
            "timing_quantum": [],
            "timing_total": [],
        }
        self.snapshots = {}

        if ifinitial:
            if FL_pinning:
                solvers.Fermi_level_pinning(self)
            #initialize Posisson problem
            self.initial_Poisson()
        if qparams is not None:
            self.update_qparams(system,qparams,ifinitial=False)
            #update the maximal filling for graphene under magnetic field.
            emin, emax = self.get_energy_bounds()
            bandwidth = max(abs(emin), abs(emax))
            self.max_fill = system.max_fill # if self.lattice_type=="square" else self.E_to_n(bandwidth, self.qparams['phi'], self.lat_spacing*1e-6)
            #print(self.qparams['phi'],self.lat_spacing,self.bandwidth,self.max_fill,self.E_to_n(self.bandwidth,self.qparams['phi'],self.lat_spacing))
        #initialize Quantum problem
        self.initial_Quantum(system,approx=approx)

    def phi_to_B(self):
        return self.qparams['phi']*sc.h/sc.e/self.unit_cell_area

    def E_to_n(self,ee,phi,a):
        """
        traslate the energy (in the unit of the hopping amplitude) to the carrier density (in the unit of 10^12 cm^-2) according to the LL energy of graphene
        """
        #the carrier density will scale with the chosen lattice spacing. Here the present carrier densiyt is the one after scaling.
        return (4*ee**2/(3*np.pi)+4*phi/np.sqrt(3))/(3* a **2)/1e16

    def reset_log(self):
        self.log = {
            "Qprime_len": [len(self.Qprime)],
            "Ui_maxdiff": [],
            "Ui_l2diff": [],
            "ni_maxdiff": [],
            "ni_l2diff": [],
            "ni_reldiff": [],
            "ildos_maxdiff": [],
            "ildos_l2diff":[],
            "ildos_meanreldiff":[],
            "timing_poisson": [],
            "timing_quantum": [],
            "timing_total": [],
        }

    def get_energy_bounds(self, margin=0.05, method="fast"):
            H = self.qsystem.get_hamiltonian()

            if method == "fast":
                diag = H.diagonal().real
                H0_like = H.copy().tolil()
                H0_like.setdiag(0.0)
                H0_like = H0_like.tocsr()

                abs_rowsum = np.abs(H0_like).sum(axis=1).A.ravel()
                hop_bound = abs_rowsum.max()

                emin = diag.min() - hop_bound
                emax = diag.max() + hop_bound

            elif method == "gershgorin":
                abs_rowsum = np.abs(H).sum(axis=1).A.ravel()
                bound = abs_rowsum.max()
                emin, emax = -bound, bound

            elif method == "eigsh":
                try:
                    emax = spla.eigsh(H, k=1, which="LA", return_eigenvectors=False)[0]
                    emin = spla.eigsh(H, k=1, which="SA", return_eigenvectors=False)[0]
                except Exception:
                    abs_rowsum = np.abs(H).sum(axis=1).A.ravel()
                    bound = abs_rowsum.max()
                    emin, emax = -bound, bound
            else:
                raise ValueError("Unknown method")

            width = emax - emin
            pad = margin * width
            return emin - pad, emax + pad

    def update_max_fill(self, system):
        emin, emax = self.get_energy_bounds()
        bandwidth = max(abs(emin), abs(emax))

        if self.lattice_type == "square":
            self.max_fill = system.max_fill
        else:
            self.max_fill = self.E_to_n(
                bandwidth,
                self.qparams["phi"],
                self.lat_spacing * 1e-6
            )
            
    def initial_Poisson(self):
        """
        initialize the Poisson problem without Quantum system for the given boundary condition.
         
        """
        psolver.calculate_delta(self)
        #initialized Delta_matrix and A_mixed
        self.update_Poisson()
        self.Ci=psolver.solve_capacitance(self)
        print("The poisson problem has been initialized.")

    def update_Poisson(self):
        #solve the initial ND poisson problem and update ni, Ui
        pre_ni = self.ni.copy()
        pre_Ui = self.Ui.copy()

        UnnD = psolver.solve_NDpoisson(self)

        self.ni[self.D_indices] = UnnD[-len(self.D_indices):]
        self.Ui[self.N_indices] = UnnD[:len(self.N_indices)]

        dni = self.ni - pre_ni
        dUi = self.Ui - pre_Ui

        self.log["ni_maxdiff"].append(np.max(np.abs(dni)))
        self.log["ni_l2diff"].append(np.linalg.norm(dni)/np.sqrt(pre_ni.size))
        self.log["Ui_maxdiff"].append(np.max(np.abs(dUi)))
        self.log["Ui_l2diff"].append(np.linalg.norm(dUi)/np.sqrt(pre_Ui.size))
        rel = np.mean(np.abs(dni) / (pre_ni + 1e-30))
        self.log["ni_reldiff"].append(rel)
        

    def initial_Quantum(self, system, approx="ED", **kwarg):
        t0 = time.perf_counter()
        qbuilder.site_map(self, system)
        print("site_map:", time.perf_counter() - t0)

        t0 = time.perf_counter()
        qbuilder.update_U(self, system)
        print("update_U:", time.perf_counter() - t0)

        
        quantum_kwargs = {
            "approx": approx,
            **kwarg,
        }
        if approx == "ED":
            quantum_kwargs["eta"] =0.0003
        else:
            quantum_kwargs["M"]=256
            quantum_kwargs["eps"]=0.05
            quantum_kwargs["kernel"]="jackson"
        t0 = time.perf_counter()
        new_ildos = qbuilder.update_ildos(
            self,
            system,
            **quantum_kwargs
        )
        print("update_ildos:", time.perf_counter() - t0)

        t0 = time.perf_counter()
        nq = len(self.Qsites)
        nE = len(self.E_global)
        self.ildos = np.zeros((nq, 2, nE), dtype=float)
        self.ildos[:, 0, :] = self.E_global[None, :]
        print("alloc+fill:", time.perf_counter() - t0)

        t0 = time.perf_counter()
        q_active = self.qsystem.Qsite_ids
        for local_idx, site_id in enumerate(q_active):
            global_idx = self.qsite_id_to_idx[site_id]
            self.ildos[global_idx] = new_ildos[local_idx]
        print("copy back:", time.perf_counter() - t0)


    def update_qparams(self, system, qparams, ifinitial=True):
        # update FSC-side params
        self.qparams = {**self.qparams, **qparams}

        # update quantum system as well
        self.qsystem.update_qparams(qparams)

        if ifinitial:
            self.initial_Quantum(system)


    def update_Quantum(self, system, **kwarg):
        pre_ildos = self.ildos.copy()[:,1,:]

        qbuilder.update_U(self, system)
        new_ildos = qbuilder.update_ildos(self, system, **kwarg)

        # write active-region LDOS back into full global container
        for local_idx, site_id in enumerate(self.Qprime):
            global_idx = self.qsite_id_to_idx[site_id]
            self.ildos[global_idx] = new_ildos[local_idx]

        dildos = self.ildos[:,1,:] - pre_ildos
        self.log["ildos_maxdiff"].append(np.max(np.abs(dildos)))
        self.log["ildos_l2diff"].append(np.linalg.norm(dildos)/np.sqrt(pre_ildos.size))
        

        eps = 1e-12

        mask = np.abs(pre_ildos) > eps
        if np.any(mask):
            rel = np.abs(dildos[mask]) / np.abs(pre_ildos[mask])
            self.log["ildos_meanreldiff"].append(np.mean(rel))
        else:
            self.log["ildos_meanreldiff"].append(0.0)



    def local_solver(self):
        alpha = 1.0 if self.local_update_count == 0 else 0.4

        dUdn = solvers.local_solver(self, alpha=alpha)
        dUi_q = dUdn[0]
        dni_q = dUdn[1]

        print(np.linalg.norm(dUi_q), np.linalg.norm(dni_q))
        pre_ni=self.ni[self.Qprime].copy()
        self.Ui[self.Qprime] += dUi_q
        self.ni[self.Qprime] = pre_ni+dni_q

        self.log["Ui_maxdiff"].append(np.max(np.abs(dUi_q)))
        self.log["Ui_l2diff"].append(np.linalg.norm(dUi_q) / np.sqrt(dUi_q.size))

        self.log["ni_maxdiff"].append(np.max(np.abs(dni_q)))
        self.log["ni_l2diff"].append(np.linalg.norm(dni_q) / np.sqrt(dni_q.size))
        rel = np.mean(np.abs(dni_q) / (pre_ni + 1e-30))
        self.log["ni_reldiff"].append(rel)
        

        self.local_update_count += 1

    def update_BC(self, updates, ifinitial=False, FL_pinning=True):
        """
        Update boundary conditions for multiple materials and properties.

        Parameters
        ----------
        updates : dict
            Example:
            {
                'gate': {'potential': -0.3},
                'dielectric': {'dielectric_constant': 4},
                'backgate': {'potential': 2}
            }
        """

        # --- apply updates ---
        for site in self.sites.values():
            mat = site.material
            if mat in updates:
                for prop, value in updates[mat].items():
                    setattr(site, prop, value)

        # --- rebuild arrays ---
        self.ni = np.array([site.charge for site in self.sites.values()])
        self.Ui = np.array([site.potential for site in self.sites.values()])

        # --- reinitialize if needed ---
        if ifinitial:
            if FL_pinning:
                solvers.Fermi_level_pinning(self)
            self.initial_Poisson()


    def update_Qprime(self, tol=1e-7):
        Qprime_old = self.Qprime.copy()
        Qprime_new = solvers.update_Qprime(self, tol)

        self.Qprime = Qprime_new

        # 🔥 CRITICAL
        self.qsystem.update_active_sites(self.Qprime)

        if len(Qprime_new) != len(Qprime_old):
            psolver.calculate_delta(self)
            self.update_Poisson()
            self.Ci = psolver.solve_capacitance(self)

        self.Qp_in_Q = {
            ii: self.qsite_id_to_idx[qpidx]
            for ii, qpidx in enumerate(self.Qprime)
        }
        

    def solve(
        self,
        system,
        save=True,
        snapshot_every=5,
        ldos_method="TF",
        snapshot_mode="iteration",   # "iteration", "step", or "final_only"
        snapshot_folder="fsc_logs",
        max_total_iter=30,
        save_ildos=True,
        qprime_stable_by="set",      # "len" or "set"
        force_switch_after=5,
        **kwarg
    ):
        """
        Flowchart-style self-consistent loop:

        1. Update Q / Q' decomposition
        - if not stable, repeat Step I
        2. Update Poisson problem
        - must run at least once for each stable decomposition
        - if n[U] not converged, repeat Step II
        3. Update Quantum / ILDOS problem
        - must run at least once after Poisson is accepted
        - if ILDOS not converged, repeat Step III
        4. Stuck-solver safeguard
        - after force_switch_after repeated Step II/III calls, force the other one once
        5. finish when all are converged
        """

        def converged_nU():
            vals = self.log.get("ni_reldiff", [])
            return len(vals) > 0 and vals[-1] < self.convergence_tol[0]

        def converged_ildos():
            if len(self.log.get("ildos_meanreldiff", [])) > 0:
                return self.log["ildos_meanreldiff"][-1] < self.convergence_tol[1]

            vals = self.log.get("ni_reldiff", [])
            return len(vals) > 0 and vals[-1] < self.convergence_tol[0]

        def qprime_is_stable(before, after):
            if qprime_stable_by == "len":
                return len(before) == len(after)
            try:
                return np.array_equal(np.sort(np.asarray(before)), np.sort(np.asarray(after)))
            except Exception:
                return list(before) == list(after)

        def save_final(tag, it):
            if save:
                self.save_snapshot(
                    f"final_iter_{it:04d}_{tag}",
                    folder=snapshot_folder,
                    save_ildos=save_ildos
                )

        def build_quantum_kwargs():
            quantum_kwargs = {
                "approx": ldos_method,
                "Ncore": self.Ncore,
                **kwarg,
            }
            if ldos_method == "kmeanssample":
                quantum_kwargs["num_sample"] = int(5 * self.Ncore)
            return quantum_kwargs

        def register_solver(name):
            nonlocal last_solver, same_solver_count
            if last_solver == name:
                same_solver_count += 1
            else:
                last_solver = name
                same_solver_count = 1

        # save static metadata once
        if save:
            self.save_static_reference(snapshot_folder)

        save_intermediate = (snapshot_mode != "final_only")

        # initialize active region once
        self.update_Qprime()
        if len(self.Qprime) == 0:
            raise RuntimeError("Qprime is empty: the active quantum region vanished.")

        # iter counters: [Step I, Step II, Step III]
        iter_num = [0, 0, 0]

        # state flags for the flowchart logic
        poisson_ready_for_current_qprime = False
        quantum_ready_for_current_poisson = False

        # new: streak tracker for solver switching
        last_solver = None          # "poisson" or "quantum"
        same_solver_count = 0

        while True:
            it = sum(iter_num)
            t_iter0 = time.perf_counter()

            print("Iteration counts [Step I, Step II, Step III]:", iter_num)
            print(f"Solver streak: {last_solver}, {same_solver_count}")
            if hasattr(self, "print_iteration_summary"):
                self.print_iteration_summary(it)

            # ----------------------------
            # Iteration-level snapshot
            # ----------------------------
            if save and save_intermediate and snapshot_mode == "iteration":
                if snapshot_every is not None and snapshot_every > 0:
                    if it % snapshot_every == 0:
                        self.save_snapshot(
                            f"iter_{it:04d}",
                            folder=snapshot_folder,
                            save_ildos=save_ildos
                        )

            # =================================
            # local electrostatic update
            # =================================
            self.local_solver()

            if save and save_intermediate and snapshot_mode == "step":
                self.save_snapshot(
                    f"iter_{it:04d}_local",
                    folder=snapshot_folder,
                    save_ildos=save_ildos
                )

            # =================================
            # Step I: update Qprime partition
            # =================================
            qprime_before = np.asarray(self.Qprime).copy()
            self.update_Qprime()
            qprime_after = np.asarray(self.Qprime).copy()
            self.log.setdefault("Qprime_len", []).append(len(qprime_after))

            if not qprime_is_stable(qprime_before, qprime_after):
                psolver.calculate_delta(self)

                t0 = time.perf_counter()
                self.Ci = psolver.solve_capacitance(self)
                self.log.setdefault("timing_poisson", []).append(time.perf_counter() - t0)

                iter_num[0] += 1

                # Qprime changed => Poisson/Quantum states are no longer trusted
                poisson_ready_for_current_qprime = False
                quantum_ready_for_current_poisson = False

                # also reset streak, because decomposition changed
                last_solver = None
                same_solver_count = 0

                self.log.setdefault("timing_total", []).append(time.perf_counter() - t_iter0)

                if sum(iter_num) >= max_total_iter:
                    print("Reached maximum iteration count during Step I.")
                    save_final("max", it)
                    break

                continue

            # =================================
            # Forced switch safeguard
            # If one solver has run too many times in a row,
            # force the other solver once.
            # =================================
            if same_solver_count >= force_switch_after:
                if last_solver == "poisson":
                    t0 = time.perf_counter()
                    self.update_Quantum(system, **build_quantum_kwargs())
                    self.log.setdefault("timing_quantum", []).append(time.perf_counter() - t0)

                    if save and save_intermediate and snapshot_mode == "step":
                        self.save_snapshot(
                            f"iter_{it:04d}_stepIII_quantum_forced",
                            folder=snapshot_folder,
                            save_ildos=save_ildos
                        )

                    iter_num[2] += 1
                    quantum_ready_for_current_poisson = True
                    register_solver("quantum")

                    self.log.setdefault("timing_total", []).append(time.perf_counter() - t_iter0)

                    if sum(iter_num) >= max_total_iter:
                        print("Reached maximum iteration count during forced Quantum step.")
                        save_final("max", it)
                        break

                    continue

                elif last_solver == "quantum":
                    t0 = time.perf_counter()
                    self.update_Poisson()
                    self.log.setdefault("timing_poisson", []).append(time.perf_counter() - t0)

                    if save and save_intermediate and snapshot_mode == "step":
                        self.save_snapshot(
                            f"iter_{it:04d}_stepII_poisson_forced",
                            folder=snapshot_folder,
                            save_ildos=save_ildos
                        )

                    iter_num[1] += 1
                    poisson_ready_for_current_qprime = True

                    # new Poisson invalidates previous Quantum convergence
                    quantum_ready_for_current_poisson = False
                    register_solver("poisson")

                    self.log.setdefault("timing_total", []).append(time.perf_counter() - t_iter0)

                    if sum(iter_num) >= max_total_iter:
                        print("Reached maximum iteration count during forced Poisson step.")
                        save_final("max", it)
                        break

                    continue

            # =================================
            # Step II: update Poisson problem
            # =================================
            if (not poisson_ready_for_current_qprime) or (not converged_nU()):
                t0 = time.perf_counter()
                self.update_Poisson()
                self.log.setdefault("timing_poisson", []).append(time.perf_counter() - t0)

                if save and save_intermediate and snapshot_mode == "step":
                    tag = "stepII_poisson_init" if not poisson_ready_for_current_qprime else "stepII_poisson"
                    self.save_snapshot(
                        f"iter_{it:04d}_{tag}",
                        folder=snapshot_folder,
                        save_ildos=save_ildos
                    )

                iter_num[1] += 1
                poisson_ready_for_current_qprime = True

                # A new Poisson environment invalidates previous quantum convergence
                quantum_ready_for_current_poisson = False
                register_solver("poisson")

                self.log.setdefault("timing_total", []).append(time.perf_counter() - t_iter0)

                if sum(iter_num) >= max_total_iter:
                    print("Reached maximum iteration count during Step II.")
                    save_final("max", it)
                    break

                continue

            # =================================
            # Step III: update Quantum / ILDOS problem
            # =================================
            if (not quantum_ready_for_current_poisson) or (not converged_ildos()):
                t0 = time.perf_counter()
                self.update_Quantum(system, **build_quantum_kwargs())
                self.log.setdefault("timing_quantum", []).append(time.perf_counter() - t0)

                if save and save_intermediate and snapshot_mode == "step":
                    tag = "stepIII_quantum_init" if not quantum_ready_for_current_poisson else "stepIII_quantum"
                    self.save_snapshot(
                        f"iter_{it:04d}_{tag}",
                        folder=snapshot_folder,
                        save_ildos=save_ildos
                    )

                iter_num[2] += 1
                quantum_ready_for_current_poisson = True
                register_solver("quantum")

                self.log.setdefault("timing_total", []).append(time.perf_counter() - t_iter0)

                if sum(iter_num) >= max_total_iter:
                    print("Reached maximum iteration count during Step III.")
                    save_final("max", it)
                    break

                continue

            # =================================
            # All gates passed
            # =================================
            print("FSC converged: Q/Q' stable, n[U] converged, and ILDOS converged.")
            save_final("converged", it)
            break

    def save_Uini(self,Uis,nis,ildoss,filename):
        mdic={}
        mdic['Uis']=Uis
        mdic['nis']=nis
        mdic['ildoss']=ildoss
        
        mat_fname=filename
        savemat(mat_fname,mdic)


    def plot_full(self, prop_values,**kwarg):
        """
        Plot the discretized sites in 3D space.
        
        If 'prop' is None, sites are colored according to their material (discrete colors).
        Otherwise, 'prop' is expected to be a property name (e.g., "charge") and sites
        will be colored using a continuous colormap based on that property's value.
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Color sites based on a continuous property, e.g., "charge".
        coords = [site.coordinates for site in self.sites.values()]

        
        coords = np.array(coords)
        prop_values = np.array(prop_values)
        
        # Create a normalization and a ScalarMappable for the colormap.
        #norm = mcolors.Normalize(vmin=np.min(prop_values), vmax=np.max(prop_values))
        cmap = cm.viridis
        sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                        c=prop_values, cmap=cmap, s=20,vmin=np.min(prop_values), vmax=np.max(prop_values),**kwarg)
        # Add a colorbar to indicate the property values.
        cbar = fig.colorbar(sc, ax=ax, pad=0.1)
        #cbar.set_label(prop_values)
        box_size=self.geometry_params['box_size']
        ax.set_box_aspect((box_size[0][1]-box_size[0][0], box_size[1][1]-box_size[1][0], box_size[2][1]-box_size[2][0]))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("System Sites")
        ax.legend()
        plt.show()


    def plot_qsystem(self, prop_values, title="System Sites", **kwarg):
        """
        Plot the discretized sites in 2D quantum system.

        Parameters
        ----------
        prop_values : array-like
            Values defined on Qsites.
        title : str
            Figure title.
        **kwarg :
            Extra kwargs passed to ax.scatter().
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        coords = np.array([self.sites[idx].coordinates for idx in self.Qsites])
        prop_values = np.array(prop_values)

        cmap = cm.viridis
        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=prop_values,
            cmap=cmap,
            s=20,
            **kwarg
        )

        fig.colorbar(sc, ax=ax, pad=0.1)
        ax.axis("equal")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(title)
        plt.show()
    
    def plot_Hamiltonian(self, ax=None):
        """
        Plot Hamiltonian on the ACTIVE quantum system (Qprime).
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        # --- active Hamiltonian ---
        H = self.qsystem.get_hamiltonian().tocsr()

        # 🔥 CRITICAL: use ACTIVE site ordering
        q_ids = self.qsystem.Qsite_ids

        coords = np.array(
            [self.sites[sid].coordinates for sid in q_ids],
            dtype=float
        )
        xy = coords[:, :2]

        # --- onsite ---
        onsite = H.diagonal().real

        norm_sites = mcolors.Normalize(vmin=onsite.min(), vmax=onsite.max())
        cmap_sites = cm.viridis

        sc = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=onsite,
            cmap=cmap_sites,
            s=40,
            norm=norm_sites,
            zorder=3
        )

        cbar = plt.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("Onsite potential")

        # --- hoppings ---
        rows, cols = H.nonzero()

        mask = rows < cols
        rows = rows[mask]
        cols = cols[mask]

        hop = H[rows, cols].A1

        # 🔥 better: plot PHASE, not real part
        hop_phase = np.angle(hop)

        norm_hop = mcolors.Normalize(vmin=-np.pi, vmax=np.pi)
        cmap_hop = cm.twilight

        for i, j, phase in zip(rows, cols, hop_phase):

            x = [xy[i, 0], xy[j, 0]]
            y = [xy[i, 1], xy[j, 1]]

            ax.plot(
                x,
                y,
                color=cmap_hop(norm_hop(phase)),
                linewidth=1.5,
                alpha=0.9,
                zorder=1
            )

        sm = cm.ScalarMappable(norm=norm_hop, cmap=cmap_hop)
        sm.set_array([])
        cbar2 = plt.colorbar(sm, ax=ax, pad=0.08)
        cbar2.set_label("arg(Hopping)")

        ax.set_aspect('equal')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Active Quantum Hamiltonian (Qprime)")

        plt.show()

    def rebuild_active_qsystem(self):

        print("🔄 Updating active Hamiltonian (Qprime)")

        self.qsystem.update_active_sites(self.Qprime)

        # update mapping
        self.Qp_in_Q = {
            ii: self.qsite_id_to_idx[sid]
            for ii, sid in enumerate(self.Qprime)
        }
########debug log info############        
    def save_snapshot(self, tag, folder="fsc_logs", save_ildos=True):
        import os
        import numpy as np

        os.makedirs(folder, exist_ok=True)

        params = self.collect_runtime_params()
        #self.energy_bounds = self.qsystem.get_energy_bounds()

        payload = {
            "tag": tag,
            "Ui": self.Ui.copy(),
            "ni": self.ni.copy(),
            "Qprime": np.array(self.Qprime, dtype=int),
            "Qsites": np.array(self.Qsites, dtype=int),
            "Ci": np.array(self.Ci, dtype=float) if self.Ci is not None else np.array([]),
            "params": np.array([params], dtype=object),
            "log": np.array([self.log], dtype=object),
            "energy_bounds": np.array(self.energy_bounds, dtype=float) if self.energy_bounds is not None else np.array([]),
        }

        if save_ildos and self.ildos is not None:
            try:
                payload["ildos"] = np.asarray(self.ildos, dtype=float)
            except Exception:
                # fallback: object if some branch still returns non-uniform structure
                payload["ildos"] = np.array(self.ildos, dtype=object)

        np.savez(os.path.join(folder, f"{tag}.npz"), **payload)

    def print_iteration_summary(self, iter_num):
        print(f"\n--- Iteration {iter_num} ---")
        print(f"Qprime size   : {len(self.Qprime)}")

        if self.log.get("ni_maxdiff"):
            print(f"max |Δn|      : {self.log['ni_maxdiff'][-1]:.3e}")
        if self.log.get("ni_l2diff"):
            print(f"|Δn| per site  : {self.log['ni_l2diff'][-1]:.3e}")
        if self.log.get("ni_reldiff"):
            print(f"|Δn| per site % : {self.log['ni_reldiff'][-1]:.3e}")
        if self.log.get("ildos_maxdiff"):
            print(f"max |ΔILDOS|  : {self.log['ildos_maxdiff'][-1]:.3e}")
        if self.log.get("ildos_l2diff"):
            print(f"|ΔILDOS| per site  : {self.log['ildos_l2diff'][-1]:.3e}")
        if self.log.get("ildos_meanreldiff"):
            print(f"|ΔILDOS| per site % : {self.log['ildos_meanreldiff'][-1]:.3e}")

        if self.log.get("timing_poisson"):
            print(f"last Poisson  : {self.log['timing_poisson'][-1]:.3f} s")
        if self.log.get("timing_quantum"):
            print(f"last Quantum  : {self.log['timing_quantum'][-1]:.3f} s")

    def plot_convergence(self):

        fig, ax = plt.subplots(1, 3, figsize=(14, 4))

        ax[0].plot(self.log["Ui_maxdiff"])
        ax[0].set_yscale("log")
        ax[0].set_title("max |ΔU|")

        ax[1].plot(self.log["ni_maxdiff"])
        ax[1].set_yscale("log")
        ax[1].set_title("max |Δn|")

        ax[2].plot(self.log["Qprime_len"])
        ax[2].set_title("Qprime size")

        plt.tight_layout()
        plt.show()


    def collect_params(self):
        params = {}

        # magnetic field / phase
        if hasattr(self, "phi"):
            params["phi"] = self.phi

        # gate potentials
        gate_potentials = {}
        for idx, site in self.sites.items():
            if site.material in ["gate", "back_gate"]:
                gate_potentials[idx] = site.potential

        params["gate_potentials"] = gate_potentials

        # optional: global settings
        if hasattr(self, "geometry_params"):
            params["geometry_params"] = self.geometry_params

        if hasattr(self, "unit_cell_area"):
            params["unit_cell_area"] = self.unit_cell_area

        if hasattr(self, "max_fill"):
            params["max_fill"] = self.max_fill

        return params
    
    def save_static_reference(self, folder="fsc_logs"):
        import os
        import numpy as np

        os.makedirs(folder, exist_ok=True)

        coords_q = np.array(
            [self.sites[idx].coordinates[:2] for idx in self.Qsites],
            dtype=float
        )
        coords_all = np.array(
            [self.sites[idx].coordinates for idx in self.sites.keys()],
            dtype=float
        )

        site_ids_all = np.array(
            list(self.sites.keys()),
            dtype=int
        )

        materials_all = np.array(
            [getattr(self.sites[idx], "material", "unknown") for idx in self.sites.keys()],
            dtype=object
        )

        gate_info = {}
        for idx, site in self.sites.items():
            mat = getattr(site, "material", None)
            if mat in ["gate", "back_gate", "top_gate"]:
                gate_info[int(idx)] = {
                    "material": mat,
                    "potential": float(site.potential),
                    "coordinates": list(site.coordinates),
                }

        # sanitize geometry_params: replace callables with readable names
        geometry_params_clean = {}
        for k, v in self.geometry_params.items():
            if callable(v):
                geometry_params_clean[k] = f"<callable:{getattr(v, '__name__', 'anonymous')}>"
            else:
                geometry_params_clean[k] = v

        static_data = {
            "Qsites": np.array(self.Qsites, dtype=int),
            "coords_q": coords_q,
            "site_ids_all": site_ids_all,
            "coords_all": coords_all,
            "materials_all": materials_all,
            "geometry_params": geometry_params_clean,
            "unit_cell_area": float(self.unit_cell_area),
            "max_fill": float(self.max_fill),
            "gate_info": gate_info,
        }

        np.savez(
            os.path.join(folder, "run_static.npz"),
            static_data=np.array([static_data], dtype=object)
        )

    def collect_runtime_params(self):
        params = {}

        if hasattr(self, "qparams"):
            params["phi"] = self.qparams.get("phi", None)
        else:
            params["phi"] = None

        gate_potentials = {}
        for idx, site in self.sites.items():
            mat = getattr(site, "material", None)
            if mat in ["gate", "back_gate", "top_gate"]:
                gate_potentials[int(idx)] = float(site.potential)

        params["gate_potentials"] = gate_potentials
        return params