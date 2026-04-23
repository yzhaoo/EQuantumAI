# EQuantum

EQuantum is a Python-based simulation tool aimed at solving self-consistent problems in quantum systems, specifically focusing on the interplay between local charge density and electrostatic potential governed by the Poisson equation.

This project is developed based on the paper [The self-consistent quantum-electrostatic problem in strongly non-linear regime](https://scipost.org/SciPostPhys.7.3.031)

## Overview

The project provides a workflow to simulate quantum devices by:
1.  **Defining the Geometry:** Creating a 3D model of the device (e.g., using Blender or analytical functions) and discretizing the space.
2.  **Assigning Properties:** Setting material properties such as potential, charge, dielectric constant, and boundary conditions.
3.  **Quantum & Poisson Solvers:** Coupling a tight-binding quantum solver with a Poisson solver to find self-consistent solutions for charge density and potential.

## Workflow

The typical simulation workflow, as illustrated in `dotgate.ipynb`, involves the following steps:

### 1. Define Sandbox & Geometry
- Define the simulation volume (length, width, height).
- Discretize the space.
- Construct the device model, either by importing from Blender (using ray tracing to assign points) or defining shapes via functions.

### 2. Material Properties
Assign physical properties to the defined geometry:
- Potential
- Charge
- Dielectric constant
- Boundary types (Neumann/Dirichlet)
- Identification of quantum system regions

### 3. Initialize Quantum System
- Build a tight-binding Hamiltonian based on the points assigned to the quantum system.
- Define hopping terms and magnetic fields.
- Use the Kernel Polynomial Method (KPM) kernel (e.g., from `pyqula` or internal modules) to calculate the Local Density of States (LDOS).

### 4. Initialize Poisson Problem
- Set up the Poisson equation with appropriate boundary conditions (fixed potential, fixed charge, etc.).

### 5. Self-Consistent Loop
Iteratively solve the coupled problem until convergence:
1.  **Calculate Charge Density:** $n_i[\delta U] = \int^\mu dE \rho_i(E)f(E+\delta U_i)$, where $\rho_i$ is the LDOS.
2.  **Solve Poisson Equation:** Update the electrostatic potential given the new charge distribution.
3.  **Check Convergence:** Repeat until the potential and charge distribution stabilize.
    - *Note:* Handling depletion regions is critical; sites under contact (fully occupied) or depletion regions may require special treatment to avoid divergence.

## Key Modules & Dependencies

Based on the notebooks and scripts, the project relies on:
- **EQsystem:** For defining and initializing the system `System`.
- **fsc:** For self-consistent field calculations `FSC`.
- **poissonsolver:** For solving the Poisson equation.
- **qbuilder:** For building quantum operators and calculating ILDOS.
- **kwant:** Python package for quantum transport.
- **Standard Scientific Stack:** `numpy`, `scipy`, `matplotlib`, `tqdm`.

## Usage

See `dotgate.ipynb` for a complete example of setting up a top-gate simulation, initializing the system, and running the self-consistent loop.
