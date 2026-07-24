import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Force JAX to use 64-bit precision for high-fidelity finite element analysis
from jax import config
config.update("jax_enable_x64", True)

# Import the linear elasticity core directly from your solver file
from folax_jax_solver import solve_linear_elasticity, compute_element_centroid_stress

# ==============================================================================
# 1. MESH & GEOMETRY GENERATORS
# ==============================================================================

def generate_true_cook_mesh(nx, ny):
    """
    Generates the classical tapered, skewed trapezoidal mesh for Cook's Membrane.
    Left edge:  X = 0,  Y spans from 0 to 44 (Clamped Wall)
    Right edge: X = 48, Y spans from 16 to 32 (Shear Loaded Tip)
    """
    xi = jnp.linspace(0.0, 1.0, nx + 1)
    eta = jnp.linspace(0.0, 1.0, ny + 1)
    XI, ETA = jnp.meshgrid(xi, eta, indexing="xy")
    
    # Geometric mapping formulas for true Cook's Membrane profile
    X = 48.0 * XI
    y_bottom = 16.0 * XI
    y_top = 44.0 - 12.0 * XI
    Y = y_bottom + ETA * (y_top - y_bottom)
    
    nodes = jnp.column_stack([X.ravel(), Y.ravel()])
    
    elements = []
    for j in range(ny):
        for i in range(nx):
            n0 = i + j * (nx + 1)
            n1 = n0 + 1
            n2 = n1 + (nx + 1)
            n3 = n0 + (nx + 1)
            elements.append([n0, n1, n2, n3])
            
    return nodes, jnp.array(elements, dtype=jnp.int32)

def generate_unit_square_mesh(nx, ny):
    """Generates a standard regular 1x1 square mesh strictly required for MMS validation."""
    x = jnp.linspace(0.0, 1.0, nx + 1)
    y = jnp.linspace(0.0, 1.0, ny + 1)
    X, Y = jnp.meshgrid(x, y, indexing="xy")
    nodes = jnp.column_stack([X.ravel(), Y.ravel()])

    elements = []
    for j in range(ny):
        for i in range(nx):
            n0 = i + j * (nx + 1)
            n1 = n0 + 1
            n2 = n1 + (nx + 1)
            n3 = n0 + (nx + 1)
            elements.append([n0, n1, n2, n3])
    return nodes, jnp.array(elements, dtype=jnp.int32)

# ==============================================================================
# 2. BOUNDARY CONDITION BUILDERS (NUMERICALLY SAFE)
# ==============================================================================

def build_cook_boundary_conditions(nodes):
    """Safely clamps the left vertical edge (X = 0.0) of the trapezoid."""
    left_edge = jnp.abs(nodes[:, 0] - 0.0) < 1e-5
    boundary_nodes = jnp.where(left_edge)[0]
    boundary_dofs = jnp.sort(jnp.concatenate([2 * boundary_nodes, 2 * boundary_nodes + 1]))
    all_dofs = jnp.arange(2 * nodes.shape[0], dtype=jnp.int32)
    free_dofs = jnp.setdiff1d(all_dofs, boundary_dofs)
    return free_dofs.astype(jnp.int32), boundary_dofs.astype(jnp.int32)

def build_mms_boundary_conditions(nodes):
    """Enforces Dirichlet conditions on all 4 boundaries of the 1x1 square for MMS."""
    tol = 1e-7
    boundary = (jnp.abs(nodes[:, 0] - 0.0) < tol) | (jnp.abs(nodes[:, 0] - 1.0) < tol) | \
               (jnp.abs(nodes[:, 1] - 0.0) < tol) | (jnp.abs(nodes[:, 1] - 1.0) < tol)
    boundary_nodes = jnp.where(boundary)[0]
    free_nodes = jnp.where(~boundary)[0]
    boundary_dofs = jnp.sort(jnp.concatenate([2 * boundary_nodes, 2 * boundary_nodes + 1]))
    free_dofs = jnp.sort(jnp.concatenate([2 * free_nodes, 2 * free_nodes + 1]))
    return free_dofs.astype(jnp.int32), boundary_dofs.astype(jnp.int32)

# ==============================================================================
# 3. FORCE ASSEMBLY ENGINES
# ==============================================================================

def assemble_cook_shear_traction(nodes, total_force_y=1.0):
    """
    Distributes a total vertical shearing force uniformly across the right edge (X = 48.0).
    The actual height of this edge is 16.0 units (from Y=16 to Y=32).
    """
    n_nodes = nodes.shape[0]
    f_ext = jnp.zeros((2 * n_nodes,), dtype=jnp.float64)
    
    right_edge = jnp.abs(nodes[:, 0] - 48.0) < 1e-5
    right_nodes = jnp.where(right_edge)[0]
    right_nodes = right_nodes[jnp.argsort(nodes[right_nodes, 1])] # Sort bottom-to-top

    # Traction intensity (Force per unit length)
    traction_y = total_force_y / 16.0 

    for i in range(right_nodes.shape[0] - 1):
        n1 = right_nodes[i]
        n2 = right_nodes[i + 1]
        edge_length = jnp.linalg.norm(nodes[n2] - nodes[n1])
        # Inject load strictly into the Y-degrees of freedom (2n + 1)
        f_ext = f_ext.at[2 * n1 + 1].add(0.5 * traction_y * edge_length)
        f_ext = f_ext.at[2 * n2 + 1].add(0.5 * traction_y * edge_length)
    return f_ext

def assemble_body_force(nodes, element_node_ids, f_body):
    """Assembles domain-wide constant body forces using 2x2 Gauss integration."""
    f_ext = jnp.zeros((2 * nodes.shape[0],), dtype=jnp.float64)
    g_pt = 1.0 / jnp.sqrt(3.0)
    gauss_pts = jnp.array([[-g_pt, -g_pt], [g_pt, -g_pt], [g_pt, g_pt], [-g_pt, g_pt]])
    
    for el in element_node_ids:
        coords = nodes[el]
        fe = jnp.zeros((8,), dtype=jnp.float64)
        for pt in gauss_pts:
            dN_dxi = 0.25 * jnp.array([
                [-(1 - pt[1]), 1 - pt[1], 1 + pt[1], -(1 + pt[1])],
                [-(1 - pt[0]), -(1 + pt[0]), 1 + pt[0], 1 - pt[0]]
            ])
            detJ = jnp.linalg.det(dN_dxi @ coords)
            N = 0.25 * jnp.array([
                (1.0 - pt[0]) * (1.0 - pt[1]),
                (1.0 + pt[0]) * (1.0 - pt[1]),
                (1.0 + pt[0]) * (1.0 + pt[1]),
                (1.0 - pt[0]) * (1.0 + pt[1])
            ])
            for a in range(4):
                fe = fe.at[2 * a:2 * a + 2].add(N[a] * f_body * detJ * 1.0)
        f_ext = f_ext.at[jnp.ravel(jnp.column_stack([2 * el, 2 * el + 1]))].add(fe)
    return f_ext

# ==============================================================================
# 4. METHOD OF MANUFACTURED SOLUTIONS (MMS) UTILITIES
# ==============================================================================

def exact_displacement(nodes, alpha=0.25, beta=0.15):
    return jnp.column_stack([alpha * nodes[:, 0]**2, beta * nodes[:, 1]**2])

def exact_stress(nodes, mu, lam, alpha=0.25, beta=0.15):
    eps_xx = 2.0 * alpha * nodes[:, 0]
    eps_yy = 2.0 * beta * nodes[:, 1]
    trace_e = eps_xx + eps_yy
    return jnp.column_stack([
        lam * trace_e + 2.0 * mu * eps_xx,
        lam * trace_e + 2.0 * mu * eps_yy,
        jnp.zeros_like(nodes[:, 0])
    ])

def body_force(mu, lam, alpha=0.25, beta=0.15):
    return jnp.array([-(2.0 * lam * alpha + 4.0 * mu * alpha), -(2.0 * lam * beta + 4.0 * mu * beta)], dtype=jnp.float64)

def compute_error(u_full, nodes, element_node_ids, mu, lam):
    u_exact = exact_displacement(nodes)
    disp_error = jnp.linalg.norm(u_full - u_exact) / jnp.linalg.norm(u_exact)

    element_coords = nodes[element_node_ids]
    element_displacements = u_full[element_node_ids]
    
    computed_stresses = jax.vmap(lambda c, u: compute_element_centroid_stress(c, u, mu, lam))(element_coords, element_displacements)
    centroids = jnp.mean(element_coords, axis=1)
    exact_sigma = exact_stress(centroids, mu, lam)
    
    stress_error = jnp.linalg.norm(computed_stresses - exact_sigma) / jnp.linalg.norm(exact_sigma)
    return float(disp_error), float(stress_error)

# ==============================================================================
# 5. PARAVIEW COMPATIBLE VTK EXPORTER
# ==============================================================================

def export_to_vtk(nodes, elements, u_full, mu, lam, filename="output.vtk"):
    nodes_np = np.array(nodes)
    elements_np = np.array(elements)
    u_full_np = np.array(u_full)
    
    num_nodes = nodes_np.shape[0]
    num_elements = elements_np.shape[0]
    
    elem_coords = nodes[elements]
    elem_u = u_full[elements]
    stresses = jax.vmap(lambda c, u: compute_element_centroid_stress(c, u, mu, lam))(elem_coords, elem_u)
    stresses_np = np.array(stresses)

    with open(filename, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("True Cook Membrane Grid Output\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n\n")
        
        f.write(f"POINTS {num_nodes} double\n")
        for pt in nodes_np:
            f.write(f"{pt[0]:.15e} {pt[1]:.15e} 0.0\n")
        f.write("\n")
        
        cell_size = num_elements * 5
        f.write(f"CELLS {num_elements} {cell_size}\n")
        for elem in elements_np:
            f.write(f"4 {elem[0]} {elem[1]} {elem[2]} {elem[3]}\n")
        f.write("\n")
        
        f.write(f"CELL_TYPES {num_elements}\n")
        for _ in range(num_elements):
            f.write("9\n")
        f.write("\n")
        
        f.write(f"POINT_DATA {num_nodes}\n")
        f.write("VECTORS displacement double\n")
        for u in u_full_np:
            f.write(f"{u[0]:.15e} {u[1]:.15e} 0.0\n")
        f.write("\n")
        
        f.write(f"CELL_DATA {num_elements}\n")
        f.write("SCALARS sigma_xx double 1\n")
        f.write("LOOKUP_TABLE default\n")
        for s in stresses_np:
            f.write(f"{s[0]:.15e}\n")
        f.write("\n")
        
        f.write("SCALARS sigma_yy double 1\n")
        f.write("LOOKUP_TABLE default\n")
        for s in stresses_np:
            f.write(f"{s[1]:.15e}\n")
        f.write("\n")
        
        f.write("SCALARS sigma_xy double 1\n")
        f.write("LOOKUP_TABLE default\n")
        for s in stresses_np:
            f.write(f"{s[2]:.15e}\n")
        f.write("\n")
        
        f.write("SCALARS von_mises double 1\n")
        f.write("LOOKUP_TABLE default\n")
        for s in stresses_np:
            vm = np.sqrt(s[0]**2 + s[1]**2 - s[0]*s[1] + 3.0 * s[2]**2)
            f.write(f"{vm:.15e}\n")
        f.write("\n")

# ==============================================================================
# 6. PIPELINE ORCHESTRATORS
# ==============================================================================

def run_cook_membrane(nx, ny, nu, formulation):
    """Runs the true tapered Cook's Membrane bending experiment under shear load."""
    # Activating the genuine tapered geometry
    nodes, elements = generate_true_cook_mesh(nx, ny)
    
    # Material tuning parameters
    E = 1.0
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    free_dofs, boundary_dofs = build_cook_boundary_conditions(nodes)
    u_boundary = jnp.zeros(boundary_dofs.shape, dtype=jnp.float64)
    
    # Enforcing a total vertical shear force of 1.0 distributed across the slanted tip
    f_ext = assemble_cook_shear_traction(nodes, total_force_y=10.0)

    u_full, _ = solve_linear_elasticity(u_boundary, boundary_dofs, free_dofs,
                                        elements, nodes, mu, lam, f_ext,
                                        formulation=formulation)
    
    # Exporting ParaView visual assets at the maximum mesh density (32x32)
    if nx == 32:
        export_to_vtk(nodes, elements, u_full, mu, lam, filename=f"cook_membrane_{formulation}.vtk")
        
    # Track the vertical displacement exactly at the top-right corner tip (48.0, 32.0)
    tip_idx = jnp.argmin(jnp.linalg.norm(nodes - jnp.array([48.0, 32.0]), axis=1))
    return float(u_full[tip_idx, 1])

def run_error_refinement(nx, ny, nu, formulation):
    """Runs the MMS precision analysis strictly on the 1x1 flat domain."""
    nodes, elements = generate_unit_square_mesh(nx, ny)
    E = 1.0
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    free_dofs, boundary_dofs = build_mms_boundary_conditions(nodes)
    u_boundary = exact_displacement(nodes).reshape(-1)[boundary_dofs]
    f_ext = assemble_body_force(nodes, elements, body_force(mu, lam))

    u_full, _ = solve_linear_elasticity(u_boundary, boundary_dofs, free_dofs,
                                        elements, nodes, mu, lam, f_ext,
                                        formulation=formulation)
    return compute_error(u_full, nodes, elements, mu, lam)

# ==============================================================================
# 7. MAIN RUNNER & GRAPHICS SUMMARY
# ==============================================================================

def estimate_convergence_rate(mesh_sizes, errors):
    mesh_sizes = np.asarray(mesh_sizes, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    positive = errors > 0.0
    if np.count_nonzero(positive) < 2:
        return np.nan
    x = np.log(mesh_sizes[positive])
    y = np.log(errors[positive])
    return float(np.polyfit(x, y, 1)[0])


def main():
    mesh_sizes = [4, 8, 16, 32]
    nu = 0.4999  # Absolute shear/volumetric locking stress limit
    methods = ["full", "sri", "bbar"]
    
    cook_results = {m: [] for m in methods}
    mms_disp_errors = {m: [] for m in methods}
    mms_stress_errors = {m: [] for m in methods}
    
    print("="*80)
    print("   GENUINE TAPERED COOK MEMBRANE & MMS STUDY SUITE FOR PRESENTATION")
    print("="*80)
    
    for method in methods:
        print(f"\nProcessing Formulation Scheme: {method.upper()}")
        for n in mesh_sizes:
            # Benchmark 1: True Trapezoidal Cook Membrane Tip Displacement
            tip_disp = run_cook_membrane(n, n, nu, method)
            cook_results[method].append(tip_disp)
            
            # Benchmark 2: Analytical L2 Errors via MMS
            d_err, s_err = run_error_refinement(n, n, nu, method)
            mms_disp_errors[method].append(d_err)
            mms_stress_errors[method].append(s_err)
            
            print(f"  Grid {n:2d}x{n:2d} | True Cook Tip Uy: {tip_disp:.5f} | MMS Disp Err: {d_err:.2e} | MMS Stress Err: {s_err:.2e}")

    colors = {'full': '#d9534f', 'sri': '#5cb85c', 'bbar': '#0275d8'}
    markers = {'full': 'o-', 'sri': 's--', 'bbar': '^:'}
    labels = {'full': 'Full (locked)', 'sri': 'SRI (locking-free)', 'bbar': 'B-bar (locking-free)'}

    plt.figure(figsize=(7.5, 5), dpi=120)
    for m in methods:
        plt.plot(mesh_sizes, cook_results[m], markers[m], color=colors[m], label=labels[m], linewidth=2)
    plt.title("Cook Membrane: Tip displacement vs mesh size", fontsize=11, fontweight='bold')
    plt.xlabel("Mesh size", fontsize=10)
    plt.ylabel("Tip vertical displacement $u_y$", fontsize=10)
    plt.xticks(mesh_sizes, [f"{n}x{n}" for n in mesh_sizes])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=9, loc='lower right')
    plt.tight_layout()
    plt.savefig("plot_1_tip_displacement_vs_mesh.png", bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(7.5, 5), dpi=120)
    for m in methods:
        slope = estimate_convergence_rate(mesh_sizes, mms_disp_errors[m])
        plt.loglog(mesh_sizes, mms_disp_errors[m], markers[m], color=colors[m],
                   label=f"{labels[m]} (slope ~ {slope:.2f})", linewidth=2)
    plt.title("Displacement error convergence vs mesh size", fontsize=11, fontweight='bold')
    plt.xlabel("Mesh size", fontsize=10)
    plt.ylabel("Relative displacement error", fontsize=10)
    plt.xticks(mesh_sizes, [f"{n}x{n}" for n in mesh_sizes])
    plt.grid(True, which='both', linestyle=":", alpha=0.5)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("plot_2_displacement_error_vs_mesh.png", bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(7.5, 5), dpi=120)
    for m in methods:
        slope = estimate_convergence_rate(mesh_sizes, mms_stress_errors[m])
        plt.loglog(mesh_sizes, mms_stress_errors[m], markers[m], color=colors[m],
                   label=f"{labels[m]} (slope ~ {slope:.2f})", linewidth=2)
    plt.title("Stress error convergence vs mesh size", fontsize=11, fontweight='bold')
    plt.xlabel("Mesh size", fontsize=10)
    plt.ylabel("Relative stress error", fontsize=10)
    plt.xticks(mesh_sizes, [f"{n}x{n}" for n in mesh_sizes])
    plt.grid(True, which='both', linestyle=":", alpha=0.5)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("plot_3_stress_error_vs_mesh.png", bbox_inches='tight')
    plt.close()

    print("\n" + "="*80)
    print("SUCCESS: Mesh-size convergence plots saved to disk.")
    print("True Cook Membrane VTK asset outputs generated for ParaView visualization.")
    print("="*80)

if __name__ == "__main__":
    main()