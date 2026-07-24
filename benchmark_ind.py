import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import config
config.update("jax_enable_x64", True)

from jax_sri_solver import assemble_global_stiffness, solve_fem_system, compute_B_matrix

def generate_block_mesh(L, H, nx, ny):
    """
    Generates a structured Q4 quadrilateral mesh for a rectangular block.
    Ordering is counter-clockwise for element nodes.
    """
    x = jnp.linspace(0.0, L, nx + 1, dtype=jnp.float64)
    y = jnp.linspace(0.0, H, ny + 1, dtype=jnp.float64)
    X, Y = jnp.meshgrid(x, y)
    nodes = jnp.stack([X.ravel(), Y.ravel()], axis=1)
    
    elements = []
    for j in range(ny):
        for i in range(nx):
            n1 = j * (nx + 1) + i
            n2 = n1 + 1
            n3 = n1 + (nx + 1) + 1
            n4 = n1 + (nx + 1)
            elements.append([n1, n2, n3, n4])
    return nodes, jnp.array(elements)

def setup_boundary_conditions(nodes, nx, ny, L):
    """
    Sets up fixed DOFs at the bottom and a consistent distributed traction
    at the top center of the block.
    """
    num_nodes = nodes.shape[0]
    F_global = jnp.zeros(2 * num_nodes, dtype=jnp.float64)
    
    # 1. Fully fixed constraints at the bottom boundary (Y = 0)
    bottom_node_indices = jnp.where(nodes[:, 1] == 0.0)[0]
    fixed_dofs = []
    for idx in bottom_node_indices:
        fixed_dofs.extend([int(2 * idx), int(2 * idx + 1)])
    fixed_dofs = jnp.array(fixed_dofs)
    
    # 2. Consistent distributed traction at top center (Y = H)
    top_node_indices = jnp.where(nodes[:, 1] == jnp.max(nodes[:, 1]))[0]
    x_min_load, x_max_load = 3.75, 6.25
    q_magnitude = -40.0  
    dx = L / nx
    
    for idx in top_node_indices:
        node_x = nodes[idx, 0]
        if node_x >= x_min_load and node_x <= x_max_load:
            if jnp.isclose(node_x, x_min_load) or jnp.isclose(node_x, x_max_load):
                nodal_force = 0.5 * q_magnitude * dx
            else:
                nodal_force = q_magnitude * dx
            F_global = F_global.at[2 * idx + 1].set(nodal_force)
            
    center_top_node = top_node_indices[len(top_node_indices) // 2]
    return fixed_dofs, F_global, center_top_node

def compute_element_von_mises(nodes, elements, U_global, E, nu):
    """
    Computes standard Von Mises stress for each element at its center.

    """
    from jax_sri_solver import get_standard_material_matrix
    E_mat = get_standard_material_matrix(E, nu)
    num_elements = elements.shape[0]
    vm_stresses = jnp.zeros(num_elements, dtype=jnp.float64)
    
    for e in range(num_elements):
        node_indices = elements[e]
        X_e = nodes[node_indices]
        dofs = jnp.array([
            2*node_indices[0], 2*node_indices[0]+1, 2*node_indices[1], 2*node_indices[1]+1,
            2*node_indices[2], 2*node_indices[2]+1, 2*node_indices[3], 2*node_indices[3]+1
        ])
        u_e = U_global[dofs]
        
        B, _ = compute_B_matrix(X_e, 0.0, 0.0)
        strain = jnp.dot(B, u_e)
        stress = jnp.dot(E_mat, strain)  # Direct multiplication with standard material matrix
        
        s11, s22, s12 = stress[0], stress[1], stress[2]
        s33 = nu * (s11 + s22)  # Out-of-plane stress components under Plane Strain assumptions
        
        vm = jnp.sqrt(0.5 * ((s11 - s22)**2 + (s22 - s33)**2 + (s33 - s11)**2 + 6.0 * s12**2))
        vm_stresses = vm_stresses.at[e].set(vm)
        
    return vm_stresses

def export_to_paraview(filename, nodes, elements, U_global, vm_stresses):
    """
    Exports the FEM results into an ASCII VTK legacy format with fixed strict headers.
    Converts JAX arrays to native NumPy arrays first to avoid string formatting issues.
    """
    import numpy as np
    
    nodes_np = np.array(nodes)
    elements_np = np.array(elements)
    U_np = np.array(U_global)
    stresses_np = np.array(vm_stresses)
    
    num_nodes = nodes_np.shape[0]
    num_elements = elements_np.shape[0]
    
    with open(filename, 'w') as f:
        # 
        f.write("# vtk DataFile Version 3.0\n")         # Line 1: Version
        f.write("FEM SRI Q4 Locking Free Mesh Output\n") # Line 2: Title (CRITICAL FIX)
        f.write("ASCII\n")                               # Line 3: File Type
        f.write("DATASET UNSTRUCTURED_GRID\n\n")         # Line 4: Dataset Type
        
        # 
        f.write(f"POINTS {num_nodes} float\n")
        for i in range(num_nodes): 
            f.write(f"{nodes_np[i, 0]:.6f} {nodes_np[i, 1]:.6f} 0.000000\n")
        f.write("\n")
        
        # 
        total_slots = num_elements * 5
        f.write(f"CELLS {num_elements} {total_slots}\n")
        for e in range(num_elements):
            f.write(f"4 {int(elements_np[e,0])} {int(elements_np[e,1])} {int(elements_np[e,2])} {int(elements_np[e,3])}\n")
        f.write("\n")
        
        
        f.write(f"CELL_TYPES {num_elements}\n")
        for _ in range(num_elements): 
            f.write("9\n")
        f.write("\n")
        
        # (Point Data)
        f.write(f"POINT_DATA {num_nodes}\nVECTORS Displacement float\n")
        for i in range(num_nodes): 
            f.write(f"{U_np[2*i]:.6f} {U_np[2*i+1]:.6f} 0.000000\n")
        f.write("\n")
        
        # (Cell Data)
        f.write(f"CELL_DATA {num_elements}\nSCALARS Von_Mises_Stress float 1\nLOOKUP_TABLE default\n")
        for e in range(num_elements): 
            f.write(f"{stresses_np[e]:.6f}\n")

if __name__ == '__main__':
    L, H, E, nu = 10.0, 5.0, 1000.0, 0.4999
    
    # Establish fixed physical location coordinates for patch recovery
    target_x, target_y = 2.5, 1.25

    # 1. Compute High-Resolution Reference Solution (64x64 Mesh)
    print("Computing high-resolution reference solution ...")
    nodes_ref, elems_ref = generate_block_mesh(L, H, 64, 64)
    fixed_ref, F_ref, center_top_ref = setup_boundary_conditions(nodes_ref, 64, 64, L)
    K_ref = assemble_global_stiffness(nodes_ref, elems_ref, E, nu, use_sri=True)
    U_ref = solve_fem_system(K_ref, F_ref, fixed_ref)
    
    u_y_ref = U_ref[2 * center_top_ref + 1]
    vm_ref_all = compute_element_von_mises(nodes_ref, elems_ref, U_ref, E, nu)
    
    # Extract reference stress value using Patch Averaging
    centers_ref = jnp.mean(nodes_ref[elems_ref], axis=1)
    dists_ref = jnp.sum((centers_ref - jnp.array([target_x, target_y]))**2, axis=1)
    closest_ref_elements = jnp.argsort(dists_ref)[:4]
    vm_ref_val = jnp.mean(vm_ref_all[closest_ref_elements])

    # 2. Evaluation Mesh Pipeline Loop
    mesh_sizes = [4, 8, 16, 32]
    
    # Track all metrics to display everything simultaneously
    errors_disp_locked, errors_disp_sri = [], []
    real_disp_locked, real_disp_sri = [], []
    errors_stress_locked, errors_stress_sri = [], []

    for n in mesh_sizes:
        print(f"Analyzing {n}x{n} mesh...")
        nodes, elements = generate_block_mesh(L, H, n, n)
        fixed_dofs, F_global, center_top_node = setup_boundary_conditions(nodes, n, n, L)
        
        centers = jnp.mean(nodes[elements], axis=1)
        dists = jnp.sum((centers - jnp.array([target_x, target_y]))**2, axis=1)
        closest_elements = jnp.argsort(dists)[:4]

        # --- Standard Full Integration (Locked Mode) ---
        K_l = assemble_global_stiffness(nodes, elements, E, nu, use_sri=False)
        U_l = solve_fem_system(K_l, F_global, fixed_dofs)
        p_l = compute_element_von_mises(nodes, elements, U_l, E, nu)
        
        vm_l_val = jnp.mean(p_l[closest_elements])
        errors_disp_locked.append(jnp.abs((U_l[2 * center_top_node + 1] - u_y_ref) / u_y_ref))
        real_disp_locked.append(U_l[2 * center_top_node + 1])
        errors_stress_locked.append(jnp.abs((vm_l_val - vm_ref_val) / vm_ref_val))

        # --- Selective Reduced Integration (SRI Mode) ---
        K_s = assemble_global_stiffness(nodes, elements, E, nu, use_sri=True)
        U_s = solve_fem_system(K_s, F_global, fixed_dofs)
        p_s = compute_element_von_mises(nodes, elements, U_s, E, nu)
        
        vm_s_val = jnp.mean(p_s[closest_elements])
        errors_disp_sri.append(jnp.abs((U_s[2 * center_top_node + 1] - u_y_ref) / u_y_ref))
        real_disp_sri.append(U_s[2 * center_top_node + 1])
        errors_stress_sri.append(jnp.abs((vm_s_val - vm_ref_val) / vm_ref_val))

        if n == 32:
            export_to_paraview("block_locked_32x32.vtk", nodes, elements, U_l, p_l)
            export_to_paraview("block_sri_32x32.vtk", nodes, elements, U_s, p_s)

    # --- 3. Post-Processing Professional 3-Panel Visualizations ---
    import matplotlib.ticker as ticker
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Displacement Relative Error (Log-Log scale)
    ax1.loglog(mesh_sizes, errors_disp_locked, 'ro-', linewidth=1.5, markersize=6, label='Standard Q4 (Locked)')
    ax1.loglog(mesh_sizes, errors_disp_sri, 'go-', linewidth=1.5, markersize=6, label='Q4 with SRI (Locking-Free)')
    ax1.set_xlabel('Mesh Size (N x N)', fontsize=11)
    ax1.set_ylabel('Relative Displacement Error', fontsize=11)
    ax1.set_title('Displacement Error Rate', fontsize=12, fontweight='bold')
    ax1.set_xticks(mesh_sizes)
    ax1.set_xticklabels([str(n) for n in mesh_sizes])
    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
    ax1.xaxis.set_minor_formatter(ticker.NullFormatter()) 
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
    ax1.yaxis.set_minor_formatter(ticker.NullFormatter()) 
    ax1.grid(True, which="major", ls="--", alpha=0.6)
    ax1.legend(fontsize=10)

    # Panel 2: Real Displacement Value Convergence (Semi-Log scale: X=Log, Y=Linear)
    ax2.plot(mesh_sizes, real_disp_locked, 'ro-', linewidth=1.5, markersize=6, label='Standard Q4 (Locked)')
    ax2.plot(mesh_sizes, real_disp_sri, 'go-', linewidth=1.5, markersize=6, label='Q4 with SRI (Locking-Free)')
    
    ax2.set_xlabel('Mesh Size (N x N)', fontsize=11)
    ax2.set_ylabel('Actual Vertical Displacement (u_y)', fontsize=11)
    ax2.set_title('Displacement Assembly', fontsize=12, fontweight='bold')
    ax2.set_xticks(mesh_sizes)
    ax2.set_xticklabels([str(n) for n in mesh_sizes])
    ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
    ax2.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
    ax2.grid(True, which="major", ls="--", alpha=0.6)
    ax2.legend(fontsize=10, loc='lower right')

    # Panel 3: Von Mises Stress Relative Error (Log-Log scale)
    ax3.loglog(mesh_sizes, errors_stress_locked, 'ro-', linewidth=1.5, markersize=6, label='Standard Q4 (Locked)')
    ax3.loglog(mesh_sizes, errors_stress_sri, 'go-', linewidth=1.5, markersize=6, label='Q4 with SRI (Locking-Free)')
    ax3.set_xlabel('Mesh Size (N x N)', fontsize=11)
    ax3.set_ylabel('Relative Von Mises Stress Error', fontsize=11)
    ax3.set_title('Von Mises Stress Convergence', fontsize=12, fontweight='bold')
    ax3.set_xticks(mesh_sizes)
    ax3.set_xticklabels([str(n) for n in mesh_sizes])
    ax3.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
    ax3.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax3.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
    ax3.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax3.grid(True, which="major", ls="--", alpha=0.6)
    ax3.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('fem_convergence_charts.png', dpi=300)
    print("\n[SUCCESS] Flawless 3-panel charts saved to 'fem_convergence_charts.png'.")