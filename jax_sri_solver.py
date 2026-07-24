import jax
import jax.numpy as jnp

# Enable 64-bit precision in JAX to eliminate round-off errors and stabilize convergence
from jax import config
config.update("jax_enable_x64", True)

def get_standard_material_matrix(E, nu):
    """
    Returns the standard, untouched Plane Strain constitutive matrix (E).
    Matches exactly with classic definitions.
    """
    factor = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    E_mat = factor * jnp.array([
        [1.0 - nu, nu,      0.0],
        [nu,      1.0 - nu, 0.0],
        [0.0,      0.0,      0.5 * (1.0 - 2.0 * nu)]
    ], dtype=jnp.float64)
    return E_mat

def q4_shape_derivatives(xi, eta):
    """
    Returns natural derivatives of Q4 shape functions with respect to xi and eta.
    """
    dN_dxi = 0.25 * jnp.array([-(1.0 - eta),  (1.0 - eta), (1.0 + eta), -(1.0 + eta)], dtype=jnp.float64)
    dN_deta = 0.25 * jnp.array([-(1.0 - xi), -(1.0 + xi),  (1.0 + xi),  (1.0 - xi)], dtype=jnp.float64)
    return jnp.stack([dN_dxi, dN_deta])

def compute_jacobian(X_e, xi, eta):
    """
    Computes the Jacobian matrix, its determinant, and its inverse for a Q4 element.
    """
    dN_dxi_eta = q4_shape_derivatives(xi, eta)
    J = jnp.dot(dN_dxi_eta, X_e)
    detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
    invJ = jnp.array([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]], dtype=jnp.float64) / detJ
    return J, detJ, invJ

def compute_B_matrix(X_e, xi, eta):
    """
    Constructs the standard strain-displacement (B) matrix at a given Gauss point.
    """
    J, detJ, invJ = compute_jacobian(X_e, xi, eta)
    dN_dxi_eta = q4_shape_derivatives(xi, eta)
    dN_dx = jnp.dot(invJ, dN_dxi_eta)
    
    B = jnp.zeros((3, 8), dtype=jnp.float64)
    B = B.at[0, 0::2].set(dN_dx[0, :])
    B = B.at[1, 1::2].set(dN_dx[1, :])
    B = B.at[2, 0::2].set(dN_dx[1, :])
    B = B.at[2, 1::2].set(dN_dx[0, :])
    return B, detJ

def compute_element_stiffness(X_e, E, nu, use_sri=True):
    """
    Computes the 8x8 element stiffness matrix by splitting the B matrix 
    into Deviatoric and Volumetric projection components.
    """
    E_mat = get_standard_material_matrix(E, nu)
    K_e = jnp.zeros((8, 8), dtype=jnp.float64)
    
    # Define projection matrices to separate volumetric and deviatoric strain contributions
    H_vol = jnp.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 0.0]], dtype=jnp.float64)
    H_dev = jnp.array([[0.5, -0.5, 0.0], [-0.5, 0.5, 0.0], [0.0, 0.0, 1.0]], dtype=jnp.float64)
    
    # Standard 2x2 Gauss integration rule points and weights
    g_pts = jnp.array([-1.0 / jnp.sqrt(3.0), 1.0 / jnp.sqrt(3.0)], dtype=jnp.float64)
    g_wts = jnp.array([1.0, 1.0], dtype=jnp.float64)
    
    # --- 2x2 Full Integration Scheme for Deviatoric Component ---
    for i in range(2):
        for j in range(2):
            xi, eta = g_pts[i], g_pts[j]
            weight = g_wts[i] * g_wts[j]
            B, detJ = compute_B_matrix(X_e, xi, eta)
            
            # Extract B_dev component
            B_dev = jnp.dot(H_dev, B)
            K_dev = jnp.dot(B_dev.T, jnp.dot(E_mat, B_dev)) * detJ * weight
            K_e = K_e.at[:, :].add(K_dev)
            
            # If SRI is disabled, integrate Volumetric component at 2x2 as well (Causes Locking)
            if not use_sri:
                B_vol = jnp.dot(H_vol, B)
                K_vol_locked = jnp.dot(B_vol.T, jnp.dot(E_mat, B_vol)) * detJ * weight
                K_e = K_e.at[:, :].add(K_vol_locked)
                
    # --- 1x1 Reduced Integration Scheme for Volumetric Component (SRI Mode) ---
    if use_sri:
        # Evaluate strictly at the center of the element (xi=0, eta=0) with weight=4.0
        B_center, detJ_center = compute_B_matrix(X_e, 0.0, 0.0)
        B_vol_center = jnp.dot(H_vol, B_center)
        
        K_vol_free = jnp.dot(B_vol_center.T, jnp.dot(E_mat, B_vol_center)) * detJ_center * 4.0
        K_e = K_e.at[:, :].add(K_vol_free)
        
    return K_e

def assemble_global_stiffness(nodes, elements, E, nu, use_sri=True):
    """
    Assembles the Global Stiffness Matrix using JAX vmap without JIT tracing.
    """
    num_nodes = nodes.shape[0]
    num_elements = elements.shape[0]
    K_global = jnp.zeros((2 * num_nodes, 2 * num_nodes), dtype=jnp.float64)
    X_all_elements = nodes[elements]
    
    # Parallelize the element stiffness computation over all elements
    compute_all_Ke = jax.vmap(lambda X: compute_element_stiffness(X, E, nu, use_sri))
    K_elements = compute_all_Ke(X_all_elements)
    
    # Assembly pipeline mapping element DOFs to global system matrix
    for e in range(num_elements):
        node_indices = elements[e]
        dofs = jnp.array([
            2 * node_indices[0], 2 * node_indices[0] + 1,
            2 * node_indices[1], 2 * node_indices[1] + 1,
            2 * node_indices[2], 2 * node_indices[2] + 1,
            2 * node_indices[3], 2 * node_indices[3] + 1
        ])
        K_global = K_global.at[jnp.ix_(dofs, dofs)].add(K_elements[e])
        
    return K_global

def solve_fem_system(K_global, F_global, fixed_dofs):
    """
    Applies Dirichlet boundary conditions and solves the system using linear algebra.
    """
    K_modified = K_global.at[fixed_dofs, :].set(0.0)
    K_modified = K_modified.at[fixed_dofs, fixed_dofs].set(1.0)
    F_modified = F_global.at[fixed_dofs].set(0.0)
    return jnp.linalg.solve(K_modified, F_modified)