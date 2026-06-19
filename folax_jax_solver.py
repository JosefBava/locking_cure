import jax
import jax.numpy as jnp

# ==========================================
# 1. MATERIAL & KINEMATICS FUNCTIONS
# ==========================================

def q4_shape_functions(xi, eta):
    """Standard Bilinear Q4 shape functions """
    return 0.25 * jnp.array([
        (1.0 - xi) * (1.0 - eta),
        (1.0 + xi) * (1.0 - eta),
        (1.0 + xi) * (1.0 + eta),
        (1.0 - xi) * (1.0 + eta)
    ])

def compute_strain(parent_coords, node_coords, node_u):
    """Computes small strain tensor using JAX Forward Auto-Diff"""
    # Mapping derivative: d(x)/d(xi)
    J = jax.jacfwd(lambda pt: jnp.dot(q4_shape_functions(pt[0], pt[1]), node_coords))(parent_coords)
    det_J = jnp.linalg.det(J)
    inv_J = jnp.linalg.inv(J)
    
    # Displacement derivative: d(u)/d(xi)
    grad_u_parent = jax.jacfwd(lambda pt: jnp.dot(q4_shape_functions(pt[0], pt[1]), node_u))(parent_coords)
    
    # Physical gradient: d(u)/d(x)
    grad_u_physical = jnp.dot(grad_u_parent, inv_J)
    strain = 0.5 * (grad_u_physical + grad_u_physical.T)
    return strain, det_J



g = 1.0 / jnp.sqrt(3.0)
GAUSS_2X2 = jnp.array([[-g, -g], [g, -g], [g, g], [-g, g]])
GAUSS_1X1 = jnp.array([[0.0, 0.0]])


def compute_element_energy_fullint(node_coords, node_u, mu, lam):
    """Standard full integration Q4 energy for linear elasticity."""
    def energy_at_point(pt):
        strain, det_J = compute_strain(pt, node_coords, node_u)
        trace_e = strain[0, 0] + strain[1, 1]
        energy_density = 0.5 * lam * (trace_e**2) + mu * (
            strain[0, 0]**2 + strain[1, 1]**2 + 2.0 * strain[0, 1]**2
        )
        return energy_density * det_J

    return jnp.sum(jax.vmap(energy_at_point)(GAUSS_2X2))


def compute_element_energy_sri(node_coords, node_u, mu, lam):
    """Selective reduced integration (SRI) cure for nearly incompressible Q4."""
    def shear_at_point(pt):
        strain, det_J = compute_strain(pt, node_coords, node_u)
        energy_shear = mu * (
            strain[0, 0]**2 + strain[1, 1]**2 + 2.0 * strain[0, 1]**2
        )
        return energy_shear * det_J

    energy_shear = jnp.sum(jax.vmap(shear_at_point)(GAUSS_2X2))
    strain, det_J = compute_strain(GAUSS_1X1[0], node_coords, node_u)
    trace_e = strain[0, 0] + strain[1, 1]
    energy_vol = 0.5 * lam * (trace_e**2) * det_J * 4.0

    return energy_shear + energy_vol


def compute_element_centroid_stress(node_coords, node_u, mu, lam):
    """Compute stress at the element centroid for error estimation."""
    strain, _ = compute_strain(GAUSS_1X1[0], node_coords, node_u)
    trace_e = strain[0, 0] + strain[1, 1]
    sigma_xx = lam * trace_e + 2.0 * mu * strain[0, 0]
    sigma_yy = lam * trace_e + 2.0 * mu * strain[1, 1]
    sigma_xy = 2.0 * mu * strain[0, 1]
    return jnp.array([sigma_xx, sigma_yy, sigma_xy])


def global_potential_energy(u_free, u_boundary, free_dofs, boundary_dofs, 
                            element_node_ids, X, mu, lam, f_ext,
                            formulation="sri"):
    """Assembles the total potential energy of the entire mesh system."""
    num_dofs = X.size
    u_full = jnp.zeros(num_dofs)
    u_full = u_full.at[free_dofs].set(u_free)
    u_full = u_full.at[boundary_dofs].set(u_boundary)
    u_full = u_full.reshape(-1, 2)
    
    elem_coords = X[element_node_ids]
    elem_u = u_full[element_node_ids]

    if formulation == "full":
        elem_energy_fn = jax.vmap(lambda coords, u: compute_element_energy_fullint(coords, u, mu, lam))
    elif formulation == "sri":
        elem_energy_fn = jax.vmap(lambda coords, u: compute_element_energy_sri(coords, u, mu, lam))
    else:
        raise ValueError("Unknown formulation: use 'full' or 'sri'")

    total_internal_energy = jnp.sum(elem_energy_fn(elem_coords, elem_u))
    external_work = jnp.dot(u_full.reshape(-1), f_ext)

    return total_internal_energy - external_work


def solve_linear_elasticity(u_boundary, boundary_dofs, free_dofs, element_node_ids, X, mu, lam, f_ext, formulation="sri"):
    """Solve the linear elastic equilibrium system with a potential energy formulation."""
    u_boundary = jnp.asarray(u_boundary, dtype=jnp.float64)
    free_dofs = jnp.asarray(free_dofs, dtype=jnp.int32)
    boundary_dofs = jnp.asarray(boundary_dofs, dtype=jnp.int32)
    f_ext = jnp.asarray(f_ext, dtype=jnp.float64)

    def energy_fn(u_free):
        return global_potential_energy(u_free, u_boundary, free_dofs, boundary_dofs,
                                       element_node_ids, X, mu, lam, f_ext,
                                       formulation=formulation)

    u0 = jnp.zeros(free_dofs.shape[0], dtype=jnp.float64)
    residual = jax.grad(energy_fn)(u0)
    stiffness = jax.hessian(energy_fn)(u0)
    u_free = jnp.linalg.solve(stiffness, -residual)

    num_dofs = X.size
    u_full = jnp.zeros(num_dofs, dtype=jnp.float64)
    u_full = u_full.at[free_dofs].set(u_free)
    u_full = u_full.at[boundary_dofs].set(u_boundary)

    return u_full.reshape(-1, 2), stiffness
