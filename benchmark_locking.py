import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from jax import config
config.update("jax_enable_x64", True)

from folax_jax_solver import q4_shape_functions, compute_element_centroid_stress, global_potential_energy, solve_linear_elasticity

# --- Mesh helpers ---

def generate_q4_mesh(nx, ny, width=1.0, height=1.0):
    x = jnp.linspace(0.0, width, nx + 1)
    y = jnp.linspace(0.0, height, ny + 1)
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


def q4_shape_derivatives(xi, eta):
    return 0.25 * jnp.array([
        [-(1 - eta), 1 - eta, 1 + eta, -(1 + eta)],
        [-(1 - xi), -(1 + xi), 1 + xi, 1 - xi]
    ])


def assemble_body_force(X, element_node_ids, f_body):
    n_nodes = X.shape[0]
    f_ext = jnp.zeros((2 * n_nodes,), dtype=jnp.float64)

    gauss_pts = jnp.array([[-1.0 / jnp.sqrt(3.0), -1.0 / jnp.sqrt(3.0)],
                           [1.0 / jnp.sqrt(3.0), -1.0 / jnp.sqrt(3.0)],
                           [1.0 / jnp.sqrt(3.0), 1.0 / jnp.sqrt(3.0)],
                           [-1.0 / jnp.sqrt(3.0), 1.0 / jnp.sqrt(3.0)]])
    weights = jnp.ones((4,), dtype=jnp.float64)

    for el in element_node_ids:
        coords = X[el]
        fe = jnp.zeros((8,), dtype=jnp.float64)
        for pt, w in zip(gauss_pts, weights):
            dN_dxi = q4_shape_derivatives(pt[0], pt[1])
            J = dN_dxi @ coords
            detJ = jnp.linalg.det(J)
            N = q4_shape_functions(pt[0], pt[1])
            for a in range(4):
                fe = fe.at[2 * a:2 * a + 2].add(N[a] * f_body * detJ * w)

        dof_indices = jnp.arange(8, dtype=jnp.int32).reshape(4, 2)
        dofs = jnp.ravel(jnp.column_stack([2 * el, 2 * el + 1]))
        f_ext = f_ext.at[dofs].add(fe)

    return f_ext

# --- Analytical solution and forcing ---

def exact_displacement(nodes, alpha=0.25, beta=0.15):
    x = nodes[:, 0]
    y = nodes[:, 1]
    return jnp.column_stack([alpha * x**2, beta * y**2])


def exact_stress(nodes, mu, lam, alpha=0.25, beta=0.15):
    x = nodes[:, 0]
    y = nodes[:, 1]
    eps_xx = 2.0 * alpha * x
    eps_yy = 2.0 * beta * y
    trace_e = eps_xx + eps_yy
    sigma_xx = lam * trace_e + 2.0 * mu * eps_xx
    sigma_yy = lam * trace_e + 2.0 * mu * eps_yy
    sigma_xy = jnp.zeros_like(x)
    return jnp.column_stack([sigma_xx, sigma_yy, sigma_xy])


def body_force(mu, lam, alpha=0.25, beta=0.15):
    f_x = - (2.0 * lam * alpha + 4.0 * mu * alpha)
    f_y = - (2.0 * lam * beta + 4.0 * mu * beta)
    return jnp.array([f_x, f_y], dtype=jnp.float64)


def build_boundary_conditions(nodes):
    boundary = (nodes[:, 0] == 0.0) | (nodes[:, 0] == 1.0) | (nodes[:, 1] == 0.0) | (nodes[:, 1] == 1.0)
    boundary_nodes = jnp.where(boundary)[0]
    free_nodes = jnp.where(~boundary)[0]
    boundary_dofs = jnp.sort(jnp.concatenate([2 * boundary_nodes, 2 * boundary_nodes + 1]))
    free_dofs = jnp.sort(jnp.concatenate([2 * free_nodes, 2 * free_nodes + 1]))
    return free_dofs.astype(jnp.int32), boundary_dofs.astype(jnp.int32)


def compute_error(u_full, nodes, element_node_ids, mu, lam):
    u_exact = exact_displacement(nodes)
    disp_error = jnp.linalg.norm((u_full - u_exact).reshape(-1)) / jnp.linalg.norm(u_exact.reshape(-1))

    element_coords = nodes[element_node_ids]
    element_displacements = u_full[element_node_ids]
    computed_stresses = jax.vmap(lambda coords, u: compute_element_centroid_stress(coords, u, mu, lam))(element_coords, element_displacements)
    centroids = jnp.mean(element_coords, axis=1)
    exact_sigma = exact_stress(centroids, mu, lam)
    stress_error = jnp.linalg.norm(computed_stresses - exact_sigma) / jnp.linalg.norm(exact_sigma)
    return float(disp_error), float(stress_error)


def run_refinement(nx, ny, nu, formulation):
    nodes, elements = generate_q4_mesh(nx, ny)
    E = 1.0
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    free_dofs, boundary_dofs = build_boundary_conditions(nodes)
    u_boundary = exact_displacement(nodes).reshape(-1)[boundary_dofs]
    f_ext = assemble_body_force(nodes, elements, body_force(mu, lam))

    u_full, _ = solve_linear_elasticity(u_boundary, boundary_dofs, free_dofs,
                                       elements, nodes, mu, lam, f_ext,
                                       formulation=formulation)
    return compute_error(u_full, nodes, elements, mu, lam)


def main():
    mesh_sizes = [4, 8, 12]
    nu = 0.4999
    results = {"full": {"disp": [], "stress": []},
               "sri": {"disp": [], "stress": []}}

    for formulation in ["full", "sri"]:
        for n in mesh_sizes:
            disp_err, stress_err = run_refinement(n, n, nu, formulation)
            print(f"{formulation.upper():>4} mesh {n}x{n}: disp err = {disp_err:.3e}, stress err = {stress_err:.3e}")
            results[formulation]["disp"].append(disp_err)
            results[formulation]["stress"].append(stress_err)

    h = 1.0 / jnp.array(mesh_sizes, dtype=jnp.float64)
    plt.figure(figsize=(8, 5))
    plt.loglog(h, results["full"]["disp"], "-o", label="Full integration")
    plt.loglog(h, results["sri"]["disp"], "-s", label="SRI (locking cure)")
    plt.xlabel("Mesh size h")
    plt.ylabel("Relative displacement error")
    plt.title("Displacement convergence: locking vs. locking cure")
    plt.gca().invert_xaxis()
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("locking_displacement_convergence.png", dpi=200)

    plt.figure(figsize=(8, 5))
    plt.loglog(h, results["full"]["stress"], "-o", label="Full integration")
    plt.loglog(h, results["sri"]["stress"], "-s", label="SRI (locking cure)")
    plt.xlabel("Mesh size h")
    plt.ylabel("Relative stress error")
    plt.title("Stress convergence: locking vs. locking cure")
    plt.gca().invert_xaxis()
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("locking_stress_convergence.png", dpi=200)

    print("Finished benchmark. Plots saved as locking_displacement_convergence.png and locking_stress_convergence.png.")

if __name__ == "__main__":
    main()
