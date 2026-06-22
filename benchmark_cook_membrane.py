import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jax import config
config.update("jax_enable_x64", True)

from folax_jax_solver import solve_linear_elasticity


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


def build_left_edge_boundary_conditions(nodes):
    left_edge = nodes[:, 0] == 0.0
    boundary_nodes = jnp.where(left_edge)[0]
    boundary_dofs = jnp.sort(jnp.concatenate([2 * boundary_nodes, 2 * boundary_nodes + 1]))
    all_dofs = jnp.arange(2 * nodes.shape[0], dtype=jnp.int32)
    free_dofs = jnp.setdiff1d(all_dofs, boundary_dofs)
    return free_dofs.astype(jnp.int32), boundary_dofs.astype(jnp.int32)


def assemble_right_edge_traction(nodes, traction):
    n_nodes = nodes.shape[0]
    f_ext = jnp.zeros((2 * n_nodes,), dtype=jnp.float64)

    right_edge = nodes[:, 0] == 1.0
    right_nodes = jnp.where(right_edge)[0]
    right_nodes = right_nodes[jnp.argsort(nodes[right_nodes, 1])]

    for i in range(right_nodes.shape[0] - 1):
        n1 = right_nodes[i]
        n2 = right_nodes[i + 1]
        edge_length = jnp.linalg.norm(nodes[n2] - nodes[n1])
        f_ext = f_ext.at[2 * n1:2 * n1 + 2].add(0.5 * traction * edge_length)
        f_ext = f_ext.at[2 * n2:2 * n2 + 2].add(0.5 * traction * edge_length)

    return f_ext


def run_cook_membrane(nx, ny, nu, formulation):
    nodes, elements = generate_q4_mesh(nx, ny)
    E = 1.0
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    free_dofs, boundary_dofs = build_left_edge_boundary_conditions(nodes)
    u_boundary = jnp.zeros(boundary_dofs.shape, dtype=jnp.float64)
    traction = jnp.array([1.0, 0.0], dtype=jnp.float64)
    f_ext = assemble_right_edge_traction(nodes, traction)

    u_full, _ = solve_linear_elasticity(u_boundary, boundary_dofs, free_dofs,
                                       elements, nodes, mu, lam, f_ext,
                                       formulation=formulation)
    return u_full


def tip_displacement(u_full, nodes):
    top_right = jnp.where((nodes[:, 0] == 1.0) & (nodes[:, 1] == 1.0))[0][0]
    return float(u_full[top_right, 0])


def main():
    mesh_sizes = [4, 8, 12, 16]
    nu = 0.4999
    methods = ["full", "sri", "bbar"]
    results = {method: [] for method in methods}

    for method in methods:
        for n in mesh_sizes:
            nodes, _ = generate_q4_mesh(n, n)
            u_full = run_cook_membrane(n, n, nu, method)
            disp = tip_displacement(u_full, nodes)
            print(f"{method.upper():>4} mesh {n}x{n}: tip x-displacement = {disp:.6e}")
            results[method].append(disp)

    plt.figure(figsize=(8, 5))
    plt.plot(mesh_sizes, results["full"], "-o", label="Full integration")
    plt.plot(mesh_sizes, results["sri"], "-s", label="SRI")
    plt.plot(mesh_sizes, results["bbar"], "-^", label="B-bar")
    plt.xlabel("Mesh refinement (nx = ny)")
    plt.ylabel("Tip x-displacement")
    plt.title("Cook's membrane-like locking benchmark: tip displacement")
    plt.gca().invert_xaxis()
    plt.grid(True, ls="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("cook_membrane_tip_displacement.png", dpi=200)

    print("Finished locking benchmark. Plot saved as cook_membrane_tip_displacement.png.")


if __name__ == "__main__":
    main()
