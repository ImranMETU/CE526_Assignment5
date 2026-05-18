"""
CE526 Assignment 5 — Part A
2D scalar Poisson FEM on an L-shaped domain using 3-node linear triangles.

Problem:
    -Δu = 1 in Ω
    u = 0 on Γ

Covers:
    1. problem definition
    2. node/element definition
    3. triangular FEM assembly
    4. nodal solution
    5. 1D plots
    6. 2D contour/nodal plots
    7. 3D surface plot
    8. gradients at requested points
    9. flux on x = 2 and y = -2
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation


# ============================================================
# 1. Problem definition
# ============================================================

K_MATERIAL = 1.0       # k in -div(k grad u) = f
SOURCE_F = 1.0         # f = 1
SAVE_FIGURES = True   # change to True if you want PNG files saved


# ============================================================
# 2. Mesh definition
# ============================================================

def build_nodes():
    """
    Build the 21 global nodes using the same numbering as the MATLAB codes.

    Origin is at the re-entrant corner:
        x = -2, -1, 0, 1, 2
        y = -2, -1, 0, 1, 2

    Missing lower-right part:
        x >= 1 and y < 0
    """
    nodes = {}
    node_id = 0

    for y in range(-2, 3):
        for x in range(-2, 3):
            if x >= 1 and y < 0:
                continue
            node_id += 1
            nodes[node_id] = (float(x), float(y))

    return nodes


def build_triangular_elements():
    """
    Triangular element connectivity.

    Each entry is a tuple of global node IDs.
    The local node order is counterclockwise, so element area is positive.

    This is the 24-triangle mesh for Part A.
    """
    return [
        (1, 2, 5),    (1, 5, 4),
        (2, 3, 5),    (3, 6, 5),
        (4, 5, 7),    (5, 8, 7),
        (5, 6, 9),    (5, 9, 8),

        (7, 8, 13),   (7, 13, 12),
        (8, 9, 13),   (9, 14, 13),
        (9, 10, 15),  (9, 15, 14),
        (10, 11, 15), (11, 16, 15),

        (12, 13, 17), (13, 18, 17),
        (13, 14, 19), (13, 19, 18),
        (14, 15, 19), (15, 20, 19),
        (15, 16, 21), (15, 21, 20),
    ]


def get_free_and_boundary_nodes():
    """
    For this assignment, all boundary nodes have u = 0.
    Free nodes are the five interior nodes.
    """
    free_nodes = [5, 8, 13, 14, 15]
    all_nodes = list(range(1, 22))
    boundary_nodes = [n for n in all_nodes if n not in free_nodes]
    return free_nodes, boundary_nodes


# ============================================================
# 3. Element calculations
# ============================================================

def triangle_area_beta_gamma(coords):
    """
    Compute area, beta, gamma for a 3-node linear triangle.

    coords:
        3 x 2 array:
            [[x1, y1],
             [x2, y2],
             [x3, y3]]

    Shape function derivatives:
        dNi/dx = beta_i / (2A)
        dNi/dy = gamma_i / (2A)
    """
    x = coords[:, 0]
    y = coords[:, 1]

    two_area = np.linalg.det(np.array([
        [1.0, x[0], y[0]],
        [1.0, x[1], y[1]],
        [1.0, x[2], y[2]],
    ]))

    area = 0.5 * two_area

    if area <= 0:
        raise ValueError(
            f"Triangle has non-positive area = {area}. "
            "Check local node ordering; use counterclockwise order."
        )

    beta = np.array([
        y[1] - y[2],
        y[2] - y[0],
        y[0] - y[1],
    ], dtype=float)

    gamma = np.array([
        x[2] - x[1],
        x[0] - x[2],
        x[1] - x[0],
    ], dtype=float)

    return area, beta, gamma


def triangular_element_matrices(coords, k=1.0, f=1.0):
    """
    Compute local stiffness matrix and force vector for one triangle.

    Ke_ij = k/(4A) * (beta_i beta_j + gamma_i gamma_j)

    For constant source f:
        Fe = f*A/3 * [1, 1, 1]^T
    """
    area, beta, gamma = triangle_area_beta_gamma(coords)

    ke = (k / (4.0 * area)) * (
        np.outer(beta, beta) + np.outer(gamma, gamma)
    )

    fe = (f * area / 3.0) * np.ones(3)

    return ke, fe, area, beta, gamma


# ============================================================
# 4. Assembly and solution
# ============================================================

def assemble_global_system(nodes, elements, k=1.0, f=1.0):
    """
    Assemble the global K and F for all 21 nodes.
    """
    n_nodes = len(nodes)
    K = np.zeros((n_nodes, n_nodes), dtype=float)
    F = np.zeros(n_nodes, dtype=float)

    element_data = {}

    for e_id, conn in enumerate(elements, start=1):
        coords = np.array([nodes[nid] for nid in conn], dtype=float)

        ke, fe, area, beta, gamma = triangular_element_matrices(coords, k, f)

        element_data[e_id] = {
            "connectivity": conn,
            "coords": coords,
            "area": area,
            "beta": beta,
            "gamma": gamma,
            "ke": ke,
            "fe": fe,
        }

        for a, I in enumerate(conn):
            I0 = I - 1
            F[I0] += fe[a]

            for b, J in enumerate(conn):
                J0 = J - 1
                K[I0, J0] += ke[a, b]

    return K, F, element_data


def solve_dirichlet_zero(K, F, free_nodes):
    """
    Solve Kff Uf = Ff for zero Dirichlet boundary conditions.
    """
    free_idx = np.array([n - 1 for n in free_nodes], dtype=int)

    Kff = K[np.ix_(free_idx, free_idx)]
    Ff = F[free_idx]

    Uf = np.linalg.solve(Kff, Ff)

    U = np.zeros(K.shape[0], dtype=float)
    U[free_idx] = Uf

    return U, Kff, Ff, Uf


# ============================================================
# 5. Gradient and flux
# ============================================================

def element_gradient(element_record, U):
    """
    Compute constant gradient inside one linear triangular element.

    grad u = 1/(2A) * [
        sum_i u_i beta_i,
        sum_i u_i gamma_i
    ]^T
    """
    conn = element_record["connectivity"]
    area = element_record["area"]
    beta = element_record["beta"]
    gamma = element_record["gamma"]

    u_local = np.array([U[nid - 1] for nid in conn], dtype=float)

    du_dx = np.dot(u_local, beta) / (2.0 * area)
    du_dy = np.dot(u_local, gamma) / (2.0 * area)

    return np.array([du_dx, du_dy], dtype=float)


def barycentric_coordinates(point, coords):
    """
    Return barycentric coordinates of point with respect to a triangle.
    """
    x, y = point

    A = np.array([
        [coords[0, 0], coords[1, 0], coords[2, 0]],
        [coords[0, 1], coords[1, 1], coords[2, 1]],
        [1.0,          1.0,          1.0],
    ], dtype=float)

    b = np.array([x, y, 1.0], dtype=float)
    return np.linalg.solve(A, b)


def find_triangles_containing_point(point, element_data, tol=1e-10):
    """
    Find all triangles containing a point.

    If the point lies on an element edge, this will usually return two triangles.
    """
    matches = []

    for e_id, record in element_data.items():
        coords = record["coords"]
        lambdas = barycentric_coordinates(point, coords)

        inside = (
            np.all(lambdas >= -tol) and
            np.all(lambdas <= 1.0 + tol)
        )

        if inside:
            matches.append((e_id, lambdas))

    return matches


def report_gradients_at_points(points, element_data, U):
    """
    Report gradients at requested points.
    """
    print("\n" + "=" * 72)
    print("GRADIENT RESULTS")
    print("=" * 72)

    for point in points:
        matches = find_triangles_containing_point(point, element_data)

        print(f"\nPoint {point}:")
        if not matches:
            print("  No containing triangle found.")
            continue

        grads = []
        for e_id, lambdas in matches:
            grad = element_gradient(element_data[e_id], U)
            grads.append(grad)

            conn = element_data[e_id]["connectivity"]
            print(f"  Element {e_id:>2}, nodes {conn}:")
            print(f"    barycentric = {lambdas}")
            print(f"    grad u      = [{grad[0]: .6f}, {grad[1]: .6f}]")

        if len(grads) > 1:
            avg = np.mean(np.vstack(grads), axis=0)
            print(f"  Average gradient:")
            print(f"    grad u_avg  = [{avg[0]: .6f}, {avg[1]: .6f}]")


def edge_on_line(p1, p2, line_type, value, tol=1e-10):
    """
    Check whether an edge lies on x=value or y=value.
    """
    if line_type == "x":
        return abs(p1[0] - value) < tol and abs(p2[0] - value) < tol
    if line_type == "y":
        return abs(p1[1] - value) < tol and abs(p2[1] - value) < tol
    raise ValueError("line_type must be 'x' or 'y'.")


def report_flux_on_boundary_line(line_type, value, normal, nodes, element_data, U, k=1.0):
    """
    Report k grad(u) vector and k grad(u) · n along a boundary line.

    line_type:
        "x" for x = value
        "y" for y = value

    normal:
        outward unit normal vector.
    """
    normal = np.array(normal, dtype=float)

    print("\n" + "=" * 72)
    print(f"FLUX RESULTS ON {line_type} = {value}")
    print("=" * 72)
    print(f"Outward normal n = [{normal[0]:.1f}, {normal[1]:.1f}]")

    found = False

    for e_id, record in element_data.items():
        conn = record["connectivity"]

        # Local triangle edges: (0,1), (1,2), (2,0)
        local_edges = [(0, 1), (1, 2), (2, 0)]

        for a, b in local_edges:
            n1 = conn[a]
            n2 = conn[b]
            p1 = nodes[n1]
            p2 = nodes[n2]

            if edge_on_line(p1, p2, line_type, value):
                found = True
                grad = element_gradient(record, U)
                k_grad = k * grad
                math_normal_flux = float(np.dot(k_grad, normal))
                physical_outward_flux = -math_normal_flux

                print(f"\nBoundary edge nodes ({n1}, {n2}) adjacent to element {e_id}:")
                print(f"  element nodes       = {conn}")
                print(f"  grad u              = [{grad[0]: .6f}, {grad[1]: .6f}]")
                print(f"  k grad u            = [{k_grad[0]: .6f}, {k_grad[1]: .6f}]")
                print(f"  k grad u · n        = {math_normal_flux: .6f}")
                print(f"  -k grad u · n       = {physical_outward_flux: .6f}  # heat-flux convention")

    if not found:
        print("No boundary edge found on this line.")


# ============================================================
# 6. Plotting
# ============================================================

def make_arrays_for_plotting(nodes, elements, U):
    """
    Convert dictionaries/connectivity to arrays for matplotlib.
    """
    node_ids = sorted(nodes.keys())
    xy = np.array([nodes[nid] for nid in node_ids], dtype=float)

    x = xy[:, 0]
    y = xy[:, 1]
    u = np.array([U[nid - 1] for nid in node_ids], dtype=float)

    triangles = np.array([[nid - 1 for nid in conn] for conn in elements], dtype=int)

    triangulation = Triangulation(x, y, triangles)

    return x, y, u, triangulation


def plot_1d_side_views(nodes, U):
    """
    1D side-view nodal plots.
    One separate figure per horizontal mesh line.
    Points only: no connecting lines.
    """
    rows_to_plot = [-1.0, 0.0, 1.0]

    filename_map = {
        -1.0: "images/partA_side_y_minus1.png",
         0.0: "images/partA_side_y_0.png",
         1.0: "images/partA_side_y_1.png",
    }

    figures = []

    for y_value in rows_to_plot:
        row_nodes = [
            (nid, coord[0], U[nid - 1])
            for nid, coord in nodes.items()
            if abs(coord[1] - y_value) < 1e-12
        ]

        row_nodes.sort(key=lambda item: item[1])

        xs = [item[1] for item in row_nodes]
        us = [item[2] for item in row_nodes]

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.scatter(xs, us, s=70)

        for nid, x_val, u_val in row_nodes:
            ax.text(
                x_val + 0.03,
                u_val + 0.01,
                f"{nid}",
                fontsize=8
            )

        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.set_title(f"Part A: 1D side-view nodal values along y = {y_value:g}")
        ax.grid(True)

        # Same vertical scale for all Part A side-view plots
        ax.set_ylim(-0.02, float(np.max(U)) * 1.15)

        fig.tight_layout()

        filename = filename_map[y_value]

        if SAVE_FIGURES:
            fig.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"Saved Part A side-view figure: {filename}")

        figures.append((fig, ax))

        # Terminal verification
        print(f"\nPart A side-view data for y = {y_value:g}")
        for nid, x_val, u_val in row_nodes:
            print(f"  node {nid:>2}: x = {x_val:>5.1f}, u = {u_val:.6f}")

    return figures

def plot_2d_nodal_map(nodes, U):
    """
    2D nodal map with node numbers and u-values.
    """
    x = np.array([nodes[nid][0] for nid in sorted(nodes)])
    y = np.array([nodes[nid][1] for nid in sorted(nodes)])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x, y, s=80)

    for nid in sorted(nodes):
        xi, yi = nodes[nid]
        ui = U[nid - 1]
        ax.text(
            xi + 0.03,
            yi + 0.03,
            f"{nid}\nu={ui:.3f}",
            fontsize=8,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("2D nodal map: node IDs and u-values")
    ax.grid(True)

    fig.tight_layout()

    if SAVE_FIGURES:
        fig.savefig("images/partA_2D_nodal_map.png", dpi=300)

    return fig, ax


def plot_2d_contour(nodes, elements, U):
    """
    2D triangular contour plot.
    """
    x, y, u, triangulation = make_arrays_for_plotting(nodes, elements, U)

    fig, ax = plt.subplots(figsize=(7, 7))

    contour = ax.tricontourf(triangulation, u, levels=12)
    ax.triplot(triangulation, linewidth=0.8)
    ax.scatter(x, y, s=20)

    fig.colorbar(contour, ax=ax, label="u")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("2D contour plot of u over triangular mesh")
    ax.grid(True)

    fig.tight_layout()

    if SAVE_FIGURES:
        fig.savefig("images/partA_2D_contour.png", dpi=300)

    return fig, ax


def plot_3d_surface(nodes, elements, U):
    """
    3D nodal-value plot.
    Points only: no triangular surface, no connecting facets.
    """
    x, y, u, triangulation = make_arrays_for_plotting(nodes, elements, U)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(x, y, u, s=60)

    # Optional: annotate node numbers
    for nid in sorted(nodes):
        xi, yi = nodes[nid]
        ui = U[nid - 1]
        ax.text(xi, yi, ui + 0.015, str(nid), fontsize=8)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u")
    ax.set_title("3D nodal-value plot, Part A")

    fig.tight_layout()

    if SAVE_FIGURES:
        fig.savefig("images/partA_3D_points_only.png", dpi=300)

    return fig, ax


# ============================================================
# 7. Main execution
# ============================================================

def main():
    nodes = build_nodes()
    elements = build_triangular_elements()
    free_nodes, boundary_nodes = get_free_and_boundary_nodes()

    K, F, element_data = assemble_global_system(
        nodes,
        elements,
        k=K_MATERIAL,
        f=SOURCE_F,
    )

    U, Kff, Ff, Uf = solve_dirichlet_zero(K, F, free_nodes)

    print("=" * 72)
    print("CE526 Assignment 5 — Part A: 2D triangular FEM")
    print("=" * 72)

    print("\nProblem:")
    print("  -Delta u = 1 in Omega")
    print("  u = 0 on all boundaries")
    print("  k = 1, f = 1")

    print("\nFree nodes:")
    print(f"  {free_nodes}")

    print("\nBoundary nodes:")
    print(f"  {boundary_nodes}")

    print("\n" + "=" * 72)
    print("REDUCED SYSTEM")
    print("=" * 72)

    print("\nKff =")
    print(np.array2string(Kff, precision=6, suppress_small=True))

    print("\nFf =")
    print(np.array2string(Ff, precision=6, suppress_small=True))

    print("\nUf =")
    print(np.array2string(Uf, precision=6, suppress_small=True))

    print("\n" + "=" * 72)
    print("NODAL SOLUTION")
    print("=" * 72)
    print(f"{'Node':>5} {'x':>8} {'y':>8} {'u':>12}")
    print("-" * 40)

    for nid in sorted(nodes):
        x, y = nodes[nid]
        print(f"{nid:>5d} {x:>8.3f} {y:>8.3f} {U[nid - 1]:>12.6f}")

    requested_points = [(-0.5, 0.0), (0.0, 0.5)]
    report_gradients_at_points(requested_points, element_data, U)

    # x = 2: outward normal = [1, 0]
    report_flux_on_boundary_line(
        line_type="x",
        value=2.0,
        normal=[1.0, 0.0],
        nodes=nodes,
        element_data=element_data,
        U=U,
        k=K_MATERIAL,
    )

    # y = -2: outward normal = [0, -1]
    report_flux_on_boundary_line(
        line_type="y",
        value=-2.0,
        normal=[0.0, -1.0],
        nodes=nodes,
        element_data=element_data,
        U=U,
        k=K_MATERIAL,
    )

    plot_1d_side_views(nodes, U)
    plot_2d_nodal_map(nodes, U)
    plot_2d_contour(nodes, elements, U)
    plot_3d_surface(nodes, elements, U)

    plt.show()


if __name__ == "__main__":
    main()