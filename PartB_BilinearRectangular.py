"""
CE526 Assignment 5 — Part B
2D scalar Poisson FEM on an L-shaped domain using 4-node bilinear rectangles.

Problem:
    -Δu = 1 in Ω
    u = 0 on Γ

Part B:
    Repeat Part A using 12 bilinear rectangular elements, each 1x1.

Covers:
    1. problem definition
    2. node/element definition
    3. bilinear rectangular FEM assembly
    4. nodal solution
    5. 1D side-view plots, points only
    6. 2D nodal/contour-style plots
    7. 3D nodal plot, points only
    8. gradients at requested points
    9. flux along x = 2 and y = -2
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


# ============================================================
# 1. Problem definition
# ============================================================

K_MATERIAL = 1.0
SOURCE_F = 1.0
SAVE_FIGURES = True


# ============================================================
# 2. Mesh definition
# ============================================================

def build_nodes():
    """
    Build the same 21 nodes as Part A.

    Node coordinates use origin at the re-entrant corner.
    Missing region:
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


def build_rectangular_elements():
    """
    12 bilinear rectangular elements.

    Local order:
        1 = BL
        2 = BR
        3 = TR
        4 = TL
    """
    return [
        (1, 2, 5, 4),
        (2, 3, 6, 5),
        (4, 5, 8, 7),
        (5, 6, 9, 8),

        (7, 8, 13, 12),
        (8, 9, 14, 13),
        (9, 10, 15, 14),
        (10, 11, 16, 15),

        (12, 13, 18, 17),
        (13, 14, 19, 18),
        (14, 15, 20, 19),
        (15, 16, 21, 20),
    ]


def get_free_and_boundary_nodes():
    free_nodes = [5, 8, 13, 14, 15]
    all_nodes = list(range(1, 22))
    boundary_nodes = [n for n in all_nodes if n not in free_nodes]
    return free_nodes, boundary_nodes


# ============================================================
# 3. Bilinear rectangular element
# ============================================================

def unit_rectangle_stiffness(k=1.0):
    """
    Stiffness matrix for a 1x1 bilinear rectangular element.

    Shape functions on 0 <= xi <= 1, 0 <= eta <= 1:

        N1 = (1-xi)(1-eta)
        N2 = xi(1-eta)
        N3 = xi eta
        N4 = (1-xi) eta

    For unit square and k=1:
        Ke_ij = ∫∫ (Ni,x Nj,x + Ni,y Nj,y) dA
    """
    Ke = np.array([
        [ 2/3, -1/6, -1/3, -1/6],
        [-1/6,  2/3, -1/6, -1/3],
        [-1/3, -1/6,  2/3, -1/6],
        [-1/6, -1/3, -1/6,  2/3],
    ], dtype=float)

    return k * Ke


def unit_rectangle_force(f=1.0):
    """
    Force vector for constant source f over a 1x1 rectangle.

    Fe_i = ∫∫ f Ni dA = f/4
    """
    return (f / 4.0) * np.ones(4)


def shape_functions_and_derivatives(xi, eta):
    """
    Bilinear shape functions and derivatives with respect to x and y.

    Since each element is 1x1:
        x = x0 + xi
        y = y0 + eta

    Therefore:
        d/dx = d/dxi
        d/dy = d/deta
    """
    N = np.array([
        (1.0 - xi) * (1.0 - eta),
        xi * (1.0 - eta),
        xi * eta,
        (1.0 - xi) * eta,
    ], dtype=float)

    dN_dx = np.array([
        -(1.0 - eta),
         (1.0 - eta),
         eta,
        -eta,
    ], dtype=float)

    dN_dy = np.array([
        -(1.0 - xi),
        -xi,
         xi,
         (1.0 - xi),
    ], dtype=float)

    return N, dN_dx, dN_dy


def physical_to_local(point, element_coords):
    """
    Convert physical point (x,y) to local coordinates (xi, eta)
    for a 1x1 axis-aligned rectangle.

    Local order:
        BL, BR, TR, TL
    """
    x, y = point
    x0, y0 = element_coords[0]  # BL node

    xi = x - x0
    eta = y - y0

    return xi, eta


def point_in_rectangle(point, element_coords, tol=1e-10):
    """
    Check whether point lies inside or on the boundary of an axis-aligned 1x1 rectangle.
    """
    xi, eta = physical_to_local(point, element_coords)

    return (
        xi >= -tol and xi <= 1.0 + tol and
        eta >= -tol and eta <= 1.0 + tol
    )


# ============================================================
# 4. Assembly and solution
# ============================================================

def assemble_global_system(nodes, elements, k=1.0, f=1.0):
    n_nodes = len(nodes)
    K = np.zeros((n_nodes, n_nodes), dtype=float)
    F = np.zeros(n_nodes, dtype=float)

    Ke = unit_rectangle_stiffness(k)
    Fe = unit_rectangle_force(f)

    element_data = {}

    for e_id, conn in enumerate(elements, start=1):
        coords = np.array([nodes[nid] for nid in conn], dtype=float)

        element_data[e_id] = {
            "connectivity": conn,
            "coords": coords,
            "ke": Ke.copy(),
            "fe": Fe.copy(),
        }

        for a, I in enumerate(conn):
            I0 = I - 1
            F[I0] += Fe[a]

            for b, J in enumerate(conn):
                J0 = J - 1
                K[I0, J0] += Ke[a, b]

    return K, F, element_data


def solve_dirichlet_zero(K, F, free_nodes):
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

def rectangle_gradient_at_point(element_record, U, point):
    """
    Compute grad(u) at a physical point inside a bilinear rectangular element.

    u,x = Σ ui Ni,x
    u,y = Σ ui Ni,y
    """
    conn = element_record["connectivity"]
    coords = element_record["coords"]

    xi, eta = physical_to_local(point, coords)
    _, dN_dx, dN_dy = shape_functions_and_derivatives(xi, eta)

    u_local = np.array([U[nid - 1] for nid in conn], dtype=float)

    du_dx = np.dot(u_local, dN_dx)
    du_dy = np.dot(u_local, dN_dy)

    return np.array([du_dx, du_dy], dtype=float), xi, eta


def find_rectangles_containing_point(point, element_data, tol=1e-10):
    matches = []

    for e_id, record in element_data.items():
        coords = record["coords"]
        if point_in_rectangle(point, coords, tol=tol):
            xi, eta = physical_to_local(point, coords)
            matches.append((e_id, xi, eta))

    return matches


def report_gradients_at_points(points, element_data, U):
    print("\n" + "=" * 72)
    print("GRADIENT RESULTS — BILINEAR RECTANGLES")
    print("=" * 72)

    for point in points:
        matches = find_rectangles_containing_point(point, element_data)

        print(f"\nPoint {point}:")
        if not matches:
            print("  No containing rectangle found.")
            continue

        grads = []

        for e_id, xi, eta in matches:
            grad, xi, eta = rectangle_gradient_at_point(element_data[e_id], U, point)
            grads.append(grad)

            conn = element_data[e_id]["connectivity"]
            print(f"  Element {e_id:>2}, nodes {conn}:")
            print(f"    local coords xi, eta = ({xi:.6f}, {eta:.6f})")
            print(f"    grad u              = [{grad[0]: .6f}, {grad[1]: .6f}]")

        if len(grads) > 1:
            avg = np.mean(np.vstack(grads), axis=0)
            print("  Average of adjacent element gradients:")
            print(f"    grad u_avg          = [{avg[0]: .6f}, {avg[1]: .6f}]")


def edge_on_line(p1, p2, line_type, value, tol=1e-10):
    if line_type == "x":
        return abs(p1[0] - value) < tol and abs(p2[0] - value) < tol
    if line_type == "y":
        return abs(p1[1] - value) < tol and abs(p2[1] - value) < tol
    raise ValueError("line_type must be 'x' or 'y'.")


def midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def report_flux_on_boundary_line(line_type, value, normal, nodes, element_data, U, k=1.0):
    """
    Reports flux at the midpoint of each boundary segment on the requested line.

    Since bilinear rectangular gradients vary inside the element,
    flux generally varies along the boundary.
    """
    normal = np.array(normal, dtype=float)

    print("\n" + "=" * 72)
    print(f"FLUX RESULTS ON {line_type} = {value}")
    print("=" * 72)
    print(f"Outward normal n = [{normal[0]:.1f}, {normal[1]:.1f}]")

    found = False

    for e_id, record in element_data.items():
        conn = record["connectivity"]

        # Rectangle local edges:
        # bottom: 1-2, right: 2-3, top: 3-4, left: 4-1
        local_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

        for a, b in local_edges:
            n1 = conn[a]
            n2 = conn[b]
            p1 = nodes[n1]
            p2 = nodes[n2]

            if edge_on_line(p1, p2, line_type, value):
                found = True
                pmid = midpoint(p1, p2)

                grad, xi, eta = rectangle_gradient_at_point(record, U, pmid)
                k_grad = k * grad

                math_normal_flux = float(np.dot(k_grad, normal))
                physical_outward_flux = -math_normal_flux

                print(f"\nBoundary edge nodes ({n1}, {n2}) adjacent to element {e_id}:")
                print(f"  midpoint            = ({pmid[0]:.3f}, {pmid[1]:.3f})")
                print(f"  local xi, eta       = ({xi:.3f}, {eta:.3f})")
                print(f"  grad u              = [{grad[0]: .6f}, {grad[1]: .6f}]")
                print(f"  k grad u            = [{k_grad[0]: .6f}, {k_grad[1]: .6f}]")
                print(f"  k grad u · n        = {math_normal_flux: .6f}")
                print(f"  -k grad u · n       = {physical_outward_flux: .6f}")

    if not found:
        print("No boundary edge found on this line.")


# ============================================================
# 6. Plotting
# ============================================================

def plot_1d_side_views(nodes, U):
    """
    1D side-view nodal plots.
    One separate figure per horizontal mesh line.
    Points only: no connecting lines.
    """
    rows_to_plot = [-1.0, 0.0, 1.0]

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
        ax.set_title(f"1D side-view nodal values along y = {y_value:g}")
        ax.grid(True)

        # Keep same visual scale across all side-view plots
        ax.set_ylim(-0.02, max(U) * 1.15)

        fig.tight_layout()

        if SAVE_FIGURES:
            filename = f"side_view_y_{int(y_value)}_points_only.png"
            fig.savefig(filename, dpi=300)

        figures.append((fig, ax))

    return figures


def plot_2d_nodal_map(nodes, elements, U):
    """
    2D nodal-value map with rectangular mesh.
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    # Draw rectangle outlines
    for conn in elements:
        poly_coords = np.array([nodes[nid] for nid in conn] + [nodes[conn[0]]])
        ax.plot(poly_coords[:, 0], poly_coords[:, 1], linewidth=1.0)

    xs = np.array([nodes[nid][0] for nid in sorted(nodes)])
    ys = np.array([nodes[nid][1] for nid in sorted(nodes)])
    us = np.array([U[nid - 1] for nid in sorted(nodes)])

    scatter = ax.scatter(xs, ys, c=us, s=90)
    fig.colorbar(scatter, ax=ax, label="u")

    for nid in sorted(nodes):
        x, y = nodes[nid]
        ui = U[nid - 1]
        ax.text(x + 0.03, y + 0.03, f"{nid}\n{ui:.3f}", fontsize=8)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Part B: 2D nodal map, bilinear rectangles")
    ax.grid(True)

    fig.tight_layout()

    if SAVE_FIGURES:
        fig.savefig("partB_2D_nodal_map.png", dpi=300)

    return fig, ax


def plot_2d_cell_average_map(nodes, elements, U):
    """
    2D cell-average plot over rectangular elements.
    This is a simple contour-style visualization for rectangular FEM.
    """
    patches = []
    values = []

    for conn in elements:
        coords = np.array([nodes[nid] for nid in conn])
        patches.append(Polygon(coords, closed=True))
        values.append(np.mean([U[nid - 1] for nid in conn]))

    fig, ax = plt.subplots(figsize=(7, 7))

    collection = PatchCollection(patches)
    collection.set_array(np.array(values))
    ax.add_collection(collection)

    fig.colorbar(collection, ax=ax, label="element-average u")

    for conn in elements:
        coords = np.array([nodes[nid] for nid in conn] + [nodes[conn[0]]])
        ax.plot(coords[:, 0], coords[:, 1], linewidth=0.8)

    ax.autoscale_view()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Part B: 2D cell-average map")
    ax.grid(True)

    fig.tight_layout()

    if SAVE_FIGURES:
        fig.savefig("partB_2D_cell_average_map.png", dpi=300)

    return fig, ax


def plot_3d_points_only(nodes, U):
    """
    3D nodal-value plot.
    Points only: no surface, no mesh lines.
    """
    xs = np.array([nodes[nid][0] for nid in sorted(nodes)])
    ys = np.array([nodes[nid][1] for nid in sorted(nodes)])
    us = np.array([U[nid - 1] for nid in sorted(nodes)])

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(xs, ys, us, s=60)

    for nid in sorted(nodes):
        x, y = nodes[nid]
        ui = U[nid - 1]
        ax.text(x, y, ui + 0.015, str(nid), fontsize=8)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u")
    ax.set_title("Part B: 3D nodal-value plot, points only")

    fig.tight_layout()

    if SAVE_FIGURES:
        fig.savefig("partB_3D_points_only.png", dpi=300)

    return fig, ax


# ============================================================
# 7. Main execution
# ============================================================

def main():
    nodes = build_nodes()
    elements = build_rectangular_elements()
    free_nodes, boundary_nodes = get_free_and_boundary_nodes()

    K, F, element_data = assemble_global_system(
        nodes,
        elements,
        k=K_MATERIAL,
        f=SOURCE_F,
    )

    U, Kff, Ff, Uf = solve_dirichlet_zero(K, F, free_nodes)

    print("=" * 72)
    print("CE526 Assignment 5 — Part B: 2D bilinear rectangular FEM")
    print("=" * 72)

    print("\nProblem:")
    print("  -Delta u = 1 in Omega")
    print("  u = 0 on all boundaries")
    print("  k = 1, f = 1")
    print("  Element type: 4-node bilinear rectangle, 1x1")

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

    # x = 2: outward normal is +x
    report_flux_on_boundary_line(
        line_type="x",
        value=2.0,
        normal=[1.0, 0.0],
        nodes=nodes,
        element_data=element_data,
        U=U,
        k=K_MATERIAL,
    )

    # y = -2: outward normal is -y
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
    plot_2d_nodal_map(nodes, elements, U)
    plot_2d_cell_average_map(nodes, elements, U)
    plot_3d_points_only(nodes, U)

    plt.show()


if __name__ == "__main__":
    main()