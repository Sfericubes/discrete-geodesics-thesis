# libigl cheat sheet for Liu et al.'s geodesic L-BFGS algorithm

**Scope:** Python implementation of *An optimization-driven approach for computing geodesic paths on triangle meshes* (Liu et al., 2017). Especially Sections 4.1-4.3, Figure 3, and Algorithm 1, printed pages 107-108.

**Implementation boundary:** libigl supplies mesh geometry and adjacency. You supply the path representation, length and gradient, vertex-event handling, and changes to the traversed face sequence. An optimizer supplies numerical steps.

**Verification:** core snippets checked with libigl **2.6.3**, Python 3.13.5, NumPy 2.2.4, and SciPy 1.15.3. The optional Polyscope snippet is documentation-based. Assumption: nondegenerate, consistently oriented, triangular manifold input. Boundary handling is an explicit algorithm choice.

## 1. The minimum to learn

1. **Mesh arrays:** `V` stores vertex positions; `F` stores triangle vertex indices.
2. **Edge topology:** `EV`, `FE`, `EF` connect vertices, edges, and faces.
3. **Face adjacency:** `TT`, `TTi` let you cross a shared edge.
4. **Vertex incidence:** incident faces plus adjacency let you traverse a vertex's triangle fan.
5. **Surface coordinates:** one scalar per crossed edge; barycentric coordinates for arbitrary endpoints.
6. **NumPy indexing:** turn topology indices into the coordinates used by the objective.
7. **Validation:** manifoldness, degenerate triangles, boundary sentinels, and valid face strips.
8. **Optional extensions:** `local_basis` for anisotropy; `exact_geodesic` for reference distances.

You do not need Laplacians, mass matrices, curvature estimation, parameterization, remeshing, or a mutable halfedge structure for the basic algorithm. The mesh stays fixed while the path changes.

## 2. Installation and Python conventions

```bash
python -m pip install libigl numpy scipy
```

```python
import igl
import numpy as np
from scipy.optimize import minimize
from importlib.metadata import version

print(version("libigl"))
help(igl.edge_topology)
```

- Distribution name: `libigl`. Import name: `igl`.
- Python functions usually return arrays instead of accepting C++ output arguments.
- Use `float64` coordinates and signed integer indices, here `int64`.
- Indices are zero-based. Python's `-1` selects the last array element, so check boundary sentinels before indexing.
- Shapes matter: `(3,)` is one vector; `(1, 3)` is a batch of one point.
- Binding docstrings occasionally retain C++ descriptions. Check the signature and actual return value, not just the prose. [Python API reference](https://libigl.github.io/libigl-python-bindings/api/igl/).

## 3. Geometry: `V` and `F`

```python
V, F = igl.read_triangle_mesh("mesh.off")
V = np.ascontiguousarray(V, dtype=np.float64)
F = np.ascontiguousarray(F, dtype=np.int64)

# V.shape == (n_vertices, 3)
# F.shape == (n_faces, 3)
triangles = V[F]                 # (n_faces, 3, 3)
triangle = V[F[0]]               # three positions of face 0
```

`F[f, c]` is the **global vertex index** at local corner `c` of face `f`. `V[F[f, c]]` is its 3D position. A corner number, vertex index, edge index, and face index are different objects even though all are integers.

For the examples below, use this square split along its diagonal:

```python
V = np.array([[0, 0, 0], [1, 0, 0],
              [1, 1, 0], [0, 1, 0]], dtype=np.float64)
F = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
```

The two triangles share edge `{0, 2}`. A path from vertex 1 to vertex 3 crosses that edge at its midpoint.

### Input checks before constructing topology

```python
assert V.ndim == 2 and V.shape[1] == 3 and len(V) > 0
assert F.ndim == 2 and F.shape[1] == 3 and len(F) > 0
assert np.isfinite(V).all()
assert F.min() >= 0 and F.max() < len(V)
assert np.all(np.diff(np.sort(F, axis=1), axis=1) > 0)
assert len(np.unique(np.sort(F, axis=1), axis=0)) == len(F)

scale = np.linalg.norm(np.ptp(V, axis=0))
assert scale > 0
# Example rejection threshold; choose for the mesh's resolution/quality.
assert np.all(igl.doublearea(V, F) > 1e-14 * scale**2)

edge_manifold, *_ = igl.is_edge_manifold(F)
assert edge_manifold
assert np.all(igl.is_vertex_manifold(F))
```

`doublearea` returns twice each triangle's area. The last two tests check edge and vertex incidence; also check consistent winding below. Do not use `if igl.is_edge_manifold(F):` in 2.6.3: a nonempty tuple is truthy even when its first element is `False`.

If you clean, weld, reorder, or remove vertices/faces, build all adjacency and path indices **after** that operation. Disconnected components are allowed, but the two endpoints must belong to the same component.

## 4. Edge topology: `EV`, `FE`, `EF`

```python
EV, FE, EF = igl.edge_topology(V, F)
```

- `EV`, shape `(n_edges, 2)`: `EV[e]` contains the endpoint vertex indices of edge `e`.
- `FE`, shape `(n_faces, 3)`: `FE[f]` contains the three global edge indices of face `f`.
- `EF`, shape `(n_edges, 2)`: `EF[e]` contains incident face indices; `-1` denotes a missing face.

```python
edge_id = 1
a_id, b_id = EV[edge_id]
a, b = V[a_id], V[b_id]
incident_faces = EF[edge_id][EF[edge_id] >= 0]
boundary_edges = np.flatnonzero(np.any(EF < 0, axis=1))
edge_vectors = V[EV[:, 1]] - V[EV[:, 0]]
edge_lengths = np.linalg.norm(edge_vectors, axis=1)
```

For the square, the verified arrays are:

```text
EV = [[0,1], [0,2], [0,3], [1,2], [2,3]]
FE = [[0,3,1], [1,4,2]]
EF = [[0,-1], [1,0], [-1,1], [0,-1], [1,-1]]
```

Do not infer a universal geometric left/right meaning from columns of `EF`. The ordering of `EV[e]` fixes the meaning of the crossing parameter: reversing its endpoints requires replacing `lambda` by `1 - lambda`.

### A lookup useful throughout the implementation

```python
edge_lookup = {
    tuple(sorted(map(int, pair))): e
    for e, pair in enumerate(EV)
}

def edge_between(u, v):
    return edge_lookup[tuple(sorted((int(u), int(v))))]

def shared_edge(f, g):
    common = np.intersect1d(F[f], F[g])
    if len(common) != 2:
        raise ValueError("Consecutive faces must share exactly one edge")
    return edge_between(*common)
```

`edge_topology` is sufficient here. `edge_flaps` is an alternative with additional corner information, but uses another local convention. Do not mix indices from separately generated edge arrays without mapping endpoint pairs. The [official edge-topology header](https://github.com/libigl/libigl/blob/main/include/igl/edge_topology.h) specifically flags `FE`'s convention as ambiguous.

## 5. Triangle adjacency: `TT`, `TTi`

```python
TT, TTi = igl.triangle_triangle_adjacency(F)
```

In the tested matrix-returning function, local edge `j` is:

```text
j = 0: corners (0,1)
j = 1: corners (1,2)
j = 2: corners (2,0)
```

- `TT[f, j]`: neighboring face across local edge `j`, or `-1` at a boundary.
- `TTi[f, j]`: local edge number in that neighboring face.
- For an interior neighbor `g = TT[f, j]`, `TT[g, TTi[f, j]] == f`.

```python
def across(f, j):
    g = int(TT[f, j])
    if g == -1:
        return None
    return g, int(TTi[f, j])

# Make the relation to global edges explicit.
face_edges = np.array([
    [edge_between(face[j], face[(j + 1) % 3]) for j in range(3)]
    for face in F
], dtype=np.int64)

for f in range(len(F)):
    for j in range(3):
        neighbor = across(f, j)
        if neighbor is None:
            continue
        g, h = neighbor
        assert TT[g, h] == f
        assert face_edges[f, j] == face_edges[g, h]
        # Shared directed edges must have opposite winding.
        assert F[f, j] == F[g, (h + 1) % 3]
        assert F[f, (j + 1) % 3] == F[g, h]
```

**Convention trap:** `igl.edge_lengths(V, F)` uses edges **opposite** corners: `(1,2)`, `(2,0)`, `(0,1)`. Its column `j` does not match matrix `TT`'s column `j`. List-based triangle adjacency also has a different convention. This sheet uses only the matrix form. [Official adjacency header](https://github.com/libigl/libigl/blob/main/include/igl/triangle_triangle_adjacency.h).

## 6. Vertex incidence and the ordered triangle fan

```python
VF, VI = igl.vertex_triangle_adjacency_lists(F, len(V))

v = 0
for f, c in zip(VF[v], VI[v]):
    assert F[f, c] == v
```

- `VF[v]`: incident face indices.
- `VI[v]`: the corresponding local corner numbers.
- These incidence lists are **not the cyclic triangle fan** required for a strip update.

In 2.6.3, the similarly named flat API returns a different representation:

```python
VF_flat, offsets = igl.vertex_triangle_adjacency(F, len(V))
incident_to_v = VF_flat[offsets[v]:offsets[v + 1]]
```

The second array here contains offsets, despite inherited docstring prose mentioning local corners.

### How to walk around a vertex

If vertex `v` occupies corner `c` of face `f`, the two local edges incident to it are `c` and `(c + 2) % 3`. Crossing either edge with `TT` enters the next face around `v`.

```python
def fan_neighbors(v, f):
    corners = np.flatnonzero(F[f] == v)
    if len(corners) != 1:
        raise ValueError("Expected one occurrence of the vertex in this face")
    c = int(corners[0])
    return [int(TT[f, j]) for j in (c, (c + 2) % 3)
            if TT[f, j] >= 0]
```

To order the fan, start in an incident face, choose one incident edge to cross, then repeatedly take the neighbor different from the previous face. Stop when you return to the starting face or reach a boundary. Track visited faces to detect invalid input.

- Interior manifold vertex: cyclic fan, with two directions between distinct incident faces.
- Boundary manifold vertex: open fan; an alternative route through missing faces does not exist.
- Nonmanifold vertex: this two-sided model is insufficient.

This is the topology needed for Figure 3. Vertex-neighbor coordinates alone cannot identify the correct sequence of faces.

## 7. Represent the path separately from the mesh

For `k` edge crossings, store:

- `face_path`, shape `(k + 1,)`: ordered global face indices.
- `crossed_edges`, shape `(k,)`: ordered global edge indices.
- `lam`, shape `(k,)`: parameters in `[0, 1]`.
- `s`, `t`, shape `(3,)`: fixed endpoint coordinates, with their supporting faces recorded.

The order and repeated **occurrences** matter. A path may revisit an edge or face; do not reduce its sequence to a set or a dictionary keyed only by global edge index.

```python
face_path = np.array([0, 1], dtype=np.int64)
crossed_edges = np.array([
    shared_edge(f, g) for f, g in zip(face_path[:-1], face_path[1:])
], dtype=np.int64)
lam = np.full(len(crossed_edges), 0.25)
s, t = V[1].copy(), V[3].copy()

A = V[EV[crossed_edges, 0]]       # (k, 3)
D = V[EV[crossed_edges, 1]] - A   # (k, 3)
P = A + lam[:, None] * D         # (k, 3)
Q = np.vstack((s, P, t))         # (k+2, 3)
```

Mathematically, with $a_i,b_i$ the ordered endpoints of crossed edge $i$:

$$
p_i=(1-\lambda_i)a_i+\lambda_i b_i,
\qquad \frac{\partial p_i}{\partial\lambda_i}=b_i-a_i.
$$

`lam[:, None]` changes `(k,)` to `(k,1)`, allowing one scalar to multiply each 3D edge vector.

**Strip invariant:** segment `Q[j] -> Q[j+1]` lies in `face_path[j]`. Both endpoints must belong to that triangle; convexity then keeps the entire segment inside it. `s` belongs to the first face, `t` to the last, and each crossing belongs to both adjacent faces.

An arbitrary sequence of nearest points projected onto the mesh does not establish this invariant.

## 8. Arbitrary endpoints and barycentric coordinates

A point in face `f` can be stored as `(f, beta)`:

$$
q=\beta_0V_{F_{f,0}}+\beta_1V_{F_{f,1}}+\beta_2V_{F_{f,2}},
\qquad \sum_c\beta_c=1,\quad \beta_c\geq0.
$$

```python
f = 0
beta = np.array([0.2, 0.3, 0.5])
q = beta @ V[F[f]]
tri = V[F[f]]
recovered = igl.barycentric_coordinates(
    q[None, :], tri[0:1], tri[1:2], tri[2:3]
)[0]
assert np.allclose(recovered, beta)
```

For a 3D query point, nearest-surface projection gives a position and a supporting face:

```python
queries = np.array([[0.8, 0.2, 0.1]])
squared_distance, face_ids, closest = igl.point_mesh_squared_distance(
    queries, V, F
)
```

This computes Euclidean proximity, not a surface geodesic. At an edge or vertex, more than one supporting face can be valid. Preserve the face/strip context when choosing one. For membership checks, test both barycentric nonnegativity and reconstruction residual, because barycentric coordinates alone do not certify that a 3D point lies in the face plane.

## 9. Length and gradient: NumPy, not `igl.grad`

Let $q_0=s$, $q_i=p_i$ for $1\leq i\leq k$, and $q_{k+1}=t$. Define

$$
r_j=q_{j+1}-q_j,\qquad \ell_j=\|r_j\|,\qquad u_j=r_j/\ell_j.
$$

Then the paper's Equations (1)-(2) become

$$
L=\sum_{j=0}^{k}\ell_j,
\qquad
\frac{\partial L}{\partial\lambda_i}
=(b_i-a_i)^T(u_{i-1}-u_i),\quad 1\leq i\leq k.
$$

Only the incoming and outgoing segments depend on each crossing. This also handles `k == 1` without separate endpoint cases.

```python
def length_and_gradient(lam, A, D, s, t):
    P = A + lam[:, None] * D
    Q = np.vstack((s, P, t))
    R = np.diff(Q, axis=0)
    lengths = np.linalg.norm(R, axis=1)
    if np.any(lengths == 0):
        raise ValueError("Zero-length segment: resolve the vertex event first")
    U = R / lengths[:, None]
    gradient = np.einsum("ij,ij->i", D, U[:-1] - U[1:])
    return float(lengths.sum()), gradient
```

- Each evaluation costs $O(k)$, independent of the total mesh size after preprocessing.
- `igl.grad(V, F)` differentiates a piecewise-linear scalar field over the mesh. It does not differentiate this objective with respect to `lam`.
- Near-zero segments require a scale-aware event policy. Replacing denominators by an epsilon generally makes the supplied gradient inconsistent with the original objective.
- A smooth substitute such as $\sqrt{r^Tr+\delta^2}$ is possible, but changes the objective; derive its gradient consistently and label the approximation.
- For fixed strip and ordinary length, the objective is convex in `lam`, since it is a sum of norms of affine functions. It may be nonsmooth or have nonunique minimizers. Choosing the strip remains a separate problem.

### Fixed-strip optimization prototype

```python
result = minimize(
    length_and_gradient,
    lam,
    args=(A, D, s, t),
    method="L-BFGS-B",
    jac=True,                    # function returns (value, gradient)
    bounds=[(0.0, 1.0)] * len(lam),
    options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-9},
)
lam = result.x
Q = np.vstack((s, A + lam[:, None] * D, t))
print(result.success, result.message, result.fun)
```

For the square example: `lam ≈ [0.5]`, `result.fun ≈ sqrt(2)`.

Handle `k == 0` directly: if both endpoints belong to the same face, their connecting segment needs no optimizer. SciPy's `gtol` uses a projected gradient; `ftol` tests relative objective reduction. Neither certifies that another strip cannot improve the path. [SciPy L-BFGS-B documentation](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html).

## 10. Vertex events: the part you must implement

**The preceding call is a fixed-strip prototype, not Algorithm 1 in full.** Section 4.2 describes L-BFGS steps, clamping parameters to `[0,1]`, and updating the face sequence as vertex events occur. Solving each strip completely with L-BFGS-B and then switching strips changes that iteration scheme.

### Detect the event

```python
parameter_tol = 1e-10  # dimensionless example tolerance
at_start = np.flatnonzero(lam <= parameter_tol)
at_end = np.flatnonzero(lam >= 1 - parameter_tol)
hit_vertex_ids = np.concatenate((
    EV[crossed_edges[at_start], 0],
    EV[crossed_edges[at_end], 1],
))
```

For a geometric tolerance, test `lam[i] * edge_length[i]` or `(1-lam[i]) * edge_length[i]` against a distance tolerance. Parameter tolerances alone imply different physical tolerances on short and long edges.

### Required local update

1. Locate the **occurrence** of the hit vertex along the path. Group consecutive crossings that coincide there; keep nonconsecutive visits distinct.
2. Identify the entry and exit faces of the affected strip section.
3. Traverse the vertex fan using `TT` and vertex incidence. For the simple interior case, replace the current fan arc by the complementary arc. This moves the vertex to the opposite side of the strip, as in Figure 3.
4. Rebuild the local face sequence, crossed edges, and crossing parameters. Preserve unaffected path portions and fixed endpoints.
5. Rebuild `A`, `D`, and any per-segment weights or metrics. The number of variables can change.
6. Establish a valid, nondegenerate next path and evaluate its actual objective. Handle repeated events and termination explicitly.

**Do not flip mesh edges:** Figure 3 changes the faces traversed by the path; `V`, `F`, and their static adjacency remain unchanged.

**Do not differentiate repeated coincident crossings:** inserting the vertex on several new edges produces zero-length segments. The paper does not specify a complete numerical policy for this case. A concrete implementation must choose and test an event-aware or regularized treatment; blindly assigning the vertex to every new crossing and calling the smooth gradient above is invalid.

**Optimizer state:** do not change the dimension or meaning of `lam` inside an active SciPy objective evaluation. End the current solve before replacing its parameterization. Restarting L-BFGS memory after such a change is a conservative implementation choice; preserving it requires a justified mapping. Repeated `maxiter=1` calls also discard history and are not equivalent to persistent L-BFGS iterations.

**Boundaries:** a closed-mesh implementation can reject boundary meshes initially. Supporting boundaries needs a rule for constrained paths along the boundary and events at boundary vertices. Never cross `TT == -1`.

The paper's method depends on initialization and does not guarantee the globally shortest route. Its stated convergence behavior should not be automatically attributed to a modified SciPy/event-handling implementation.

## 11. Obtain an initial path

For the first implementation, supply a known valid face strip and interior edge parameters. Any consecutive crossing points lying in the same triangle can be joined by a valid segment.

For a vertex-to-vertex graph initialization:

```python
VV = igl.adjacency_list(F)
distance, previous = igl.dijkstra(V, VV, 1, {3})
if not np.isfinite(distance[3]):
    raise ValueError("Endpoints are disconnected")
vertex_path = igl.dijkstra_backtrack(3, previous.reshape(-1, 1))[::-1]
```

The tested backtracking API expects a column-shaped predecessor array and returns target-to-source order, hence the reversal.

This path runs along mesh edges. It still needs conversion into an ordered face strip, with choices at vertex fans. Because it initially contains vertex events, it is a more difficult starting case than a prescribed strip. A heat-method distance field also requires a separate path-tracing procedure. Neither directly supplies Liu's path state.

## 12. Optional extensions from the paper

### Per-face density: Section 4.3, Equations (3)-(4)

Store positive `rho`, shape `(n_faces,)`. Segment `j` uses `rho[face_path[j]]`, not a value indexed by its crossed edge.

$$
L_\rho=\sum_{j=0}^{k}\rho_{f_j}\ell_j,
\qquad
\frac{\partial L_\rho}{\partial\lambda_i}
=(b_i-a_i)^T(\rho_{f_{i-1}}u_{i-1}-\rho_{f_i}u_i).
$$

In the objective above, compute `weights = rho[face_path]`, replace the value by `weights @ lengths`, and replace `U` by `weights[:, None] * U`. This assumes density is constant within each triangle. A continuously varying density requires a different integral and derivative.

### Anisotropy: local tangent coordinates

```python
B1, B2, N = igl.local_basis(V, F)  # each has shape (n_faces, 3)
B = np.stack((B1, B2), axis=2)    # (n_faces, 3, 2)
```

The columns of $B_f$ form an orthonormal tangent basis. Convert a 3D face segment $r$ into a 2D tangent vector $x=B_f^Tr$. Then use a face-specific $2\times2$ transform $T_f$:

$$
\ell_f(r)=\|T_fB_f^Tr\|,
\qquad
\nabla_r\ell_f
=B_fT_f^T\frac{T_fB_f^Tr}{\|T_fB_f^Tr\|}.
$$

Use this 3D segment derivative in place of `U` before taking the difference between incoming and outgoing contributions. Include density as an additional scalar factor if needed. [libigl local-basis definition](https://github.com/libigl/libigl/blob/main/include/igl/local_basis.h).

**Derived implementation correction:** for the literal linear-transform objective in Equation (5), the chain rule requires $T_f^T$. Equation (6), as printed, omits this factor and leaves the 2D-to-3D mapping implicit. Use the dimensionally explicit derivative above and check it numerically.

If instead you store the symmetric positive-definite metric tensor $G_f$, use

$$
\ell_f(r)=\sqrt{x^TG_fx},
\qquad
\nabla_r\ell_f=B_f\frac{G_fx}{\sqrt{x^TG_fx}},
\qquad G_f=T_f^TT_f.
$$

Multiplying by `G` and taking a Euclidean norm would apply the metric twice. The orientation of the tangent basis and the coordinates of the metric must agree.

### Additional constraints: Section 4.1

For residual vector $H(\lambda)$, the quadratic penalty and its gradient are

$$
E=L+\mu\|H\|^2,
\qquad \nabla E=\nabla L+2\mu J_H^TH.
$$

libigl does not construct these residuals. For a coplanarity residual $H_i=n^Tp_i-d$, the derivative is $n^T(b_i-a_i)$. Monitor constraint residuals separately: a finite penalty parameter does not enforce exact equality.

### Closed curves

Use cyclic segment connectivity and cyclic gradient contributions. The open-path code fixes `s` and `t`; duplicating the first node at the end without changing the variables and derivatives does not implement a freely moving closed curve.

## 13. Validation and visualization

### Exact distance benchmark

```python
empty = np.empty(0, dtype=np.int64)
reference = igl.exact_geodesic(
    V, F,
    np.array([1], dtype=np.int64), empty,  # source vertices, source faces
    np.array([3], dtype=np.int64), empty,  # target vertices, target faces
)[0]
```

This returns a **distance**, not the crossing sequence needed to initialize or update Liu's algorithm. Use it to benchmark the ordinary isotropic, unweighted, unconstrained problem with the same endpoints. It is not a reference for density-weighted or anisotropic length. [libigl exact-geodesic documentation](https://libigl.github.io/dox/exact__geodesic_8h.html).

For valid paths, $L\geq d_{\mathrm{exact}}$ up to numerical tolerance. A larger value can indicate a longer locally selected route; it does not by itself prove an implementation error. On the square above, both values should equal $\sqrt{2}$.

### Checks that reveal real implementation errors

1. **Gradient:** compare analytic derivatives with central differences at interior parameters, away from zero-length segments and topology changes.
2. **Surface validity:** certify each segment against its assigned triangle, including its plane.
3. **Topology:** verify shared edges, reciprocal `TT/TTi`, fan traversal, and boundary handling on tiny meshes.
4. **Vertex updates:** construct a path forced to hit an interior vertex; verify sequence replacement and consecutive coincident-crossing handling.
5. **Solver behavior:** record objective, projected gradient, stop reason, crossing count, and vertex events. An iteration limit is not convergence.
6. **Scale:** check that tolerances behave predictably when the entire mesh is rescaled.

### Optional Polyscope display

```python
import polyscope as ps

ps.init()
ps.register_surface_mesh("mesh", V, F)
ps.register_curve_network("path", Q, edges="line")
ps.show()
```

With unchanged crossing count, update node positions; with changed path connectivity/count, register the curve again. [Polyscope curve-network documentation](https://polyscope.run/py/structures/curve_network/basics/).

The mesh arrays can be saved with `igl.write_triangle_mesh("mesh.off", V, F)`. Save path state separately, for example with `np.savez`, including the face sequence, crossed edges, parameters, endpoints, and the identity of the mesh used.

## 14. Implementation order

1. Load a small mesh, validate it, and inspect `V`, `F`, `EV`, `EF`, and `TT`.
2. Construct a known face strip; reconstruct and display its path from `lam`.
3. Verify the length gradient; solve the fixed-strip square example.
4. Implement and verify ordered vertex-fan traversal.
5. Implement vertex events, strip replacement, and optimizer restarts with an explicit zero-segment policy.
6. Add general initialization and larger-mesh benchmarks.
7. Add density, anisotropy, constraints, or closed curves only when needed.

**Paper-to-code map:** Section 4.1 -> path coordinates and objective; Section 4.2/Figure 3/Algorithm 1 -> optimization plus vertex-event updates; Section 4.3 -> face weights and tangent metrics; Section 5.3 -> additional residuals and cyclic paths.
