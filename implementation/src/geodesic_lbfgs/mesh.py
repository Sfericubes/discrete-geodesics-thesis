import igl
import numpy as np


class Mesh:

    def __init__(self, V, F):
        self.V = np.asarray(V, dtype=float)
        self.F = np.asarray(F, dtype=int)

        self.EV, self.FE, self.EF = igl.edge_topology(self.V, self.F)

        self.TT, self.TTi = igl.triangle_triangle_adjacency(self.F)

        self.VF, _ = igl.vertex_triangle_adjacency_lists(self.F, len(self.V))

        self.boundary_edges = np.flatnonzero(np.any(self.EF == -1, axis=1))

    def shared_edge(self, first_face, second_face):
        local_edges = np.flatnonzero(self.TT[first_face] == second_face)

        if len(local_edges) == 0:
            raise ValueError("The faces do not share an edge.")

        local_edge = local_edges[0]

        return int(self.FE[first_face, local_edge])


def load_mesh(filename):
    V, F = igl.read_triangle_mesh(filename)
    return Mesh(V, F)