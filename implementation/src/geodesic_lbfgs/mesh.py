"""Triangle-mesh geometry and topology."""

import igl
import numpy as np

class Mesh:
    """Store a fixed triangular mesh and its connectivity."""

    def __init__(self, V, F):
        """Construct a mesh from vertex coordinates and triangle indices."""
        self.V = np.asarray(V, dtype=float).copy()
        faces = np.asarray(F)

        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("F must have shape (number_of_faces, 3).")

        if not np.issubdtype(faces.dtype, np.integer):
            raise TypeError("F must contain integer vertex indices.")

        self.F = faces.astype(int, copy=True)

        if self.V.ndim != 2 or self.V.shape[1] != 3:
            raise ValueError("V must have shape (number_of_vertices, 3).")

        if not np.all(np.isfinite(self.V)):
            raise ValueError("Vertex coordinates must be finite.")

        if np.any(self.F < 0) or np.any(self.F >= len(self.V)):
            raise ValueError("Face vertex indices are out of range.")

        for face in self.F:
            if len(set(face)) != 3:
                raise ValueError("A triangle contains repeated vertices.")

        self.EV, self.FE, self.EF = igl.edge_topology(self.V, self.F)

        self.TT, self.TTi = igl.triangle_triangle_adjacency(self.F)

        self.VF, _ = igl.vertex_triangle_adjacency_lists(self.F, len(self.V))

        self.boundary_edges = np.flatnonzero(np.any(self.EF == -1, axis=1))
        
        self.V.setflags(write=False)
        self.F.setflags(write=False)

    def shared_edge(self, first_face, second_face):
        """Return the global edge shared by two adjacent faces."""
    
        number_of_faces = len(self.F)
    
        if first_face < 0 or first_face >= number_of_faces:
            raise IndexError("The first face index is out of range.")
    
        if second_face < 0 or second_face >= number_of_faces:
            raise IndexError("The second face index is out of range.")
    
        local_edges = np.flatnonzero(
            self.TT[first_face] == second_face
        )
    
        if len(local_edges) == 0:
            raise ValueError("The faces do not share an edge.")
    
        local_edge = local_edges[0]
    
        return int(self.FE[first_face, local_edge])


def load_mesh(filename):
    """Load a triangular mesh from a file."""
    
    V, F = igl.read_triangle_mesh(filename)
    return Mesh(V, F)