"""Representation of points and paths on a triangle mesh."""

import numpy as np

class SurfacePoint:
    """Represent a point inside a specified mesh face."""

    def __init__(self, face, barycentric):
        """Construct a surface point from barycentric coordinates."""

        self.face = int(face)
        self.barycentric = np.asarray(barycentric, dtype=float,)

        if self.barycentric.shape != (3,):
            raise ValueError(
                "Barycentric coordinates must have shape (3,)."
            )

        if not np.all(np.isfinite(self.barycentric)):
            raise ValueError(
                "Barycentric coordinates must be finite."
            )

        if np.any(self.barycentric < 0.0):
            raise ValueError(
                "Barycentric coordinates cannot be negative."
            )

        if not np.isclose(self.barycentric.sum(), 1.0):
            raise ValueError(
                "Barycentric coordinates must sum to one."
            )

    def position(self, mesh):
        """Return the three-dimensional position on the mesh."""

        triangle_vertices = mesh.V[mesh.F[self.face]]

        return self.barycentric @ triangle_vertices

class Path:
    """Represent a piecewise-linear path on a triangle mesh."""
    
    def __init__(self, mesh, source, target, face_sequence, lambdas):
        """Construct a path from surface endpoints and edge parameters."""
        self.mesh = mesh
        self.source = source
        self.target = target
        face_sequence_array = np.asarray(face_sequence)

        if face_sequence_array.ndim != 1:
            raise ValueError("The face sequence must be one-dimensional.")

        if not np.issubdtype(face_sequence_array.dtype, np.integer):
            raise TypeError("The face sequence must contain integer indices.")

        self.face_sequence = face_sequence_array.astype(int, copy=True)
        self.lambdas = np.asarray(lambdas, dtype=float)

        if self.source.face != self.face_sequence[0]:
            raise ValueError(
                "The source must belong to the first face."
            )

        if self.target.face != self.face_sequence[-1]:
            raise ValueError(
                "The target must belong to the last face."
            )

        if self.lambdas.ndim != 1:
            raise ValueError("Lambdas must be one-dimensional.")

        if len(self.face_sequence) != len(self.lambdas) + 1:
            raise ValueError(
                "The face sequence must contain one more element "
                "than lambdas."
            )

        if np.any(self.face_sequence < 0):
            raise ValueError("Face indices cannot be negative.")

        if np.any(self.face_sequence >= len(self.mesh.F)):
            raise ValueError("A face index is out of range.")

        if not np.all(np.isfinite(self.lambdas)):
            raise ValueError("Lambdas must be finite.")

        if np.any(self.lambdas < 0.0) or np.any(self.lambdas > 1.0):
            raise ValueError("Each lambda must belong to [0, 1].")
        
        if self.source.shape != (3,):
            raise ValueError("Source must be a three-dimensional point.")

        if self.target.shape != (3,):
            raise ValueError("Target must be a three-dimensional point.")

        if not np.all(np.isfinite(self.source)):
            raise ValueError("Source coordinates must be finite.")

        if not np.all(np.isfinite(self.target)):
            raise ValueError("Target coordinates must be finite.")

    def crossed_edges(self):
        """Return the ordered indices of the crossed mesh edges."""

        edges = []
        
        for i in range(len(self.face_sequence) - 1):
            first_face = self.face_sequence[i]
            second_face = self.face_sequence[i + 1]
            
            edge = self.mesh.shared_edge(first_face, second_face)
            
            edges.append(edge)
            
        return np.asarray(edges, dtype=int)
    
    def crossing_points(self):
        """Return the ordered crossing points."""
        
        edges = self.crossed_edges()
        points = []

        for i in range(len(edges)):
            edge = edges[i]
            first_vertex = self.mesh.EV[edge, 0]
            second_vertex = self.mesh.EV[edge, 1]

            a = self.mesh.V[first_vertex]
            b = self.mesh.V[second_vertex]

            point = (1 - self.lambdas[i]) * a + self.lambdas[i] * b
            points.append(point)

        if len(points) == 0:
            return np.empty((0, 3), dtype=float)

        return np.asarray(points, dtype=float)
    
    def points(self):
        """Return the source, crossing points, and target in order."""
        
        source_position = self.source.position(self.mesh)
        target_position = self.target.position(self.mesh)
        crossings = self.crossing_points()
        
        return np.vstack((source_position, crossings, target_position))
        