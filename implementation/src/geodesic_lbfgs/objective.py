"""Length objective and gradient for a mesh path."""

import numpy as np

# [p1-s, p2-p1, ..., t-pk]
def path_segment_vectors(path) -> np.ndarray:
    """Return the ordered vectors of the path segments."""
    
    points = path.points()
    segment_vectors = points[1:] - points[:-1]
    return segment_vectors

def crossing_edge_vectors(path) -> np.ndarray:
    """Return the ordered direction vectors of the crossed edges."""
    
    edges = path.crossed_edges()
    edge_directions = []

    for edge in edges:
        edge_vertices = path.mesh.EV[edge]
        vertices = path.mesh.V[edge_vertices]

        edge_direction = vertices[1] - vertices[0]
        edge_directions.append(edge_direction)

    if len(edge_directions) == 0:
        return np.empty((0, 3), dtype=float)

    return np.asarray(edge_directions, dtype=float)

# length of the path L(face_sequence; lambdas)
def length(path):
    """Return the total length of the piecewise-linear path."""
    
    segment_vectors = path_segment_vectors(path)
    return float(np.linalg.norm(segment_vectors, axis=1).sum())

# gradient of L
# how the length change wrt to moving the crossing points
# along the crossed edges
def gradient(path) -> np.ndarray:
    """Return the derivative of path length with respect to lambdas."""

    partial_derivatives = []
    segment_vectors = path_segment_vectors(path)
    crossing_edge_directions = crossing_edge_vectors(path)

    for i in range(len(path.lambdas)):
        edge_direction = crossing_edge_directions[i]

        incoming_vector = segment_vectors[i]
        outgoing_vector = segment_vectors[i + 1]

        incoming_norm = np.linalg.norm(incoming_vector)
        outgoing_norm = np.linalg.norm(outgoing_vector)

        if incoming_norm == 0.0 or outgoing_norm == 0.0:
            raise ValueError(
                "The gradient is undefined for a zero-length segment."
            )

        incoming_unit_vector = incoming_vector / incoming_norm
        outgoing_unit_vector = outgoing_vector / outgoing_norm

        segment_length_change = (
            incoming_unit_vector - outgoing_unit_vector
        )

        partial_derivative = np.dot(
            edge_direction,
            segment_length_change,
        )

        partial_derivatives.append(partial_derivative)

    return np.asarray(partial_derivatives, dtype=float) 