import numpy as np
from geodesic_lbfgs.path import Path

# ordered array containing each segment composing the path
# [p1-s, p2-p1, ..., t-pk]
def path_segments(path: Path) -> np.ndarray:
    points = path.points()
    segment_vectors = points[1:] - points[:-1]
    return segment_vectors

# what is the order of the vertices topologically?
def crossing_edge_vectors(path: Path) -> np.ndarray:
    edges = path.crossed_edges()
    edge_directions = []

    for edge in edges:
        edge_vertices = path.mesh.EV[edge]
        vertices = path.mesh.V[edge_vertices]

        edge_direction = vertices[1] - vertices[0]
        edge_directions.append(edge_direction)

    return np.asarray(edge_directions, dtype=float)

# length of the path L(face_sequence; lambdas)
def length(path):
    segment_vectors = path_segments(path)
    return float(np.linalg.norm(segment_vectors, axis=1).sum())

# gradient of L
# how the length change wrt to moving the crossing points
# along the crossed edges
def gradient(path: Path) -> np.ndarray:
    partial_derivatives = []
    segment_vectors = path_segments(path)
    crossing_edges_direction = crossing_edge_vectors(path)
    
    for i in range(len(path.lambdas)):
            edge_direction = crossing_edges_direction[i]
            
            incoming_unit_vector = (
                segment_vectors[i] / np.linalg.norm(segment_vectors[i])
            )
            outgoing_unit_vector = (
                segment_vectors[i+1] / np.linalg.norm(segment_vectors[i+1])
            )
            
            segment_length_change = (
                incoming_unit_vector - outgoing_unit_vector
            )
                
            partial_derivative = np.dot(
            edge_direction,
            segment_length_change,
            )
            
            partial_derivatives.append(partial_derivative)
    
    return np.asarray(partial_derivatives, dtype=float)  