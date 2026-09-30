from geodesic_lbfgs.mesh import Mesh
import numpy as np

class Path:
    
    def __init__(self, mesh, source, target, face_sequence, lambdas):
        self.mesh = mesh
        self.source = np.asarray(source, dtype=float)
        self.target = np.asarray(target, dtype=float)
        self.face_sequence = np.asarray(face_sequence, dtype=int)
        self.lambdas = np.asarray(lambdas, dtype=float)

        if len(self.face_sequence) != len(self.lambdas) + 1:
            raise ValueError("The face sequence must contain one more element than lambdas.")

    # ordered index array of crossed edges along the path
    def crossed_edges(self):
        edges = []
        
        for i in range(len(self.face_sequence) - 1):
            first_face = self.face_sequence[i]
            second_face = self.face_sequence[i+1]
            
            edge = self.mesh.shared_edge(first_face, second_face)
            
            edges.append(edge)
            
        return np.asarray(edges, dtype=int)
    
    # ordered array of crossing points on the respective edges 
    # along the path
    # shape: (dim edges, 3)
    def crossing_points(self):
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
            return np.empty((0, 3))

        return np.asarray(points)
    
    # complete path: the crossing_points + source + target
    def points(self):
        crossings = self.crossing_points()
        return np.vstack((self.source, crossings, self.target))
        