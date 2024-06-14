import numpy as np
import cupy as cp
from dataclasses import dataclass

if __name__ == "__main__":
    import quantum as q
    from exceptions import *

else:
    from . import quantum as q
    from .exceptions import *


class GateMatrices:
    identity = np.array([[1, 0], [0, 1]])

    hadamard = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])

    x = np.array([[0, 1], [1, 0]])
    y = np.array([[0, -1j], [1j, 0]])
    z = np.array([[1, 0], [0, -1]])

    t = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])

    cnot = np.array([[1, 0, 0, 0], 
                     [0, 1, 0, 0], 
                     [0, 0, 0, 1], 
                     [0, 0, 1, 0]])
    
    cz = np.array([[1, 0, 0, 0], 
                   [0, 1, 0, 0], 
                   [0, 0, 1, 0], 
                   [0, 0, 0, -1]])
    
    swap = np.array([[1, 0, 0, 0], 
                     [0, 0, 1, 0], 
                     [0, 1, 0, 0], 
                     [0, 0, 0, 1]])
    
    ccnot = np.array([[1, 0, 0, 0, 0, 0, 0, 0], 
                      [0, 1, 0, 0, 0, 0, 0, 0], 
                      [0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0], 
                      [0, 0, 0, 0, 1, 0, 0, 0], 
                      [0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1], 
                      [0, 0, 0, 0, 0, 0, 1, 0]])

class Gate:
    def __init__(self, matrix, partition):
        self.matrix = matrix
        self.partition = partition

    def __call__(self, data):
        if not isinstance(data, q.Qubit):
            raise InvalidType("Gate is only defined for Qubit objects.")
        
        matrix = data.module.asarray(self.matrix)
        gate_matrix = data.module.asarray(self.matrix)

        for i in range(int(data.space/self.partition) - 1):
            gate_matrix = data.module.kron(gate_matrix, matrix)

        output = data.copy()
        data = gate_matrix @ data.data
        output.data = data

        return output

    def extend_matrix_space(self, space):
        output = self.matrix.copy()
        matrix = self.matrix
        for i in range(int(space/self.partition) - 1):
            output = np.kron(self.matrix, matrix)
        
        return output

identity = Gate(GateMatrices.identity, 1)
hadamard = Gate(GateMatrices.hadamard, 1)
x = Gate(GateMatrices.x, 1)
y = Gate(GateMatrices.y, 1)
z = Gate(GateMatrices.z, 1)
t = Gate(GateMatrices.t, 1)
cnot = Gate(GateMatrices.cnot, 2)
cz = Gate(GateMatrices.cz, 2)
swap = Gate(GateMatrices.swap, 2)
ccnot = Gate(GateMatrices.ccnot, 3)