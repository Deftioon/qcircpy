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

class Gates:
    @staticmethod
    def hadamard():
        def gate(qubit):
            hadamard_matrix = qubit.module.asarray(GateMatrices.hadamard)

            for i in range(qubit.space - 1):
                hadamard_matrix = qubit.module.kron(hadamard_matrix, GateMatrices.hadamard)

            output = qubit.copy()
            data = hadamard_matrix @ qubit.data
            output.data = data

            return output
        return gate
    
    @staticmethod
    def x():
        def gate(qubit):
            x_matrix = qubit.module.asarray(GateMatrices.x)

            for i in range(qubit.space - 1):
                x_matrix = qubit.module.kron(x_matrix, GateMatrices.x)

            output = qubit.copy()
            data = x_matrix @ qubit.data
            output.data = data

            return output
        return gate
    
    @staticmethod
    def y():
        def gate(qubit):
            y_matrix = qubit.module.asarray(GateMatrices.y)

            for i in range(qubit.space - 1):
                y_matrix = qubit.module.kron(y_matrix, GateMatrices.y)

            output = qubit.copy()
            data = y_matrix @ qubit.data
            output.data = data

            return output
        return gate
    
    @staticmethod
    def z():
        def gate(qubit):
            z_matrix = qubit.module.asarray(GateMatrices.z)

            for i in range(qubit.space - 1):
                z_matrix = qubit.module.kron(z_matrix, GateMatrices.z)

            output = qubit.copy()
            data = z_matrix @ qubit.data
            output.data = data

            return output
        return gate
    
    @staticmethod
    def p(phi):
        def gate(qubit):
            p_matrix = qubit.module.asarray([[1, 0], [0, np.exp(1j * phi)]])

            for i in range(qubit.space - 1):
                p_matrix = qubit.module.kron(p_matrix, [[1, 0], [0, np.exp(1j * phi)]])

            output = qubit.copy()
            data = p_matrix @ qubit.data
            output.data = data

            return output
        return gate
    
    @staticmethod
    def t():
        def gate(qubit):
            t_matrix = qubit.module.asarray(GateMatrices.t)

            for i in range(qubit.space - 1):
                t_matrix = qubit.module.kron(t_matrix, GateMatrices.t)

            output = qubit.copy()
            data = t_matrix @ qubit.data
            output.data = data

            return output
        return gate

