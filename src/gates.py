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
        def gate(data):
            if isinstance(data, q.Qubit):
                Matrix = data.module.asarray(GateMatrices.hadamard)
                hadamard_matrix = data.module.asarray(GateMatrices.hadamard)

                for i in range(data.space - 1):
                    hadamard_matrix = data.module.kron(hadamard_matrix, Matrix)

                output = data.copy()
                data = hadamard_matrix @ data.data
                output.data = data

                return output
        return gate
    
    @staticmethod
    def x():
        def gate(data):
            if isinstance(data, q.Qubit):
                Matrix = data.module.asarray(GateMatrices.x)
                x_matrix = data.module.asarray(GateMatrices.x)

                for i in range(data.space - 1):
                    x_matrix = data.module.kron(x_matrix, Matrix)

                output = data.copy()
                data = x_matrix @ data.data
                output.data = data

                return output
        return gate
    
    @staticmethod
    def y():
        def gate(data):
            if isinstance(data, q.Qubit):
                Matrix = data.module.asarray(GateMatrices.y)
                y_matrix = data.module.asarray(GateMatrices.y)

                for i in range(data.space - 1):
                    y_matrix = data.module.kron(y_matrix, Matrix)

                output = data.copy()
                data = y_matrix @ data.data
                output.data = data

                return output
        return gate
    
    @staticmethod
    def z():
        def gate(data):
            if isinstance(data, q.Qubit):
                Matrix = data.module.asarray(GateMatrices.z)
                z_matrix = data.module.asarray(GateMatrices.z)

                for i in range(data.space - 1):
                    z_matrix = data.module.kron(z_matrix, Matrix)

                output = data.copy()
                data = z_matrix @ data.data
                output.data = data

                return output
        return gate
    
    @staticmethod
    def p(phi):
        def gate(data):
            if isinstance(data, q.Qubit):
                Matrix = data.module.asarray([[1, 0], [0, np.exp(1j * phi)]])
                p_matrix = data.module.asarray([[1, 0], [0, np.exp(1j * phi)]])

                for i in range(data.space - 1):
                    p_matrix = data.module.kron(p_matrix, Matrix)

                output = data.copy()
                data = p_matrix @ data.data
                output.data = data

                return output
        return gate
    
    @staticmethod
    def t():
        def gate(data):
            if isinstance(data, q.Qubit):
                Matrix = data.module.asarray(GateMatrices.t)
                t_matrix = data.module.asarray(GateMatrices.t)

                for i in range(data.space - 1):
                    t_matrix = data.module.kron(t_matrix, Matrix)

                output = data.copy()
                data = t_matrix @ data.data
                output.data = data

                return output
        return gate
    
    @staticmethod
    def swap():
        def gate(data):
            if not isinstance(data, q.Qubit):
                raise InvalidType("Swap gate is only defined for Qubit objects.")

            if data.space < 2:
                raise InvalidGate("CNOT gate is not defined for less than 2 qubits.")

            if data.space % 2  != 0:
                raise InvalidGate("Swap gate is not defined for odd number of qubits.")
            
            Matrix = data.module.asarray(GateMatrices.swap)
            swap_matrix = data.module.asarray(GateMatrices.swap)

            for i in range(int(data.space / 2) - 1):
                swap_matrix = data.module.kron(swap_matrix, Matrix)

            output = data.copy()
            data = swap_matrix @ data.data
            output.data = data

            return output
        return gate

    @staticmethod
    def cnot():
        def gate(data):
            if not isinstance(data, q.Qubit):
                raise InvalidType("CNOT gate is only defined for Qubit objects.")

            if data.space < 2:
                raise InvalidGate("CNOT gate is not defined for less than 2 qubits.")
            
            if data.space % 2  != 0:
                raise InvalidGate("CNOT gate is not defined for odd number of qubits.")
            
            Matrix = data.module.asarray(GateMatrices.cnot)
            cnot_matrix = data.module.asarray(GateMatrices.cnot)

            for i in range(int(data.space / 2) - 1):
                cnot_matrix = data.module.kron(cnot_matrix, Matrix)


            output = data.copy()
            data = cnot_matrix @ data.data
            output.data = data

            return output
        return gate
    
    @staticmethod
    def ccnot():
        def gate(data):
            if not isinstance(data, q.Qubit):
                raise InvalidType("CCNOT gate is only defined for Qubit objects.")
            
            if data.space < 3:
                raise InvalidGate("CCNOT gate is not defined for less than 3 qubits.")
            
            if data.space % 3  != 0:
                raise InvalidGate("CCNOT gate is not defined for non-3-factorable number of qubits.")
            
            Matrix = data.module.asarray(GateMatrices.ccnot)
            ccnot_matrix = data.module.asarray(GateMatrices.ccnot)

            for i in range(int(data.space / 3) - 1):
                ccnot_matrix = data.module.kron(ccnot_matrix, Matrix)

            output = data.copy()
            data = ccnot_matrix @ data.data
            output.data = data

            return output
        return gate