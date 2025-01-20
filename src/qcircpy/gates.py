import numpy as np
import cupy as cp

GATES = {
    "cpu": {
        "HADAMARD": np.array([[1, 1], 
                              [1, -1]], dtype=complex) / np.sqrt(2),

        "PAULI_X": np.array([[0, 1], 
                             [1, 0]], dtype=complex),

        "PAULI_Y": np.array([[0, -1j], 
                             [1j, 0]], dtype=complex),

        "PAULI_Z": np.array([[1, 0], 
                             [0, -1]], dtype=complex),

        "CNOT": np.array([[1, 0, 0, 0], 
                          [0, 1, 0, 0], 
                          [0, 0, 0, 1], 
                          [0, 0, 1, 0]], dtype=complex),

        "SWAP": np.array([[1, 0, 0, 0], 
                          [0, 0, 1, 0], 
                          [0, 1, 0, 0], 
                          [0, 0, 0, 1]], dtype=complex),

        "TOFFOLI": np.array([[1, 0, 0, 0, 0, 0, 0, 0], 
                             [0, 1, 0, 0, 0, 0, 0, 0], 
                             [0, 0, 1, 0, 0, 0, 0, 0], 
                             [0, 0, 0, 1, 0, 0, 0, 0], 
                             [0, 0, 0, 0, 1, 0, 0, 0], 
                             [0, 0, 0, 0, 0, 1, 0, 0], 
                             [0, 0, 0, 0, 0, 0, 0, 1], 
                             [0, 0, 0, 0, 0, 0, 1, 0]], dtype=complex),
        
        "CSWAP": np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1]], dtype=complex),
    },
    "gpu": {
        "HADAMARD": cp.array([[1, 1], 
                              [1, -1]], dtype=complex) / np.sqrt(2),

        "PAULI_X": cp.array([[0, 1], 
                             [1, 0]], dtype=complex),

        "PAULI_Y": cp.array([[0, -1j], 
                             [1j, 0]], dtype=complex),

        "PAULI_Z": cp.array([[1, 0], 
                             [0, -1]], dtype=complex),

        "CNOT": cp.array([[1, 0, 0, 0], 
                          [0, 1, 0, 0], 
                          [0, 0, 0, 1], 
                          [0, 0, 1, 0]], dtype=complex),

        "SWAP": cp.array([[1, 0, 0, 0], 
                          [0, 0, 1, 0], 
                          [0, 1, 0, 0], 
                          [0, 0, 0, 1]], dtype=complex),

        "TOFFOLI": cp.array([[1, 0, 0, 0, 0, 0, 0, 0], 
                             [0, 1, 0, 0, 0, 0, 0, 0], 
                             [0, 0, 1, 0, 0, 0, 0, 0], 
                             [0, 0, 0, 1, 0, 0, 0, 0], 
                             [0, 0, 0, 0, 1, 0, 0, 0], 
                             [0, 0, 0, 0, 0, 1, 0, 0], 
                             [0, 0, 0, 0, 0, 0, 0, 1], 
                             [0, 0, 0, 0, 0, 0, 1, 0]], dtype=complex),
        
        "CSWAP": cp.array([[1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1]], dtype=complex), 
    }
}