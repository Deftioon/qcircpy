import numpy as np

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
        
        "CY": np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, -1j],
                        [0, 0, 1j, 0]], dtype=complex),
        
        "CZ": np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, -1]], dtype=complex),
    },
}

def wrap_in_controls_upwards(gate: np.ndarray) -> np.ndarray:
    identity = np.eye(gate.shape[0], dtype=complex)
    top = np.kron(np.array([[1, 0], [0, 0]], dtype=complex), identity)
    bottom = np.kron(np.array([[0, 0], [0, 1]], dtype=complex), gate)
    return top + bottom

def wrap_in_controls_downwards(gate: np.ndarray) -> np.ndarray:
    identity = np.eye(gate.shape[0], dtype=complex)
    top = np.kron(np.array([[0, 0], [0, 1]], dtype=complex), gate)
    bottom = np.kron(np.array([[1, 0], [0, 0]], dtype=complex), identity)
    return top + bottom


# TODO: Implement the controlled_gate function
def controlled_gate(qubits: int, gate: np.ndarray, target: int, *controls: int) -> np.ndarray:
    identity = np.eye(2, dtype=complex)
    applied_gate = gate
    controls = sorted(list(controls))
    print(controls)

    for qubit in range(controls[0], target):
        print(qubit)
        if qubit not in controls:
            applied_gate = np.kron(applied_gate, identity)
        else:
            applied_gate = wrap_in_controls_downwards(applied_gate)
    
    for i in range(0, controls[0]):
        applied_gate = np.kron(identity, applied_gate)
    
    for i in range(target+1, qubits):
        applied_gate = np.kron(applied_gate, identity)
    
    return applied_gate