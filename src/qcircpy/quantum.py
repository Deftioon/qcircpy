import numpy as np
import matplotlib.pyplot

from . import gates
from . import exceptions

class Circuit:
    def __init__(self, qubits: int, device: str = "cpu") -> None:
        self.device = device
        if device != "cpu" and device != "gpu":
            raise exceptions.DeviceError(device)
        elif device == "cpu":
            self.module = np
        
        self.matrix = self.module.identity(2 ** qubits, dtype=complex)
        self.qubits = qubits
        self.gates = []
    
    def __call__(self, state: str) -> np.ndarray:
        if len(state) != self.qubits:
            raise exceptions.StateError(state)
        if state[0] == "1":
            output = self.module.array([[0], [1]], dtype=complex)
        elif state[0] == "0":
            output = self.module.array([[1], [0]], dtype=complex)
        
        for i, bit in enumerate(state[1:]):
            if bit == "1":
                qubit = self.module.array([[0], [1]], dtype=complex)
            elif bit == "0":
                qubit = self.module.array([[1], [0]], dtype=complex)
            else:
                raise exceptions.StateError(state)
            output = self.module.kron(output, qubit)
        
        for gate in self.gates:
            output = gate @ output
        
        return output
        
    def __single_qubit_gate(self, gate: np.ndarray, qubit: int) -> np.ndarray:
        identity = self.module.identity(2, dtype=complex)

        if qubit == 0:
            applied_gate = gate
        else:
            applied_gate = identity
        for i in range(1, self.qubits):
            if i == qubit:
                applied_gate = self.module.kron(applied_gate, gate)
            else:
                applied_gate = self.module.kron(applied_gate, identity)
        return applied_gate
    
    def __double_qubit_gate(self, gate: np.ndarray, qubit1: int, qubit2: int) -> np.ndarray:
        identity = self.module.identity(2, dtype=complex)

        if qubit1 == 0:
            applied_gate = gate
        else:
            applied_gate = identity
        for i in range(1, self.qubits-1):
            if i == qubit1:
                applied_gate = self.module.kron(applied_gate, gate)

    def compile(self) -> np.ndarray:
        for gate in self.gates:
            self.matrix = gate @ self.matrix
        return self.matrix

    def measure(self, state: str) -> str:
        called = self(state)
        probabilities = np.abs(called) ** 2
        probabilities = probabilities.flatten()
        output_states = [str(i) for i in range(len(probabilities))]
        output = np.random.choice(output_states, p=probabilities)
        output = f"{bin(int(output))[2:]:0>{self.qubits}}"
        return output
    
    def hadamard(self, qubit: int) -> None:
        self.gates.append(self.__single_qubit_gate(gates.GATES[self.device]["HADAMARD"], qubit))

    def pauli_x(self, qubit: int) -> None:
        self.gates.append(self.__single_qubit_gate(gates.GATES[self.device]["PAULI_X"], qubit))
    
    def pauli_y(self, qubit: int) -> None:
        self.gates.append(self.__single_qubit_gate(gates.GATES[self.device]["PAULI_Y"], qubit))
    
    def pauli_z(self, qubit: int) -> None:
        self.gates.append(self.__single_qubit_gate(gates.GATES[self.device]["PAULI_Z"], qubit))
    
    def swap(self, qubit1: int, qubit2: int) -> None:
        self.gates.append(self.__double_qubit_gate(gates.GATES[self.device]["SWAP"], qubit1, qubit2))
