from .exceptions import *
from . import gates

import numpy as np
import matplotlib.pyplot as plt

class Runtime:
    def __init__(self, state: str = "0", device: str = "cpu") -> None:
        self.space = 2 ** len(state)
        self.qubits = len(state)
        self.device = device
        if device != "cpu" and device != "gpu":
            raise DeviceError(device)
        elif device == "cpu":
            self.module = np
        
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
                raise StateError(state)
            output = self.module.kron(output, qubit)
        
        self.__state = output
    
    def measure(self) -> str:
        probabilities = np.abs(self.__state) ** 2
        probabilities = probabilities.flatten()
        output_states = [str(i) for i in range(len(probabilities))]
        output = np.random.choice(output_states, p=probabilities)
        self.__state = self.module.zeros((2 ** self.qubits, 1), dtype=complex)
        self.__state[int(output),0] = 1
        output = f"{bin(int(output))[2:]:0>{self.qubits}}"
        return output

    def measure_no_reset(self) -> str:
        probabilities = np.abs(self.__state) ** 2
        probabilities = probabilities.flatten()
        output_states = [str(i) for i in range(len(probabilities))]
        output = np.random.choice(output_states, p=probabilities)
        output = f"{bin(int(output))[2:]:0>{self.qubits}}"
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
        
    def __check_qubit(self, qubit: int) -> None:
        if qubit < 0 or qubit >= self.qubits:
            raise StateError(str(qubit))
    
    def get_state(self) -> np.ndarray:
        return self.__state

    def gate(self, gate: np.ndarray, qubit: int) -> None:
        self.__check_qubit(qubit)
        
        self.__state = self.__single_qubit_gate(gate, qubit) @ self.__state
    
    def hadamard(self, qubit: int) -> None:
        self.__check_qubit(qubit)
        
        hadamard = gates.GATES[self.device]["HADAMARD"]
        self.__state = self.__single_qubit_gate(hadamard, qubit) @ self.__state
    
    def pauli_x(self, qubit: int) -> None:
        self.__check_qubit(qubit)
        
        pauli_x = gates.GATES[self.device]["PAULI_X"]
        self.__state = self.__single_qubit_gate(pauli_x, qubit) @ self.__state
    
    def pauli_y(self, qubit: int) -> None:
        self.__check_qubit(qubit)
        
        pauli_y = gates.GATES[self.device]["PAULI_Y"]
        self.__state = self.__single_qubit_gate(pauli_y, qubit) @ self.__state
    
    def pauli_z(self, qubit: int) -> None:
        self.__check_qubit(qubit)
        
        pauli_z = gates.GATES[self.device]["PAULI_Z"]
        self.__state = self.__single_qubit_gate(pauli_z, qubit) @ self.__state
    
    def cnot(self, control: int, target: int) -> None:
        if control < 0 or control >= self.qubits:
            raise StateError(str(control))
        if target < 0 or target >= self.qubits:
            raise StateError(str(target))
        
        cnot = gates.GATES[self.device]["CNOT"]
        identity = self.module.identity(2, dtype=complex)

        if control == 0:
            applied_gate = cnot
        else:
            applied_gate = identity
        for i in range(1, self.qubits-1):
            if i == control:
                applied_gate = self.module.kron(applied_gate, cnot)
            else:
                applied_gate = self.module.kron(applied_gate, identity)
        self.__state = applied_gate @ self.__state
    
    def swap(self, qubit1: int, qubit2: int) -> None:
        if qubit1 < 0 or qubit1 >= self.qubits:
            raise StateError(str(qubit1))
        if qubit2 < 0 or qubit2 >= self.qubits:
            raise StateError(str(qubit2))
        
        swap = gates.GATES[self.device]["SWAP"]
        identity = self.module.identity(2, dtype=complex)

        if qubit1 == 0:
            applied_gate = swap
        else:
            applied_gate = identity
        for i in range(1, self.qubits-1):
            if i == qubit1:
                applied_gate = self.module.kron(applied_gate, swap)
            else:
                applied_gate = self.module.kron(applied_gate, identity)
        self.__state = applied_gate @ self.__state
    
    def toffoli(self, control1: int, control2: int, target: int) -> None:
        if control1 < 0 or control1 >= self.qubits:
            raise StateError(str(control1))
        if control2 < 0 or control2 >= self.qubits:
            raise StateError(str(control2))
        if target < 0 or target >= self.qubits:
            raise StateError(str(target))
        
        toffoli = gates.GATES[self.device]["TOFFOLI"]
        identity = self.module.identity(2, dtype=complex)

        if control1 == 0:
            applied_gate = toffoli
        else:
            applied_gate = identity
        for i in range(1, self.qubits-2):
            if i == control1:
                applied_gate = self.module.kron(applied_gate, toffoli)
            else:
                applied_gate = self.module.kron(applied_gate, identity)
        self.__state = applied_gate @ self.__state
    
    def cswap(self, control: int, target1: int, target2: int) -> None:
        if control < 0 or control >= self.qubits:
            raise StateError(str(control))
        if target1 < 0 or target1 >= self.qubits:
            raise StateError(str(target1))
        if target2 < 0 or target2 >= self.qubits:
            raise StateError(str(target2))
        
        cswap = gates.GATES[self.device]["CSWAP"]
        identity = self.module.identity(2, dtype=complex)

        if control == 0:
            applied_gate = cswap
        else:
            applied_gate = identity
        for i in range(1, self.qubits-2):
            if i == control:
                applied_gate = self.module.kron(applied_gate, cswap)
            else:
                applied_gate = self.module.kron(applied_gate, identity)
        self.__state = applied_gate @ self.__state