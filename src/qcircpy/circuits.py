import numpy as np
import cupy as cp
import typing

if __name__ == "__main__":
    import quantum as q
    from exceptions import *
    import gates as g

else:
    from . import quantum as q
    from .exceptions import *
    from . import gates as g

class Circuit:
    def __init__(self):
        self.device = None
        self.matrix = None
    
    def __call__(self, qubit: q.Qubit) -> q.Qubit:
        return self.parse(qubit)
    
    def to_device(self, device: str):
        if device == "cpu" and self.device == "gpu":
            self.matrix = cp.asnumpy(self.matrix)
            self.device = "cpu"
        elif device == "gpu" and self.device == "cpu":
            self.matrix = cp.asarray(self.matrix)
            self.device = "gpu"
        elif device != "cpu" and device != "gpu":
            raise DeviceError(device)
        
        return self
    
    def parse(self, qubit: q.Qubit) -> q.Qubit:
        if not isinstance(qubit, q.Qubit):
            raise InvalidType("Circuit is only defined for Qubit objects.")
        
        if qubit.space != self.in_channels:
            raise ConnectionError(f"Given input shape {qubit.space} does not match Circuit input shape {self.in_channels}.")
        
        output = qubit.copy()
        output.data = self.matrix @ output.data
        return output

class Wire(Circuit):
    """
    Represents a wire in a quantum circuit.

    Attributes:
        gates (tuple): A tuple of quantum gates applied to the wire.
        matrix (ndarray): The matrix representation of the gates compounded, applied to the wire.

    Methods:
        parse(qubit): Applies a series of quantum gates to a qubit.

    """
    def __init__(self, device: str, *args: g.Gate) -> None:
        self.device = device
        self.gates = list(args)
        self.matrix = None

        if device == "cpu":
            self.module = np
        elif device == "gpu":
            self.module = cp
        else:
            raise DeviceError(device)
        
        for i, gate in enumerate(self.gates):
            if not isinstance(gate, g.Gate):
                raise InvalidType("Wire only accepts Gate objects.")
            
            if gate.device != self.device:
                self.gates[i] = gate.to_device(self.device)
            
        self.matrix = self.module.asarray(self.gates[-1].matrix)
        for gate in self.gates[:-1]:
            self.matrix = self.module.asarray(gate.matrix) @ self.matrix

        self.in_channels: int = self.module.round(self.module.log2(self.matrix.shape[0]))
        self.out_channels: int = self.module.round(self.module.log2(self.matrix.shape[1]))
    
    def init_matrix(self, space: int, device: str) -> None:
        if device == "cpu":
            module = np
        elif device == "gpu":
            module = cp
        
        self.matrix = module.asarray(self.gates[-1].extend_matrix_space(space).matrix)
        for gate in self.gates[:-1]:
            self.matrix = module.asarray(gate.extend_matrix_space(space).matrix) @ self.matrix

class Connection(Circuit):
    def __init__(self, device: str, gate: g.Gate, *args: Wire) -> None:
        self.device = device
        self.gate = gate
        self.wires = list(args)
        self.matrix = None

        if device == "cpu":
            self.module = np
        elif device == "gpu":
            self.module = cp
        else:
            raise DeviceError(device)
        
        self.in_channels: int = self.module.round(self.module.log2(gate.matrix.shape[0]))
        self.out_channels: int = self.module.round(self.module.log2(gate.matrix.shape[1]))
        
        # if self.module.log2(gate.matrix.shape[0]) != len(self.wires):
        #     raise IncompatibleShapes(len(self.wires), self.module.log2(gate.matrix.shape[0]))
        
        for wire in self.wires:
            wire.init_matrix(self.module.log2(gate.matrix.shape[0])/self.gate.partition, device)
        
        self.matrix = self.module.asarray(self.wires[0].matrix)
        for wire in self.wires[1:]:
            if not isinstance(wire, Wire):
                raise InvalidType("Connection only accepts Wire objects.")
            
            self.matrix = self.module.kron(self.matrix, self.module.asarray(wire.matrix))
        
        self.matrix = self.matrix @ self.module.asarray(gate.matrix)