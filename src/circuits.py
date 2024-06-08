import numpy as np
import cupy as cp
import typing

if __name__ == "__main__":
    import quantum as q
    from exceptions import *

else:
    from . import quantum as q
    from .exceptions import *

class Wire:
    """
    Represents a wire in a quantum circuit.

    Attributes:
        gates (tuple): A tuple of quantum gates applied to the wire.
        matrix (ndarray): The matrix representation of the gates compounded, applied to the wire.

    Methods:
        parse(qubit): Applies a series of quantum gates to a qubit.

    """
    def __init__(self, *args):
        self.gates = args
        self.matrix = None
    
    def parse(self, qubit):
        '''
        Applies a series of quantum gates to a qubit.

        Args:
            qubit (Qubit): The input qubit to apply the gates to.

        Returns:
            Qubit: The output qubit after applying the gates.
        '''
        self.matrix = qubit.module.asarray(self.gates[-1].extend_matrix_space(qubit.space))
        for gate in self.gates[:-1]:
            self.matrix = qubit.module.asarray(gate.extend_matrix_space(qubit.space)) @ self.matrix
        
        output = qubit.copy()
        output.data = self.matrix @ output.data
        return output

class Connection:
    def __init__(self, device, gate, *args):
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
        
        if self.module.log2(gate.matrix.shape[0]) != len(self.wires):
            raise IncompatibleShapes(len(self.wires), self.module.log2(gate.matrix.shape[0]))
        
        self.matrix = self.module.asarray(self.wires[0].matrix)
        for wire in self.wires[1:]:
            if not isinstance(wire, Wire):
                raise InvalidType("Connection only accepts Wire objects.")
            
            self.matrix = self.module.kron(self.matrix, self.module.asarray(wire.matrix))
        
        self.matrix = self.matrix @ self.module.asarray(gate.matrix)
            
            