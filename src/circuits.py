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
    def __init__(self, *args):
        self.gates = list(args)
    
    def parse(self, qubit):
        output = qubit.copy()
        for gate in self.gates:
            output = gate(output)
        return output

class Connection:
    def __init__(self, device, gate, *args):
        self.device = device
        self.gate = gate
        self.wires = list(args)

        if device == "cpu":
            self.module = np
        elif device == "gpu":
            self.module = cp
        else:
            raise DeviceError(device)