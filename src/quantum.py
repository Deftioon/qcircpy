import math
import typing
import numpy as np
import cupy as cp

if __name__ == "__main__":
    from exceptions import *
    from debugging import *

else:
    from .exceptions import *
    from .debugging import *

class Qubit:
    def __init__(self, base_state: str, device: str):
        self.device = device

        if device == "cpu":
            self.module = np
        elif device == "gpu":
            self.module = cp
        else:
            raise DeviceError(device)

        self.space = len(base_state)
        self.width = 2 ** self.space

        self.data = self.module.zeros((self.width, 1), dtype=np.cfloat)
        self.data[int(base_state, 2)] = 1
    
    
    def __str__(self):
        return f"{self.data}"
    
    def to_device(self, device):
        if device == "cpu" and self.module == cp:
            self.data = cp.asnumpy(self.data)
            self.module = np
        elif device == "gpu" and self.module == np:
            self.data = cp.asarray(self.data)
            self.module = cp
        elif device != "cpu" and device != "gpu":
            raise DeviceError(device)
        
        return self
    
    
    def copy(self):
        output = Qubit("0", self.device)
        output.module = self.module
        output.data = self.data
        output.space = self.space
        output.width = self.width
        return output
    
    def measure(self):
        probs = [float(abs(self.data[i])) ** 2 for i in range(len(self.data))]
        choice = self.module.random.choice([i for i in range(len(self.data))], p=probs)

        self.data = self.module.zeros((self.width, 1), dtype=np.cfloat)
        self.data[choice] = 1

        return bin(choice)[2:].ljust(int(math.log2(self.width)), "0")
    
class QRAM:
    def __init__(self, device: str, *args):
        self.device = device
        self.qubits = list(args)
        self.address_lengths = []
        
        if device == "cpu":
            self.module = np
        elif device == "gpu":
            self.module = cp
        else:
            raise DeviceError(device)
        
        self.data = self.module.zeros((2, 1), dtype=np.cfloat)
    
    def __str__(self):
        return f"Quantum Memory Unit with {len(self.qubits)} qubits and data of spaces {self.address_lengths}, occupying {[qubit.width for qubit in self.qubits]} vector spaces."
    
    def init(self):
        self.address_lengths = [self.qubits[0].space]
        self.data = self.qubits[0].data
        for qubit in self.qubits[1:]:
            self.address_lengths.append(qubit.space)
            self.data = np.vstack((self.data, qubit.data))



    
    def store(self, *args):
        for qubit in args:
            self.qubits.append(qubit)
        
        self.init()

    def fetch(self, address):
        return self.qubits[address]

    def delete(self, address):
        self.qubits.pop(address)
        self.init()
    
    def read_contents(self):
        return self.data