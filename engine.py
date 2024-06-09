from src import quantum, gates, circuits
from src.exceptions import *
import time

class Engine:
    def __init__(self, device, clock):
        self.device = device
        self.clock = clock

        if device != "cpu" and device != "gpu":
            raise DeviceError(device)
        
        self.hadamard = gates.hadamard
        self.x = gates.x
        self.y = gates.y
        self.z = gates.z
        self.t = gates.t
        self.cnot = gates.cnot
        self.cz = gates.cz
        self.swap = gates.swap
        self.ccnot = gates.ccnot

    def Qubit(self, base_state):
        return quantum.Qubit(base_state, self.device)
    
    def Wire(self, *args):
        return circuits.Wire(*args)
    
    def Connection(self, gate, *args):
        return circuits.Connection(self.device, gate, *args)
    
    def hadamard(self):
        return gates.hadamard
    
    def x(self):
        return gates.x
    
    def y(self):
        return gates.y
    
    def z(self):
        return gates.z
    
    def t(self):
        return gates.t
    
    def cnot(self):
        return gates.cnot
    
    def cz(self):
        return gates.cz
    
    def swap(self):
        return gates.swap
    
    def ccnot(self):
        return gates.ccnot

    def run(self, circuit):
        def execute(data):
            if isinstance(circuit, circuits.Wire):
                qubit = data.to_device(self.device)
                gates = len(circuit.gates)
                return circuit.parse(qubit)
        return execute

    def benchmark(self, circuit):
        @timer
        def execute(data):
            if isinstance(circuit, circuits.Wire):
                qubit = data.to_device(self.device)
                gates = len(circuit.gates)
                print(f"Wire took {gates} gates. At {self.clock}Hz, it will take {gates/self.clock: .2f}s.", end = " ")
                return circuit.parse(qubit)
            
            if isinstance(circuit, circuits.Connection):
                qcirc = circuit.to_device(self.device)
                qubit = data.to_device(self.device)
                gates = len(qcirc.wires)
                print(f"Connection took {gates} wires, each of {[len(i.gates) for i in qcirc.wires]}. At {self.clock}Hz, it will take {gates/self.clock: .2f}s.", end = " ")
                return qcirc.parse(qubit)
        return execute
    
hadamard = gates.hadamard
x = gates.x
y = gates.y
z = gates.z
t = gates.t
cnot = gates.cnot
cz = gates.cz
swap = gates.swap
ccnot = gates.ccnot

def timer(func):
    def new_func(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Execution time: {end - start: .10f}")
        return result
    return new_func 