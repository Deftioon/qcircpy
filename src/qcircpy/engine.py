if __name__ == "__main__":
    import quantum, gates, circuits
    from exceptions import *
else:
    from . import quantum, gates, circuits
    from .exceptions import *
    
import time
import typing

class Engine:
    """
    The Engine class represents a quantum computing engine.

    Args:
        device (str): The device to be used for computation. Must be either "cpu" or "gpu".
        clock (float): The clock speed of the device in Hz.

    Raises:
        DeviceError: If an invalid device is provided.

    Attributes:
        device (str): The device used for computation.
        clock (float): The clock speed of the device.
        hadamard: The Hadamard gate.
        x: The Pauli-X gate.
        y: The Pauli-Y gate.
        z: The Pauli-Z gate.
        t: The T gate.
        cnot: The controlled-NOT gate.
        cz: The controlled-Z gate.
        swap: The SWAP gate.
        ccnot: The doubly-controlled-NOT gate.
    """

    def __init__(self, device: str, clock: int):
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

    def Qubit(self, base_state: str) -> quantum.Qubit:
        """
        Create a qubit with the given base state.

        Args:
            base_state: The base state of the qubit.

        Returns:
            quantum.Qubit: The created qubit.
        """
        return quantum.Qubit(base_state, self.device)
    
    def Wire(self, *args: gates.Gate) -> circuits.Wire:
        """
        Create a wire with the given gates.

        Args:
            *args: The gates to be applied to the wire.

        Returns:
            circuits.Wire: The created wire.
        """
        return circuits.Wire(*args)
    
    def Connection(self, gate: gates.Gate, *args: circuits.Wire):
        """
        Create a connection with the given gate and wires.

        Args:
            gate: The gate to be applied in the connection.
            *args: The wires to be connected.

        Returns:
            circuits.Connection: The created connection.
        """
        return circuits.Connection(self.device, gate, *args)
    
    def hadamard(self) -> gates.Gate:
        """
        Get the Hadamard gate.

        Returns:
            The Hadamard gate.
        """
        return gates.hadamard
    
    def x(self) -> gates.Gate:
        """
        Get the Pauli-X gate.

        Returns:
            The Pauli-X gate.
        """
        return gates.x
    
    def y(self) -> gates.Gate:
        """
        Get the Pauli-Y gate.

        Returns:
            The Pauli-Y gate.
        """
        return gates.y
    
    def z(self) -> gates.Gate:
        """
        Get the Pauli-Z gate.

        Returns:
            The Pauli-Z gate.
        """
        return gates.z
    
    def t(self) -> gates.Gate:
        """
        Get the T gate.

        Returns:
            The T gate.
        """
        return gates.t
    
    def cnot(self) -> gates.Gate:
        """
        Get the controlled-NOT gate.

        Returns:
            The controlled-NOT gate.
        """
        return gates.cnot
    
    def cz(self) -> gates.Gate:
        """
        Get the controlled-Z gate.

        Returns:
            The controlled-Z gate.
        """
        return gates.cz
    
    def swap(self) -> gates.Gate:
        """
        Get the SWAP gate.

        Returns:
            The SWAP gate.
        """
        return gates.swap
    
    def ccnot(self) -> gates.Gate:
        """
        Get the doubly-controlled-NOT gate.

        Returns:
            The doubly-controlled-NOT gate.
        """
        return gates.ccnot

    def run(self, circuit: circuits.Wire | circuits.Connection) -> function:
        """
        Execute the given circuit.

        Args:
            circuit: The circuit to be executed.

        Returns:
            function: The execute function.
        """
        def execute(data: quantum.Qubit) -> quantum.Qubit:
            if isinstance(circuit, circuits.Wire):
                qubit = data.to_device(self.device)
                gates = len(circuit.gates)
                return circuit.parse(qubit)
        return execute

    def benchmark(self, circuit: circuits.Wire | circuits.Connection) -> function:
        """
        Execute the given circuit and print benchmark information.

        Args:
            circuit: The circuit to be executed.

        Returns:
            function: The execute function.
        """
        @timer
        def execute(data: quantum.Qubit) -> quantum.Qubit:
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