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

import numpy as np
import cupy as cp
import math

class Qubit:
    """
    Represents a quantum bit (qubit) in a quantum computer.
    
    Attributes:
        device (str): The device on which the qubit is stored ("cpu" or "gpu").
        module (module): The module used for computations (numpy or cupy).
        space (int): The number of qubits in the base state.
        width (int): The total number of possible states.
        data (ndarray): The state vector of the qubit.
    """
    
    def __init__(self, base_state: str, device: str):
        """
        Initializes a qubit with the given base state and device.
        
        Args:
            base_state (str): The base state of the qubit in binary representation.
            device (str): The device on which the qubit will be stored ("cpu" or "gpu").
        
        Raises:
            DeviceError: If an invalid device is provided.
        """
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
    
    def to_device(self, device):
        """
        Transfers the qubit to the specified device.
        
        Args:
            device (str): The device on which the qubit will be transferred ("cpu" or "gpu").
        
        Returns:
            Qubit: The qubit object after the transfer.
        
        Raises:
            DeviceError: If an invalid device is provided.
        """
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
        """
        Creates a copy of the qubit.
        
        Returns:
            Qubit: A new qubit object with the same state and properties.
        """
        output = Qubit("0", self.device)
        output.module = self.module
        output.data = self.data
        output.space = self.space
        output.width = self.width
        return output
    
    def measure(self):
        """
        Measures the qubit and returns the observed state. Resets the qubit to the observed state.
        
        Returns:
            str: The observed state of the qubit in binary representation.
        """
        probs = [float(abs(self.data[i])) ** 2 for i in range(len(self.data))]
        choice = self.module.random.choice([i for i in range(len(self.data))], p=probs)

        self.data = self.module.zeros((self.width, 1), dtype=np.cfloat)
        self.data[choice] = 1

        return bin(choice)[2:].ljust(int(math.log2(self.width)), "0")
    
class QRAM:
    """
    Quantum Random Access Memory (QRAM) class.

    Args:
        device (str): The device to be used for computation. Can be "cpu" or "gpu".
        *args: Variable length argument list of qubits.

    Attributes:
        device (str): The device used for computation.
        qubits (list): List of qubits.
        address_lengths (list): List of address lengths.
        module (module): The module used for computation based on the device.
        data (ndarray): The data stored in the QRAM.

    Methods:
        __str__(): Returns a string representation of the QRAM.
        init(): Initializes the QRAM with the first qubit.
        store(*args): Stores additional qubits in the QRAM.
        fetch(address): Fetches a qubit from the QRAM based on the given address.
        delete(address): Deletes a qubit from the QRAM based on the given address.
        read_contents(): Returns the data stored in the QRAM.

    Raises:
        DeviceError: If an invalid device is provided.

    """

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
        """
        Stores additional qubits in the QRAM.

        Args:
            *args: Variable length argument list of qubits.

        """
        for qubit in args:
            self.qubits.append(qubit)
        
        self.init()

    def fetch(self, address):
        """
        Fetches a qubit from the QRAM based on the given address.

        Args:
            address: The address of the qubit to fetch.

        Returns:
            The qubit at the given address.

        """
        return self.qubits[address]

    def delete(self, address):
        """
        Deletes a qubit from the QRAM based on the given address.

        Args:
            address: The address of the qubit to delete.

        """
        self.qubits.pop(address)
        self.init()
    
    def read_contents(self):
        """
        Returns the data stored in the QRAM.

        Returns:
            The data stored in the QRAM.

        """
        return self.data