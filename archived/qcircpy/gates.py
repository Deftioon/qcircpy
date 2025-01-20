import numpy as np
import cupy as cp

if __name__ == "__main__":
    import quantum as q
    from exceptions import *
else:
    from . import quantum as q
    from .exceptions import *


class GateMatrices:
    """Class containing predefined gate matrices."""
    identity = np.array([[1, 0], [0, 1]])

    hadamard = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])

    x = np.array([[0, 1], [1, 0]])
    y = np.array([[0, -1j], [1j, 0]])
    z = np.array([[1, 0], [0, -1]])

    t = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])

    cnot = np.array([[1, 0, 0, 0], 
                     [0, 1, 0, 0], 
                     [0, 0, 0, 1], 
                     [0, 0, 1, 0]])

    cz = np.array([[1, 0, 0, 0], 
                   [0, 1, 0, 0], 
                   [0, 0, 1, 0], 
                   [0, 0, 0, -1]])

    swap = np.array([[1, 0, 0, 0], 
                     [0, 0, 1, 0], 
                     [0, 1, 0, 0], 
                     [0, 0, 0, 1]])

    ccnot = np.array([[1, 0, 0, 0, 0, 0, 0, 0], 
                      [0, 1, 0, 0, 0, 0, 0, 0], 
                      [0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0], 
                      [0, 0, 0, 0, 1, 0, 0, 0], 
                      [0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1], 
                      [0, 0, 0, 0, 0, 0, 1, 0]])


class Gate:
    """Class representing a gate."""
    def __init__(self, matrix: np.ndarray | cp.ndarray, partition: int, device: str):
        """
        Initialize a Gate object.

        Args:
            matrix (np.ndarray | cp.ndarray): The matrix representing the gate.
            partition (int): The partition size of the gate.
            device (str): The device to use for computation ("cpu" or "gpu").

        Raises:
            DeviceError: If an invalid device is specified.
        """
        self.matrix = matrix
        self.partition = partition
        self.device = device

        if device == "cpu":
            self.module = np
        elif device == "gpu":
            self.module = cp
        else:
            raise DeviceError(device)

        self.matrix = self.module.asarray(self.matrix)

    def __call__(self, data):
        """
        Apply the gate to a qubit.

        Args:
            data (q.Qubit): The qubit to apply the gate to.

        Returns:
            q.Qubit: The resulting qubit after applying the gate.

        Raises:
            InvalidType: If the input data is not a qubit.
        """
        if not isinstance(data, q.Qubit):
            raise InvalidType("Gate is only defined for Qubit objects.")

        matrix = data.module.asarray(self.matrix)
        gate_matrix = data.module.asarray(self.matrix)

        for i in range(int(data.space/self.partition) - 1):
            gate_matrix = data.module.kron(gate_matrix, matrix)

        output = data.copy()
        data = gate_matrix @ data.data
        output.data = data

        return output

    def extend_matrix_space(self, space: int):
        """
        Extend the matrix space of the gate.

        Args:
            space (int): The desired matrix space size.

        Returns:
            np.ndarray | cp.ndarray: The extended gate matrix.

        """
        output = self.matrix.copy()
        matrix = self.matrix.copy()
        for i in range(int(space/self.partition) - 1):
            output = np.kron(self.matrix, matrix)

        return Gate(output, self.partition, self.device)

    def to_device(self, device: str):
        """
        Move the gate to a different device.

        Args:
            device (str): The device to move the gate to ("cpu" or "gpu").

        Returns:
            Gate: The gate object after moving to the specified device.

        Raises:
            DeviceError: If an invalid device is specified.
        """
        if device == "cpu" and self.module == cp:
            self.matrix = cp.asnumpy(self.matrix)
            self.module = np
        elif device == "gpu" and self.module == np:
            self.matrix = cp.asarray(self.matrix)
            self.module = cp
        elif device != "cpu" and device != "gpu":
            raise DeviceError(device)

        return self


CPU_IDENTITY = Gate(GateMatrices.identity, 1, device="cpu")
CPU_HADAMARD = Gate(GateMatrices.hadamard, 1, device="cpu")
CPU_X = Gate(GateMatrices.x, 1, device="cpu")
CPU_Y = Gate(GateMatrices.y, 1, device="cpu")
CPU_Z = Gate(GateMatrices.z, 1, device="cpu")
CPU_T = Gate(GateMatrices.t, 1, device="cpu")
CPU_CNOT = Gate(GateMatrices.cnot, 2, device="cpu")
CPU_CZ = Gate(GateMatrices.cz, 2, device="cpu")
CPU_SWAP = Gate(GateMatrices.swap, 2, device="cpu")
CPU_CCNOT = Gate(GateMatrices.ccnot, 3, device="cpu")

GPU_IDENTITY = Gate(GateMatrices.identity, 1, device="gpu")
GPU_HADAMARD = Gate(GateMatrices.hadamard, 1, device="gpu")
GPU_X = Gate(GateMatrices.x, 1, device="gpu")
GPU_Y = Gate(GateMatrices.y, 1, device="gpu")
GPU_Z = Gate(GateMatrices.z, 1, device="gpu")
GPU_T = Gate(GateMatrices.t, 1, device="gpu")
GPU_CNOT = Gate(GateMatrices.cnot, 2, device="gpu")
GPU_CZ = Gate(GateMatrices.cz, 2, device="gpu")
GPU_SWAP = Gate(GateMatrices.swap, 2, device="gpu")
GPU_CCNOT = Gate(GateMatrices.ccnot, 3, device="gpu")

GATE_DICT = {
    "cpu": {
        "identity": CPU_IDENTITY,
        "hadamard": CPU_HADAMARD,
        "x": CPU_X,
        "y": CPU_Y,
        "z": CPU_Z,
        "t": CPU_T,
        "cnot": CPU_CNOT,
        "cz": CPU_CZ,
        "swap": CPU_SWAP,
        "ccnot": CPU_CCNOT
    },
    "gpu": {
        "identity": GPU_IDENTITY,
        "hadamard": GPU_HADAMARD,
        "x": GPU_X,
        "y": GPU_Y,
        "z": GPU_Z,
        "t": GPU_T,
        "cnot": GPU_CNOT,
        "cz": GPU_CZ,
        "swap": GPU_SWAP,
        "ccnot": GPU_CCNOT
    }
}