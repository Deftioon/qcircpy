if __name__ == "__main__":
    from debugging import *
    from quantum import *

else:
    from .debugging import *
    from .quantum import *


class DeviceError(Exception):
    def __init__(self, device: str) -> None:
        self.device = device

    def __str__(self) -> str:
        return f"DeviceError: '{self.device}' is not a valid device. Please use 'cpu' or 'gpu'."

class IncompatibleShapes(Exception):
    def __init__(self, shape1: int, shape2: int) -> None:
        self.shape1 = shape1
        self.shape2 = shape2

    def __str__(self) -> str:
        return f"IncompatibleShapes: Qubit Spaces {self.shape1} and {self.shape2} are incompatible."
    
class ConnectionError(Exception):
    def __init__(self, string: str) -> None:
        self.string = string

    def __str__(self) -> str:
        return f"ConnectionError: {self.string}"

class InvalidType(Exception):
    def __init__(self, string: str) -> None:
        self.string = string

    def __str__(self) -> str:
        return f"InvalidType: {self.string}"
    
class InvalidGate(Exception):
    def __init__(self, string: str) -> None:
        self.string = string

    def __str__(self) -> str:
        return f"InvalidGate: {self.string}"
    
class InvalidQubit(Exception):
    def __init__(self, qubit) -> None:
        self.qubit = qubit

    def __str__(self):
        return f"InvalidQubit: Qubit is invalid, sums to: {self.qubit.module.sum([float(abs(self.qubit.data[i])) ** 2 for i in range(len(self.qubit.data))])}"