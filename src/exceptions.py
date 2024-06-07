class DeviceError(Exception):
    def __init__(self, device):
        self.device = device

    def __str__(self):
        return f"DeviceError: '{self.device}' is not a valid device. Please use 'cpu' or 'gpu'."

class IncompatibleShapes(Exception):
    def __init__(self, shape1, shape2):
        self.shape1 = shape1
        self.shape2 = shape2

    def __str__(self):
        return f"IncompatibleShapes: Qubit Spaces {self.shape1.space} and {self.shape2.space} are incompatible."

class InvalidType(Exception):
    def __init__(self, string):
        self.string = string

    def __str__(self):
        return f"InvalidType: {self.string}"
    
class InvalidGate(Exception):
    def __init__(self, string):
        self.string = string

    def __str__(self):
        return f"InvalidGate: {self.string}"
    
class InvalidQubit(Exception):
    def __init__(self, qubit):
        self.qubit = qubit

    def __str__(self):
        return f"InvalidQubit: Qubit is invalid, sums to: {self.qubit.module.sum([float(abs(self.qubit.data[i])) ** 2 for i in range(len(self.qubit.data))])}"