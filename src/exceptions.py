class DeviceError(Exception):
    def __init__(self, device):
        self.device = device

    def __str__(self):
        return f"DeviceError: '{self.device}' is not a valid device. Please use 'cpu' or 'gpu'."
    
class InvalidQubit(Exception):
    def __init__(self, qubit):
        self.qubit = qubit

    def __str__(self):
        return f"InvalidQubit: Qubit is invalid, sums to: {self.qubit.module.sum([float(abs(self.qubit.data[i])) ** 2 for i in range(len(self.qubit.data))])}"