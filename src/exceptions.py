class DeviceError(Exception):
    def __init__(self, device):
        self.device = device

    def __str__(self):
        return f"DeviceError: '{self.device}' is not a valid device. Please use 'cpu' or 'gpu'."