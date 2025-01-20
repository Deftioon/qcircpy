class DeviceError(Exception):
    def __init__(self, device: str) -> None:
        self.device = device
        self.message = f"Device {device} is not supported."
        super().__init__(self.message)

class StateError(Exception):
    def __init__(self, state: str) -> None:
        self.state = state
        self.message = f"State {state} is invalid."
        super().__init__(self.message)