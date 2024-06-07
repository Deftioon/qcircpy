from src import quantum, gates, circuits
import time

def run(qubit, device):
    def execute(wire):
        if isinstance(wire, circuits.Wire):
            return wire.parse(qubit)
    return execute

def benchmark(data, device):
    @timer
    def execute(wire):
        if isinstance(wire, circuits.Wire):
            qubit = data.to_device(device)
            print(f"Wire took {len(wire.gates)} gates.", end = " ")
            return wire.parse(qubit)
    return execute

def timer(func):
    def new_func(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Execution time: {end - start}")
        return result
    return new_func 