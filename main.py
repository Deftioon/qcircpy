import src.qcircpy as qcircpy

runtime = qcircpy.Runtime("00", "cpu")
runtime.hadamard(0)
runtime.cnot(0, 1)
print(runtime.state)