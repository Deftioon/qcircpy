from src import quantum, gates, circuits
import numpy as np
import engine

wire = circuits.Wire(gates.Gates.hadamard(), gates.Gates.y(), gates.Gates.p(1j * np.pi/2))
q1 = quantum.Qubit("0101010110", "gpu")
q2 = wire.parse(q1)

engine.benchmark(q1, "gpu")(wire)