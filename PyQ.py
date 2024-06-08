from src import quantum, gates, circuits
from src.debugging import *
import numpy as np
import engine

wire = circuits.Wire(gates.hadamard, gates.x, gates.y, gates.z, gates.t)
q1 = quantum.Qubit("01", "cpu")
q2 = wire.parse(q1)

benchmarker = engine.benchmark(q1, "cpu", 10)
q3 = benchmarker(wire)
print(verify(q3))