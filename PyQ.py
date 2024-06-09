from engine import *

GPU_Engine = Engine("gpu", 100)
CPU_Engine = Engine("cpu", 100)

engine = CPU_Engine

q1 = engine.Qubit("01")
w1 = engine.Wire(hadamard, x, y, z, t)
w2 = engine.Wire(hadamard, y, z, x) 
c1 = engine.Connection(cnot, w1, w2)