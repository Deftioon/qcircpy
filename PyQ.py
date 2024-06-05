from src import quantum, gates, circuits
from src.debugging import verify


device = "cpu"


q1 = quantum.Qubit("1", device)
print(q1)
gates = gates.Gates

circuit = circuits.Wire(device, gates.y())
q3 = circuit.parse(q1)
print(q3)