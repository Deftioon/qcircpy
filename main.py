import src.qcircpy as qcircpy
import matplotlib.pyplot as plt

x = qcircpy.gates.GATES["cpu"]["PAULI_X"]

gate = qcircpy.gates.controlled_gate(3, x, 2, 0, 1)
print(gate.shape)
plt.imshow(gate.real, cmap='viridis', interpolation='none')
plt.colorbar()
plt.title("Controlled Gate Visualization")
plt.savefig("controlled_gate.png")