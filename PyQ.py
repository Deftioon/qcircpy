from src import quantum

device = "cpu"


q1 = quantum.Qubit("1101", device)
print(q1.measure())