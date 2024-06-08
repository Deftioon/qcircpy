import math

def verify(qubit):
    probability_sum = qubit.module.sum(qubit.module.asarray([float(abs(qubit.data[i])) ** 2 for i in range(len(qubit.data))]))
    return math.isclose(probability_sum, 1)