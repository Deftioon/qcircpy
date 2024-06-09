import math

import math

def verify(qubit):
    """
    Verify if the given qubit is valid.

    Parameters:
    qubit (Qubit): The qubit to be verified.

    Returns:
    bool: True if the qubit is valid, False otherwise.
    """
    probability_sum = qubit.module.sum(qubit.module.asarray([float(abs(qubit.data[i])) ** 2 for i in range(len(qubit.data))]))
    return math.isclose(probability_sum, 1)