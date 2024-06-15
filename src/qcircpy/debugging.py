import math

if __name__ == "__main__":
    import quantum
else:
    from . import quantum

def verify(qubit: quantum.Qubit) -> bool:
    """
    Verify if the given qubit is valid.

    Parameters:
    qubit (Qubit): The qubit to be verified.

    Returns:
    bool: True if the qubit is valid, False otherwise.
    """
    probability_sum = qubit.module.sum(qubit.module.asarray([float(abs(qubit.data[i])) ** 2 for i in range(len(qubit.data))]))
    return math.isclose(probability_sum, 1)