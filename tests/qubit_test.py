from src.qcircpy import quantum
import unittest

class QubitTests(unittest.TestCase):
    def test_create_qubit_cpu(self):
        q = quantum.Qubit("0", "cpu")
        q = quantum.Qubit("00", "cpu")
        q = quantum.Qubit("000", "cpu")

    def test_create_qubit_gpu(self):
        q = quantum.Qubit("0", "gpu")
        q = quantum.Qubit("00", "gpu")
        q = quantum.Qubit("000", "gpu")
    
    @unittest.expectedFailure
    def test_create_failed_qubit(self):
        q = quantum.Qubit("0", "tpu")
        q = quantum.Qubit("00", "tpu")
        q = quantum.Qubit("000", "tpu")
        q = quantum.Qubit(1209471, "cpu")
    
    def test_transfer_qubit(self):
        q = quantum.Qubit("0", "cpu")
        q.to_device("gpu")
        q.to_device("cpu")
        q.to_device("gpu")
    
    def test_copy_qubit(self):
        q = quantum.Qubit("0", "cpu")
        q2 = q.copy()
    
    def test_measure_qubit(self):
        q = quantum.Qubit("0", "cpu")
        q.measure()
        q = quantum.Qubit("00", "cpu")
        q.measure()
        q = quantum.Qubit("000", "cpu")
        q.measure()

def suite():
    suite = unittest.TestSuite()
    suite.addTest(QubitTests('test_create_qubit_cpu'))
    suite.addTest(QubitTests('test_create_qubit_gpu'))
    suite.addTest(QubitTests('test_create_failed_qubit'))
    suite.addTest(QubitTests('test_transfer_qubit'))
    suite.addTest(QubitTests('test_copy_qubit'))
    suite.addTest(QubitTests('test_measure_qubit'))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())