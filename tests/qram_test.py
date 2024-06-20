from src.qcircpy import quantum
import unittest

class QRAMTest(unittest.TestCase):    
    def test_store_cpu(self):
        self.qc = quantum.QRAM("cpu")
        q1 = quantum.Qubit("0", "cpu")
        q2 = quantum.Qubit("00", "cpu")
        q3 = quantum.Qubit("000", "cpu")
        self.qc.store(q1, q2, q3)

    def test_store_gpu(self):
        self.qg = quantum.QRAM("gpu")
        q1 = quantum.Qubit("0", "gpu")
        q2 = quantum.Qubit("00", "gpu")
        q3 = quantum.Qubit("000", "gpu")
        self.qg.store(q1, q2, q3)

    def test_fetch_cpu(self):
        self.qc = quantum.QRAM("cpu")
        q1 = quantum.Qubit("0", "cpu")
        q2 = quantum.Qubit("00", "cpu")
        q3 = quantum.Qubit("000", "cpu")
        self.qc.store(q1, q2, q3)
        self.assertEqual(self.qc.fetch(0), q1)
        self.assertEqual(self.qc.fetch(1), q2)
        self.assertEqual(self.qc.fetch(2), q3)
    
    def test_fetch_gpu(self):
        self.qg = quantum.QRAM("gpu")
        q1 = quantum.Qubit("0", "gpu")
        q2 = quantum.Qubit("00", "gpu")
        q3 = quantum.Qubit("000", "gpu")
        self.qg.store(q1, q2, q3)
        self.assertEqual(self.qg.fetch(0), q1)
        self.assertEqual(self.qg.fetch(1), q2)
        self.assertEqual(self.qg.fetch(2), q3)
    
    def test_del_cpu(self):
        self.qc = quantum.QRAM("cpu")
        q1 = quantum.Qubit("0", "cpu")
        q2 = quantum.Qubit("00", "cpu")
        q3 = quantum.Qubit("000", "cpu")
        self.qc.store(q1, q2, q3)
        self.qc.delete(0)
        self.assertEqual(self.qc.fetch(0), q2)
        self.qc.delete(0)
        self.assertEqual(self.qc.fetch(0), q3)
    
    def test_del_gpu(self):
        self.qg = quantum.QRAM("gpu")
        q1 = quantum.Qubit("0", "gpu")
        q2 = quantum.Qubit("00", "gpu")
        q3 = quantum.Qubit("000", "gpu")
        self.qg.store(q1, q2, q3)
        self.qg.delete(0)
        self.assertEqual(self.qg.fetch(0), q2)
        self.qg.delete(0)
        self.assertEqual(self.qg.fetch(0), q3)

def suite():
    suite = unittest.TestSuite()
    suite.addTest(QRAMTest('test_store_cpu'))
    suite.addTest(QRAMTest('test_store_gpu'))
    suite.addTest(QRAMTest('test_fetch_cpu'))
    suite.addTest(QRAMTest('test_fetch_gpu'))
    suite.addTest(QRAMTest('test_del_cpu'))
    suite.addTest(QRAMTest('test_del_gpu'))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())