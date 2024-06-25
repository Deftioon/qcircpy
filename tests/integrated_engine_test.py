from src.qcircpy import engine as qp
import unittest
import numpy as np
import cupy as cp

class IntegratedEngineTest(unittest.TestCase):
    def create_engine(self):
        self.engine = qp.Engine("cpu", 1)
    
    def test_create_wire(self):
        self.create_engine()
        wire = self.engine.Wire(self.engine.hadamard(), self.engine.x())
    
    def test_apply_wire(self):
        self.create_engine()
        wire = self.engine.Wire(self.engine.hadamard(), self.engine.x())
        q = self.engine.Qubit("0")
        out = wire(q)
    
    def test_create_connection(self):
        self.create_engine()
        wire1 = self.engine.Wire(self.engine.hadamard(), self.engine.x())
        wire2 = self.engine.Wire(self.engine.hadamard(), self.engine.x())
        connection = self.engine.Connection(self.engine.cnot(), wire1, wire2)
    
    def test_apply_connection(self):
        self.create_engine()
        wire1 = self.engine.Wire(self.engine.hadamard(), self.engine.x())
        wire2 = self.engine.Wire(self.engine.hadamard(), self.engine.x())
        connection = self.engine.Connection(self.engine.cnot(), wire1, wire2)
        q = self.engine.Qubit("00")
        out = connection(q)
    

def suite():
    suite = unittest.TestSuite()
    suite.addTest(IntegratedEngineTest('test_create_wire'))
    suite.addTest(IntegratedEngineTest('test_apply_wire'))
    suite.addTest(IntegratedEngineTest('test_create_connection'))
    suite.addTest(IntegratedEngineTest('test_apply_connection'))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
    
