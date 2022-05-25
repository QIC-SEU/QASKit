from Basic import Basic
from OperationPool import OperationPool
from ProcCircuit import ProcCircuit

class DifferentiableQAS:
    def __init__(self, n_qubits, coupling, n_layers, noise_model=None):
        self.basic = Basic(n_qubits, coupling, n_layers, noise_model=noise_model)
        self.OperationPool = OperationPool(self.basic)
        self.ProcCircuit = ProcCircuit(self.basic)


