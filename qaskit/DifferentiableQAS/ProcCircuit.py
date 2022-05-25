from Basic import Basic
from Method import Method
import copy


class ProcCircuit(Method):
    def __init__(self, basic: Basic):
        super().__init__(basic)

    def __call__(self, circuit_with_layer):
        proc_circuit = []
        for layer in circuit_with_layer:
            for i in range(self.n_qubits)

