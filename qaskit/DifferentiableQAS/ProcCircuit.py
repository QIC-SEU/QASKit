from Basic import Basic
from Method import Method
import copy


class ProcCircuit(Method):
    def __init__(self, basic: Basic):
        super().__init__(basic)

    def circuit_from_single_layer(self, layer):
        circuit = []
        for i in range(self.n_qubits):
            if layer[0][i] != 'I':
                circuit.append((layer[0][i], i))
        for cx in layer[1]:
            circuit.append(('CX', cx[0], cx[1]))
        return circuit

    def __call__(self, circuit_with_layer):
        proc_circuit = []
        for layer in circuit_with_layer:
            proc_circuit += self.circuit_from_single_layer(layer)
        return proc_circuit


def TEST_ProcCircuit():
    basic = Basic(4, [(0, 1), (1, 2), (2, 3), (3, 0)], 3)
    circ = [[['RZ', 'I', 'I', 'RY'], [(0, 1)]], [['RZ', 'I', 'RZ', 'RY'], [(1, 2)]]]
    pc = ProcCircuit(basic)
    print(pc(circ))

