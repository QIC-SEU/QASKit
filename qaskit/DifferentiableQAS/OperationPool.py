from Basic import Basic
from Method import Method
import itertools


class OperationPool(Method):
    def __init__(self, basic: Basic):
        super().__init__(basic)
        valid_su = ['I', 'RY', 'RZ']
        su_space = list(itertools.product(valid_su, repeat=self.n_qubits))
        CNOT_space = [[y for y in CNOTs if y is not None] for CNOTs in
                      list(itertools.product(*([x, None] for x in list(self.coupling))))]
        self.operation_pool = list(itertools.product(su_space, CNOT_space))

    def __call__(self, layer_id):
        """
        Get a layer from the operation identifier.
        :param layer_id: operation identifier of a layer.
        :return: a layer represented by a structured list [single_qubit_layer, cx_layer], where single_qubit_layer and
        cx_layer are lists.
        """
        layer = [list(self.operation_pool[layer_id][0]), list(self.operation_pool[layer_id][1])]
        return layer

    def __len__(self):
        return len(self.operation_pool)

    def circuit_from_id_list(self, id_list):
        """
        Get a circuit from a list of operation identifiers.
        :param id_list: list of operation identifiers
        :return: the circuit represented by a list of layers.
        """

        circuit = []
        for lay_id in id_list:
            circuit += self(lay_id)
        return circuit


def TEST_OperationPool():
    basic = Basic(4, [(0, 1), (1, 2), (2, 3), (3, 0)], 3)
    ss = OperationPool(basic)
    print(len(ss))
    print(len(ss.operation_pool))
    print(ss.operation_pool)
    print(ss(0))
    print(ss.circuit_from_id_list([0, 1, 2]))
