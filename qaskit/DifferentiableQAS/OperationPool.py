from Basic import Basic
from Method import Method
import copy
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
        layer = [list(self.operation_pool[layer_id][0]), list(self.operation_pool[layer_id][1])]
        return layer

    def __len__(self):
        return len(self.operation_pool)


def TEST_OperationPool():
    basic = Basic(4, [(0, 1), (1, 2), (2, 3), (3, 0)], 3)
    ss = OperationPool(basic)
    print(len(ss))
    print(len(ss.operation_pool))
    print(ss.operation_pool)
    print(ss(0))
