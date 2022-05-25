from Basic import Basic
from Method import Method
import copy
import itertools


class SearchSpace(Method):
    def __init__(self, basic: Basic):
        super().__init__(basic)
        valid_su = ['I', 'RY', 'RZ']
        su_space = list(itertools.product(valid_su, repeat=self.n_qubits))
        CNOT_space = [[y for y in CNOTs if y is not None] for CNOTs in
                      list(itertools.product(*([x, None] for x in tuple(self.coupling))))]
        self.search_space = list(itertools.product(su_space, CNOT_space))

    def __call__(self, layer_id):
        return self.search_space[layer_id]
