from Basic import Basic
from Method import Method
import copy


class ParameterPool(Method):
    def __init__(self, basic: Basic):
        super().__init__(basic)
        self.parameter_pool = {}

    def get_layer_parameters(self, layer):
        # Transform the layer into the standard form in which alphabets are in upper format.
        standard_layer = copy.deepcopy(layer)
        for i in range(self.n_qubits):
            standard_layer[0][i] = standard_layer[0][i].upper()

        par = self.parameter_pool.get(str(layer), None)
        n_params = 0
        if par is None:
            for i in range(self.n_qubits):
                if layer[0][i] != 'I':
                    n_params += 1
            par = [0.0 for _ in range(n_params)]
        return par

    def __call__(self, circuit_with_layer):
        parameters = []
        for layer in range(circuit_with_layer):
            parameters += self.get_layer_parameters(layer)
        return parameters

    def update(self, parameters, circuit_with_layer):
        param_count = 0
        for layer in range(circuit_with_layer):
            dropped_par = self.get_layer_parameters(layer)
            self.parameter_pool[str(layer)] = parameters[param_count:param_count+dropped_par]
            param_count += dropped_par


