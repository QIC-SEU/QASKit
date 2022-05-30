from Method import Method
from Basic import Basic
import numpy as np
import random


class AnsatzSampling(Method):
    def __init__(self, basic: Basic, size_of_operation_pool: int):
        super().__init__(basic)
        self.size_of_operation_pool = size_of_operation_pool

    def __call__(self, alpha: list, n_samples: int):
        """
        Sample n_samples ansatzes represented by a list of operation identifiers (int).
        :param alpha: parameters of soft_max sampling.
        :param n_samples: the number of samples.
        :return: a list of sampled ansatzes represented by a list of operation identifiers, e.g., [[1,2,3], [2,3,4]] for
        n_samples = 2, n_layers = 3.
        """
        samples = []
        identifiers = [i for i in range(self.size_of_operation_pool)]
        for sam in range(n_samples):
            sample = []
            for lay in range(self.n_layers):
                soft_max_component = [np.exp(alpha[lay][i]) for i in range(self.size_of_operation_pool)]
                sample.append(random.choices(identifiers, weights=soft_max_component, k=1)[0])
            samples.append(sample)
        return samples

    def greedy(self, alpha: list):
        """
        Get the circuit with the largest probability.
        :param alpha: parameters of soft_max sampling.
        :return: a list of operation identifiers with length n_layers.
        """
        circuit = []
        for lay in range(self.n_layers):
            soft_max_component = [np.exp(alpha[lay][i]) for i in range(self.size_of_operation_pool)]
            circuit.append(soft_max_component.index(max(soft_max_component)))
        return circuit


