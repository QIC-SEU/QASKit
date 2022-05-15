import pennylane as qml
from pennylane import numpy as np
from QuantumProcess import objective_function_measurement, gradient_measurement, gradient_decent_one_step, \
    optimal_cost_estimation
import math
from QuantumHardwareArchitecture import QuantumHardwareArchitecture
from Simplification import simplification

def single_qubit_block(qb):
    return [('rz', qb), ('rx', qb), ('rz', qb)]


def double_qubit_block(c_qb, t_qb):
    return [('cx', c_qb, t_qb), ('rz', c_qb), ('rx', t_qb), ('rx', c_qb), ('rz', t_qb), ('rz', c_qb), ('rx', t_qb),
            ('cx', c_qb, t_qb)]


def is_accepted(cost, new_cost):
    if cost is None:
        return True
    elif cost > new_cost:
        return True
    else:
        tmp_temperature = 10
        acceptance_percentage = 0.01
        return math.exp(-tmp_temperature * (new_cost - cost) / cost) < acceptance_percentage


class VAns:
    def __init__(self, qha: QuantumHardwareArchitecture, Hamiltonian, **kwargs):
        self.n_qubits = qha.get_qubit_num()
        self.coupling = qha.get_coupling()  # 允许的CNOT集，例如：{(0,1), (1,2), ..., (n-2, n-1), (n-1, 0)}，单向。
        self.Hamiltonian = Hamiltonian

        self.optimal_ansatz = None
        self.optimal_parameters = None
        self.optimal_cost = None

        self.detail_record = []
        self.cumulative_measure_count = 0

        self.insertion_config = kwargs.get('estimation_config', {'epsilon': 0.1, 'initialization': "epsilon",
                                                                 'selector_temperature': 10})
        self.estimation_config = kwargs.get('estimation_config', {'one_step_optimizer': gradient_decent_one_step})
        self.training_config = kwargs.get('training_config', {'training_steps': 100})

    def estimation(self, ansatz, parameters, **kwargs):
        one_step_optimizer = kwargs['one_step_optimizer'] if kwargs.get('one_step_optimizer', None) is not None \
            else self.estimation_config['one_step_optimizer']
        Hamiltonian = kwargs['Hamiltonian'] if kwargs.get('Hamiltonian', None) is not None else self.Hamiltonian
        result = optimal_cost_estimation(Hamiltonian, ansatz, parameters, one_step_optimizer, **self.estimation_config)
        record = result['detail']

        for piece in record:
            self.cumulative_measure_count += piece['mea']
            self.detail_record.append({'mea': self.cumulative_measure_count, 'cost': piece['cost']})

        cost = result['cost']
        trained_parameters = result['record']
        return cost, trained_parameters

    def insertion(self, ansatz, parameters):
        """
        Module encapsulated, see insert.py
        """
        new_ansatz = []
        new_parameters = []

        return new_ansatz, new_parameters

    def simplification(self, ansatz, parameters):
        return simplification(ansatz, parameters)

    def training(self):
        training_steps = self.training_config['training_steps']
        for _ in range(training_steps):
            accept = False
            while not accept:
                new_ansatz, new_parameters = self.insertion(self.optimal_ansatz, self.optimal_parameters)

                new_ansatz, new_parameters = self.simplification(new_ansatz, new_parameters)

                new_cost, new_parameters = self.estimation(new_ansatz, new_parameters)

                new_ansatz, new_parameters = self.simplification(new_ansatz, new_parameters)

                cost_fn = objective_function_measurement(self.Hamiltonian, new_ansatz)
                new_cost = cost_fn(new_parameters)
                if is_accepted(self.optimal_cost, new_cost):
                    self.optimal_ansatz = new_ansatz
                    self.optimal_parameters = new_parameters
                    self.optimal_cost = new_cost
                    accept = True

    def get_training_detail(self):
        return self.detail_record
