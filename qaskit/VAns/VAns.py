import pennylane as qml
from pennylane import numpy as np
from QuantumProcess import objective_function_measurement, gradient_measurement, gradient_decent_one_step, \
    optimal_cost_estimation
import math
from QuantumHardwareArchitecture import QuantumHardwareArchitecture
from Simplification import simplification
from insert import IdInserter


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

        self.optimal_ansatz = [('Rx', i) for i in range(self.n_qubits)]
        self.optimal_parameters = [0.0 for _ in range(self.n_qubits)]
        self.optimal_cost = None

        self.detail_record = []
        self.cumulative_measure_count = 0

        self.insertion_config = kwargs.get('insertion_config', {'epsilon': 0.1, 'initialization': "epsilon",
                                                                'selector_temperature': 10})
        self.estimation_config = kwargs.get('estimation_config',
                                            {'one_step_optimizer': gradient_decent_one_step,
                                             'learning_rate': 0.2,
                                             'stop_threshold': 1E-10,
                                             'print_flag': True})
        self.training_config = kwargs.get('training_config', {'training_steps': 100, 'stable_threshold': 25, 'print_flag': True})


    def estimation(self, ansatz, parameters, **kwargs):
        if kwargs.get('one_step_optimizer', None) is None:
            kwargs['one_step_optimizer'] = self.estimation_config['one_step_optimizer']
        Hamiltonian = kwargs['Hamiltonian'] if kwargs.get('Hamiltonian', None) is not None else self.Hamiltonian
        result = optimal_cost_estimation(Hamiltonian, ansatz, parameters, **self.estimation_config)
        record = result['detail']

        for piece in record:
            self.detail_record.append({'mea': self.cumulative_measure_count + piece['mea'], 'cost': piece['cost']})
        self.cumulative_measure_count += record[-1]['mea']
        cost = result['cost']
        trained_parameters = result['parameters']
        return cost, trained_parameters

    def insertion(self, ansatz, parameters):
        """
        Module encapsulated, see insert.py
        """
        idi = IdInserter(n_qubits=self.n_qubits, coupling_set=self.coupling, **self.insertion_config)

        new_ansatz, new_parameters = idi.insertion(ansatz, parameters)

        return new_ansatz, new_parameters

    def simplification(self, ansatz, parameters):
        reformulated_ansatz = []
        for tp in ansatz:
            if tp[0].lower() == 'cx':
                new_tp = (tp[0].lower(), tp[1], tp[2])
                reformulated_ansatz.append(new_tp)
            else:
                new_tp = (tp[0].lower(), tp[1])
                reformulated_ansatz.append(new_tp)
        return simplification(reformulated_ansatz, parameters)

    def training(self):
        stable_threshold = self.training_config.get('stable_threshold', 25)
        print_flag = self.training_config.get('print_flag', False)
        training_steps = self.training_config['training_steps']
        for iteration in range(training_steps):
            if print_flag:
                print('Iteration: ' + str(iteration))
            accept = False
            unaccepted_count = 0
            termination_flag = False
            while not accept:
                init_parameters = [0.0 for _ in range(len(self.optimal_parameters))]
                new_ansatz, new_parameters = self.insertion(self.optimal_ansatz, init_parameters)
                if print_flag:
                    print(new_ansatz)
                    print('parameter number: ' + str(len(new_parameters)))

                new_ansatz, new_parameters = self.simplification(new_ansatz, new_parameters)
                if print_flag:
                    print(new_ansatz)
                    print('parameter number: ' + str(len(new_parameters)))
                new_cost, new_parameters = self.estimation(new_ansatz, new_parameters)

                if print_flag:
                    print('trail mea: ' + str(self.cumulative_measure_count))
                    print('trail cost: ' + str(new_cost))
                    print('optimal cost: ' + str(self.optimal_cost))

                new_ansatz, new_parameters = self.simplification(new_ansatz, new_parameters)

                if print_flag:
                    print('simplified mea: ' + str(self.cumulative_measure_count))
                    print('simplified cost: ' + str(new_cost))
                    print('optimal cost: ' + str(self.optimal_cost))

                cost_fn = objective_function_measurement(self.Hamiltonian, new_ansatz)
                new_cost = cost_fn(new_parameters)
                if is_accepted(self.optimal_cost, new_cost):
                    if print_flag:
                        print('accepted')
                    self.optimal_ansatz = new_ansatz
                    self.optimal_parameters = new_parameters
                    self.optimal_cost = new_cost
                    accept = True
                else:
                    unaccepted_count += 1
                    print('without improvement: ' + str(unaccepted_count))
                    if unaccepted_count >= stable_threshold:
                        termination_flag = True
                        break
            if termination_flag:
                break
        return self.optimal_ansatz, self.optimal_parameters, self.optimal_cost

    def get_training_detail(self):
        return self.detail_record
