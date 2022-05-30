from Basic import Basic
from OperationPool import OperationPool
from ProcCircuit import ProcCircuit
from ParameterPool import ParameterPool
from AnsatzSampling import AnsatzSampling
from ProbModUpdate import ProbModUpdate
from QuantumProcess import objective_function_measurement, gradient_measurement, gradient_decent_one_step
import numpy as np


class DifferentiableQAS:
    def __init__(self, Hamiltonian, n_qubits, coupling, n_layers, noise_model=None, learning_rate=0.1,
                 early_stopping_threshold=0.3):
        self.Hamiltonian = Hamiltonian
        self.basic = Basic(n_qubits, coupling, n_layers, noise_model=noise_model)
        self.OperationPool = OperationPool(self.basic)
        self.ProcCircuit = ProcCircuit(self.basic)
        self.ParameterPool = ParameterPool(self.basic)
        self.AnsatzSampling = AnsatzSampling(self.basic, len(self.OperationPool))
        self.ProbModUpdate = ProbModUpdate(self.basic)
        self.alpha = [[0.0 for _ in range(len(self.OperationPool))] for i in range(n_layers)]
        self.learning_rate = learning_rate
        self.early_stopping_threshold = early_stopping_threshold

    def early_stopping(self):
        variance = 0
        for lay in range(self.basic.n_layers):
            soft_max_component = [np.exp(self.alpha[lay][i]) for i in range(len(self.OperationPool))]
            summation = sum(soft_max_component)
            soft_max_prob = np.array([soft_max_component[i] / summation for i in range(len(soft_max_component))])
            variance += soft_max_prob.std() / self.basic.n_layers

        print(variance)
        if variance > self.early_stopping_threshold:
            return True
        else:
            return False

    def training(self, n_samples, iterations):
        current_best_id_list = []
        current_best_circuit = []
        current_best_par = []
        current_best_cost = None
        for it_index in range(iterations):
            # Sampling n_samples ansatzes based on the probabilistic model
            sampled_id_lists = self.AnsatzSampling(self.alpha, n_samples)

            # Costs and gradients of sampled ansatzes
            costs = []
            gradients = []

            # Parameters of sampled ansatzes
            params = []

            for id_list in sampled_id_lists:
                # Get circuit from the id_list
                circuit = self.OperationPool.circuit_from_id_list(id_list)

                # Generate the cost function and gradient function
                cost_fn = objective_function_measurement(Hamiltonian=self.Hamiltonian,
                                                         circuit_in=self.ProcCircuit(circuit))
                grad_fn = gradient_measurement(Hamiltonian=self.Hamiltonian,
                                               circuit_in=self.ProcCircuit(circuit))

                # Get parameters from the parameter pool
                par = self.ParameterPool(circuit)
                params.append(par)

                # Calculate the cost function value and the gradient
                cost = float(cost_fn(par))
                grad = grad_fn(par)
                costs.append(cost)
                gradients.append(grad)

            # Update parameter pool
            for circ_index in range(len(sampled_id_lists)):
                # Get circuit from the id_list
                id_list = sampled_id_lists[circ_index]
                circuit = self.OperationPool.circuit_from_id_list(id_list)

                # Get parameters from the parameter pool
                par = self.ParameterPool(circuit)
                par = list(np.array(par) - self.learning_rate * gradients[circ_index])
                self.ParameterPool.update(par, circuit)

            # Update alpha the parameters of probabilistic model
            self.alpha = self.ProbModUpdate(self.alpha, sampled_id_lists, costs, self.learning_rate)

            # Get current best circuit
            current_best_id_list = self.AnsatzSampling.greedy(self.alpha)
            current_best_circuit = self.OperationPool.circuit_from_id_list(current_best_id_list)
            current_best_par = self.ParameterPool(current_best_circuit)
            current_best_cost_fn = objective_function_measurement(Hamiltonian=self.Hamiltonian,
                                                                  circuit_in=self.ProcCircuit(current_best_circuit))
            current_best_cost = current_best_cost_fn(current_best_par)
            print('Iteration: ' + str(it_index) + '; Cost: ' + str(current_best_cost))
            print(self.alpha)
            if self.early_stopping():
                print('Early stopped.')
                break
        last_learning_rate = 0.5
        opt_id_list = self.AnsatzSampling.greedy(self.alpha)
        opt_circuit = self.OperationPool.circuit_from_id_list(opt_id_list)
        opt_par = self.ParameterPool(current_best_circuit)
        cost_fn = objective_function_measurement(Hamiltonian=self.Hamiltonian,
                                                 circuit_in=self.ProcCircuit(opt_circuit))
        grad_fn = gradient_measurement(Hamiltonian=self.Hamiltonian,
                                       circuit_in=self.ProcCircuit(opt_circuit))
        opt_cost = cost_fn(opt_par)
        for i in range(50):
            cost = cost_fn(opt_par)
            grad = grad_fn(opt_par)
            update = gradient_decent_one_step(grad, cost, cost_fn, opt_par, learning_rate=last_learning_rate * 2)
            opt_par = update['parameters']
            opt_cost = update['cost']
            last_learning_rate = update['learning_rate']
            print('Iteration: ' + str(i) + '; Cost: ' + str(opt_cost))

        return {'structure_id': opt_id_list,
                'circuit': opt_circuit,
                'parameter': opt_par,
                'cost': opt_cost}
