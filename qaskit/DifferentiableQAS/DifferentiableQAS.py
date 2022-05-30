from Basic import Basic
from OperationPool import OperationPool
from ProcCircuit import ProcCircuit
from ParameterPool import ParameterPool
from AnsatzSampling import AnsatzSampling
from ProbModUpdate import ProbModUpdate
from QuantumProcess import objective_function_measurement, gradient_measurement
import numpy as np


class DifferentiableQAS:
    def __init__(self, Hamiltonian, n_qubits, coupling, n_layers, noise_model=None, learning_rate=0.1):
        self.Hamiltonian = Hamiltonian
        self.basic = Basic(n_qubits, coupling, n_layers, noise_model=noise_model)
        self.OperationPool = OperationPool(self.basic)
        self.ProcCircuit = ProcCircuit(self.basic)
        self.ParameterPool = ParameterPool(self.basic)
        self.AnsatzSampling = AnsatzSampling(self.basic, len(self.OperationPool))
        self.ProbModUpdate = ProbModUpdate(self.basic)
        self.alpha = [0.0 for _ in range(len(self.OperationPool))]
        self.learning_rate = learning_rate

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
                cost = cost_fn(par)
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

            # Get current best circuit
            current_best_id_list = self.AnsatzSampling.greedy(self.alpha)
            current_best_circuit = self.OperationPool.circuit_from_id_list(current_best_id_list)
            current_best_par = self.ParameterPool(current_best_circuit)
            current_best_cost_fn = objective_function_measurement(Hamiltonian=self.Hamiltonian,
                                                                  circuit_in=self.ProcCircuit(current_best_circuit))
            current_best_cost = current_best_cost_fn(current_best_par)
            print('Iteration: ' + str(it_index) + '; Cost: ' + str(current_best_cost))
        return {'structure_id': current_best_id_list,
                'circuit': current_best_circuit,
                'parameter': current_best_par,
                'cost': current_best_cost}
