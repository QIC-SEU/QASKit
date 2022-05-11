import pennylane as qml
from pennylane import numpy as np
from QuantumProcess import objective_function_measurement, gradient_measurement

def single_qubit_block(qb):
    return [('rz', qb), ('rx', qb), ('rz', qb)]
    


def double_qubit_block(c_qb, t_qb):
    return [('cx', c_qb, t_qb), ('rz', c_qb), ('rx', t_qb), ('rx', c_qb), ('rz', t_qb), ('rz', c_qb), ('rx', t_qb), ('cx', c_qb, t_qb)]
    


def insertion(ansatz, parameters):
    new_ansatz = None
    new_parameters = None
    return new_ansatz, new_parameters


def simplification(ansatz, parameters):
    new_ansatz = None
    new_parameters = None
    return new_ansatz, new_parameters


def estimation(ansatz, parameters, cost_fn, grad_fn):
    cost = None
    trained_parameters = None
    return cost, trained_parameters


def is_accepted(cost, new_cost):
    return False


def VAns_training(training_steps, Hamiltonian):
    # Initialization
    ansatz = []
    parameters = []
    cost = None
    # Training
    for _ in range(training_steps):
        accept = False
        while not accept:
            new_ansatz, new_parameters = insertion(ansatz, parameters)

            new_ansatz, new_parameters = simplification(new_ansatz, new_parameters)

            cost_fn = objective_function_measurement(Hamiltonian, new_ansatz)
            grad_fn = gradient_measurement(Hamiltonian, new_ansatz)
            new_cost, new_parameters = estimation(new_ansatz, new_parameters, cost_fn, grad_fn)

            new_ansatz, new_parameters = simplification(new_ansatz, new_parameters)

            cost_fn = objective_function_measurement(Hamiltonian, new_ansatz)
            new_cost = cost_fn(new_parameters)
            if is_accepted(cost, new_cost):
                ansatz = new_ansatz
                parameters = new_parameters
