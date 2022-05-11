import pennylane as qml
from pennylane import numpy as np
from QuantumProcess import objective_function_measurement, gradient_measurement, gradient_decent_one_step


def single_qubit_block():
    pass


def double_qubit_block():
    pass


def insertion(ansatz, parameters):
    new_ansatz = None
    new_parameters = None
    return new_ansatz, new_parameters


def simplification(ansatz, parameters):
    new_ansatz = None
    new_parameters = None
    return new_ansatz, new_parameters


def estimation(ansatz, parameters):
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

            new_cost, new_parameters = estimation(new_ansatz, new_parameters)

            new_ansatz, new_parameters = simplification(new_ansatz, new_parameters)

            cost_fn = objective_function_measurement(Hamiltonian, new_ansatz)
            new_cost = cost_fn(new_parameters)
            if is_accepted(cost, new_cost):
                ansatz = new_ansatz
                parameters = new_parameters
