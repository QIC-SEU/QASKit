import pennylane as qml
from pennylane import numpy as np
from QuantumProcess import objective_function_measurement, gradient_measurement, gradient_decent_one_step, \
    optimal_cost_estimation
import math
import re


detail_record = []
cumulative_measure_count = 0


def single_qubit_block(qb):
    return [('rz', qb), ('rx', qb), ('rz', qb)]


def double_qubit_block(c_qb, t_qb):
    return [('cx', c_qb, t_qb), ('rz', c_qb), ('rx', t_qb), ('rx', c_qb), ('rz', t_qb), ('rz', c_qb), ('rx', t_qb),
            ('cx', c_qb, t_qb)]


def insertion(n_qubits, ansatz, parameters):
    """
    Module encapsulated, see insert.py
    """
    new_ansatz = []
    new_parameters = []

    return new_ansatz, new_parameters


def simplification(n_qubits,ansatz, parameters):
    """
    Simplifies the circuit according to some rules that preserve the expected value of target hamiltornian.
    takes help from index_to_symbol (dict) and symbol_to_value (dict).
    Importantly, it keeps the parameter values of the untouched gates.
    Works on circuits containing CNOTS, Z-rotations and X-rotations.It applies the following rules:
    Rules:  1. CNOT just after initializing, it does nothing (if |0> initialization).
            2. Two consecutive and equal CNOTS compile to identity.
            3. Rotation around z axis of |0> only adds phase hence leaves invariant <H>. It kills it.
            4. two equal rotations: add the values.
            5. Scan for U_3 = Rz Rx Rz, or Rx Rz Rx; if found, abosrb consecutive rz/rx (until a CNOT is found)
            6. Scan qubits and abosrb rotations  (play with control-qubit of CNOT and rz)
            7. Scan qubits and absorb rotations if possible (play with target-qubit of CNOT and rx)
            8. Rz(control) and CNOT(control, target) Rz(control) --> Rz(control) CNOT
            9. Rx(target) and CNOT(control, target) Rx(target) --> Rx(target) CNOT
    Finally, if the circuit becomes too short, for example, there're no gates at a given qubit, an Rx(0) is placed.
    """

    new_ansatz = None
    new_parameters = None
    return new_ansatz, new_parameters


def estimation(ansatz, parameters, one_step_optimizer, **kwargs):
    global cumulative_measure_count, detail_record
    Hamiltonian = kwargs['Hamiltonian']
    result = optimal_cost_estimation(Hamiltonian, ansatz, parameters, one_step_optimizer, **kwargs)
    record = result['detail']

    for piece in record:
        cumulative_measure_count += piece['mea']
        detail_record.append({'mea': cumulative_measure_count, 'cost': piece['cost']})

    cost = result['cost']
    trained_parameters = result['record']
    return cost, trained_parameters


def is_accepted(cost, new_cost):
    if cost > new_cost:
        return True
    else:
        tmp_temperature = 10
        acceptance_percentage = 0.01
        return math.exp(-tmp_temperature * (new_cost - cost) / cost) < acceptance_percentage


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
