from QuantumHardwareArchitecture import QuantumHardwareArchitecture
from Hamiltonians import H2_Standard, H4_Standard, HF_Standard, Standard_to_Pennylane_Observables
import argparse
import numpy as np
from VAns import VAns
import random
from QuantumProcess import quantum_processor_launch, quantum_processor_terminate, set_noise_model, \
    gradient_decent_one_step

# parser = argparse.ArgumentParser("VAns")
# parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
# parser.add_argument('--noisy', type=bool, default=False, help='use noise')
# parser.add_argument('--Hamiltonian', type=str, default='H2', help='Hamiltonian')
# parser.add_argument('--bond', type=str, default='0.7', help='The Hamiltonian.')
# parser.add_argument('--trials', type=int, default=100, help='The number of trials')
# args = parser.parse_args()

#
# def get_Hamiltonian(hamil: str, bond):
#     if hamil == 'H2':
#         return Standard_to_Pennylane_Observables(H2_Standard, bond)
#     elif hamil == 'H4':
#         return Standard_to_Pennylane_Observables(H4_Standard, bond)
#     elif hamil == 'HF':
#         return Standard_to_Pennylane_Observables(HF_Standard, bond)
#     else:
#         raise Exception('Invalid Hamiltonian input.')


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)

    # Set environment and task
    set_noise_model({'depolarizing': {'single': 0.001, 'multi': 0.01}})
    Hamiltonian = Standard_to_Pennylane_Observables(H2_Standard, '0.7')
    qubit_num = Hamiltonian['qubit']
    coupling = {(i, np.mod(i + 1, qubit_num)) for i in range(qubit_num)}
    gate_set = {'single': {'Ry', 'Rz'}, 'double': {'Cx'}}
    qha = QuantumHardwareArchitecture(qubit_num, coupling, gate_set)
    quantum_processor_launch(wires=qubit_num, shots=10000, proc_num=16)

    # Set VAns configurations

    insertion_config = {'epsilon': 0.1, 'initialization': "epsilon", 'selector_temperature': 10}
    simplification_config = {'with_improvement_judgement': True, 'error_factor': 0.01}
    estimation_config = {'one_step_optimizer': gradient_decent_one_step,
                         'learning_rate': 0.2,
                         'stop_threshold': 1E-8,
                         'print_flag': True}
    training_config = {'training_steps': 100,
                       'stable_threshold': 10,
                       'print_flag': True}

    configurations = {'insertion_config': insertion_config, 'simplification_config': simplification_config,
                      'estimation_config': estimation_config, 'training_config': training_config}

    # Set VAns and train
    vans = VAns(qha, Hamiltonian, **configurations)
    optimal_ansatz, optimal_parameters, optimal_cost = vans.training()
    print(optimal_ansatz)
    print('cost = ' + str(optimal_cost))

    quantum_processor_terminate()
