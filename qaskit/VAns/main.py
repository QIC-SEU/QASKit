from QuantumHardwareArchitecture import QuantumHardwareArchitecture
from Hamiltonians import H2_Standard, H4_Standard, HF_Standard, Standard_to_Pennylane_Observables
import argparse
import numpy as np
from VAns import VAns
import random
from QuantumProcess import quantum_processor_launch, quantum_processor_terminate, set_noise_model
# parser = argparse.ArgumentParser("VAns")
# parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
# parser.add_argument('--noisy', type=bool, default=False, help='use noise')
# parser.add_argument('--Hamiltonian', type=str, default='H2', help='Hamiltonian')
# parser.add_argument('--bond', type=str, default='0.7', help='The Hamiltonian.')
# parser.add_argument('--trials', type=int, default=100, help='The number of trials')
# args = parser.parse_args()


def get_Hamiltonian(hamil: str, bond):
    if hamil == 'H2':
        return Standard_to_Pennylane_Observables(H2_Standard, bond)
    elif hamil == 'H4':
        return Standard_to_Pennylane_Observables(H4_Standard, bond)
    elif hamil == 'HF':
        return Standard_to_Pennylane_Observables(HF_Standard, bond)
    else:
        raise Exception('Invalid Hamiltonian input.')


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    quantum_processor_launch(wires=4, shots=1000, proc_num=8)
    # Hamiltonian = get_Hamiltonian(args.Hamiltonian, args.bond)
    Hamiltonian = Standard_to_Pennylane_Observables(H2_Standard, '0.7')
    qubit_num = Hamiltonian['qubit']
    coupling = {(i, np.mod(i + 1, qubit_num)) for i in range(qubit_num)}
    gate_set = {'single': {'Ry', 'Rz'}, 'double': {'Cx'}}
    qha = QuantumHardwareArchitecture(qubit_num, coupling, gate_set)
    vans = VAns(qha, Hamiltonian)
    vans.training()
    quantum_processor_terminate()

