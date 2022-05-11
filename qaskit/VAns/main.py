from QuantumHardwareArchitecture import QuantumHardwareArchitecture
from Hamiltonians import H2_Standard, H4_Standard, HF_Standard, Standard_to_Pennylane_Observables
import argparse
import numpy as np

parser = argparse.ArgumentParser("VAns")
parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
parser.add_argument('--noisy', type=bool, default=False, help='use noise')
parser.add_argument('--Hamiltonian', type=str, default='H2', help='Hamiltonian')
parser.add_argument('--bond', type=str, default='0.7', help='The Hamiltonian.')
parser.add_argument('--trials', type=int, default=100, help='The number of trials')
args = parser.parse_args()


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
    Hamiltonian = get_Hamiltonian(args.Hamiltonian, args.bond)
    qubit_num = Hamiltonian['qubit']
    coupling = {(i, np.mod(i + 1, qubit_num)) for i in range(qubit_num)}
    gate_set = {'single': {'Ry', 'Rz'}, 'double': {'Cx'}}
    qha = QuantumHardwareArchitecture(qubit_num, coupling, gate_set)
