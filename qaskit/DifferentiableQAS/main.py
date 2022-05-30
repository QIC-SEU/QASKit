from DifferentiableQAS import DifferentiableQAS
from Hamiltonians import H2_Standard, Standard_to_Pennylane_Observables
import numpy as np


if __name__ == '__main__':
    Hamiltonian = H2_Standard
    bond = '0.7'
    n_qubits = 4
    coupling = [(i, np.mod(i+1, n_qubits)) for i in range(n_qubits)]
    n_samples = 100
    iterations = 500
    dqas = DifferentiableQAS(Hamiltonian=Standard_to_Pennylane_Observables(Hamiltonian, bond),
                             n_qubits=n_qubits,
                             coupling=coupling,
                             n_layers=3)
    results = dqas.training(n_samples, iterations)
    print(results)

