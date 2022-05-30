from DifferentiableQAS import DifferentiableQAS
from Hamiltonians import H2_Standard, Standard_to_Pennylane_Observables
from QuantumProcess import quantum_processor_launch, quantum_processor_terminate, set_noise_model
import numpy as np
import random

if __name__ == '__main__':
    random.seed(0)
    Hamiltonian = H2_Standard
    bond = '0.7'
    n_qubits = 4
    coupling = [(i, np.mod(i+1, n_qubits)) for i in range(n_qubits)]
    n_samples = 128
    iterations = 500
    set_noise_model({'depolarizing': {'single': 0.001, 'multi': 0.01}})
    quantum_processor_launch(n_qubits, proc_num=14)
    dqas = DifferentiableQAS(Hamiltonian=Standard_to_Pennylane_Observables(Hamiltonian, bond),
                             n_qubits=n_qubits,
                             coupling=coupling,
                             n_layers=4,
                             learning_rate=0.05,
                             early_stopping_threshold=0.31)
    results = dqas.training(n_samples, iterations)
    print(results)
    quantum_processor_terminate()

