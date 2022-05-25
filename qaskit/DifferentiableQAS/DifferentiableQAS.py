from Basic import Basic


class DifferentiableQAS:
    def __init__(self, n_qubits, coupling, n_layers, noise_model=None):
        self.basic = Basic(n_qubits, coupling, n_layers, noise_model=noise_model)
