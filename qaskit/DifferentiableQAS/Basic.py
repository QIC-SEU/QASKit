class Basic:
    def __init__(self, n_qubits, coupling, n_layers, **kwargs):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.coupling = coupling
        self.noise_model = kwargs.get('noise_model', None)






