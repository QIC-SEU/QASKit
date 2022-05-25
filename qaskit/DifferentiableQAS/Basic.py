class Basic:
    def __init__(self, n_qubits, coupling, **kwargs):
        self.n_qubits = n_qubits
        self.coupling = coupling
        self.noise_model = kwargs.get('noise_model', None)

