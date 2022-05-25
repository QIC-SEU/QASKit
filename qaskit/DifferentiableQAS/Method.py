from Basic import Basic


class Method(Basic):
    def __init__(self, basic: Basic):
        super().__init__(basic.n_qubits, basic.coupling, basic.n_layers, noise_model=basic.noise_model)

    def __call__(self, *args, **kwargs):
        pass
