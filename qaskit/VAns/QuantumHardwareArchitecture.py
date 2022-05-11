class QuantumHardwareArchitectureError(Exception):
    pass


class QuantumHardwareArchitecture:
    def __init__(self, qubit_num, coupling: set, gate_set: dict):
        self.qubit_num = qubit_num
        self.coupling = coupling.copy()
        self.single_qubit_gate = gate_set['single']
        self.double_qubit_gate = gate_set['double']
        self.noise_model = kwargs.get('noise_model', None)

    def get_coupling(self) -> set:
        return self.coupling

    def get_qubit_num(self) -> int:
        return self.qubit_num

    def get_noise_model(self):
        return self.noise_model