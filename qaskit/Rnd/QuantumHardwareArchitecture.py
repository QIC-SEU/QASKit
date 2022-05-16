class QuantumHardwareArchitectureError(Exception):
    pass


class QuantumHardwareArchitecture:
    def __init__(self, hardware, coupling: set, **kwargs):
        self.hardware_simulator = hardware
        self.qubit_num = self.hardware_simulator.num_wires
        self.coupling = coupling.copy()
        self.err_prob_single = 0.001
        self.err_prob_cx = 0.01

    def get_coupling(self) -> set:
        return self.coupling.copy()

    def get_qubit_num(self) -> int:
        return self.qubit_num

    def get_quantum_device(self):
        return self.hardware_simulator
