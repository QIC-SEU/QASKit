import numpy as np
import cirq
from VAns import single_qubit_block,double_qubit_block

class IdInserter():
    def __init__(self, n_qubits=3,epsilon=0.1, initialization="epsilon", selector_temperature=10):
        """
        epsilon: perturbation strength
        initialization: how parameters at ientity compilation are perturbated on single-qubit unitary.
                        Options = ["PosNeg", "epsilon"]
        """
        super(IdInserter, self).__init__(n_qubits=n_qubits)
        self.n_qubits=n_qubits
        self.epsilon = epsilon
        self.init_params = initialization
        self.selector_temperature=selector_temperature
        self.qubits = cirq.GridQubit.rect(1, n_qubits)

    """ function 1:(utils)count cnots"""
    def count_cnots(self):
        indexed_cnots = {}
        cnots_index = {}
        count = 0
        for control in range(self.n_qubits):
            for target in range(self.n_qubits):
                if control != target:
                    indexed_cnots[str(count)] = [control, target]
                    cnots_index[str([control, target])] = count
                    count += 1
        number_of_cnots = len(indexed_cnots)
        return number_of_cnots, cnots_index

    """ function 2:(utils)count gates on qubits """

    def gate_counter_on_qubits(self, indexed_circuit):
        ngates = {k: [0, 0] for k in range(len(self.qubits))}
        cnot_num, cnot_ind = self.count_cnots()
        ind1, g = enumerate(indexed_circuit)
        for ind in ind1:
            if ind < cnot_num:
                control, target = cnot_ind[str(ind)]
                ngates[control][1] += 1
                ngates[target][1] += 1
            else:
                qind = (ind - cnot_num) % self.n_qubits
                ngates[qind][0] += 1
        return ngates

    """function 3:choose where to insert the block(somewhere in the circuit)"""

    def where_to_insert(self, indexed_circuit):
        if len(indexed_circuit) == self.n_qubits:
            insertion_index = self.n_qubits - 1
        else:
            insertion_index = np.squeeze(np.random.choice(range(self.n_qubits, len(indexed_circuit)), 1))
        return insertion_index

    """function 4:choose target block"""

    def choose_target_bodies(self, ngates={}, gate_type="one-qubit"):
        """
            gate_type: "one-qubit" or "two-qubit"
            ngates: gate_counter_on_qubits
            Note that selector_temperature could be annealed as energy decreases.. (at beta = 0 we get uniform sampling)
            function that selects qubit according to how many gates are acting on each one in the circuit
        """
        # set selector_temperature

        if gate_type == "one-qubit":
            gc = np.array(list(ngates.values()))[:, 0] + 1  #### gives the gate population for each qubit
            probs = np.exp(self.selector_temperature * (1 - gc / np.sum(gc))) / np.sum(
                np.exp(self.selector_temperature * (1 - gc / np.sum(gc))))
            return np.random.choice(range(self.n_qubits), 1, p=probs)[0]
        elif gate_type == "two-qubit":
            gc = np.array(list(ngates.values()))[:, 1] + 1  #### gives the gate population for each qubit
            probs = np.exp(self.selector_temperature * (1 - gc / np.sum(gc))) / np.sum(
                np.exp(self.selector_temperature * (1 - gc / np.sum(gc))))
            qubits = np.random.choice(range(self.n_qubits), 2, p=probs, replace=False)
            return qubits
        else:
            raise NameError("typo code here.")

    """function 5:decide where to insert (place and target qubit(s)) and which block to insert"""

    def choose_block(self, indexed_circuit):
        """
        randomly choices an identity resolution and index to place it at indexed_circuit.
        """
        ngates = self.gate_counter_on_qubits(indexed_circuit)
        ### if no qubit is affected by a CNOT in the circuit... (careful, since this might be bias the search if problem is too easy)
        if np.count_nonzero(
                np.array(list(self.gate_counter_on_qubits(indexed_circuit).values()))[:, 1] < 1) <= self.n_qubits:
            which_block = np.random.choice([0, 1], p=[.2, .8])
            insertion_index = self.where_to_insert(indexed_circuit)
        else:
            which_block = np.random.choice([0, 1], p=[.5, .5])
            insertion_index = np.random.choice(max(1, len(indexed_circuit)))
        if which_block == 0:
            # qubit = np.random.choice(self.n_qubits)
            qubit = self.choose_target_bodies(ngates=ngates, gate_type="one-qubit")
            block_to_insert = single_qubit_block(qubit)
        else:
            qubits = self.choose_target_bodies(ngates=ngates, gate_type="two-qubit")
            # qubits = np.random.choice(self.n_qubits, 2,replace = False)
            block_to_insert = double_qubit_block(qubits[0], qubits[1])

        return block_to_insert, insertion_index

    def insertion(self,ansatz, parameters,block_to_insert,insert_index):

        new_ansatz = []
        new_parameters = []
        epsilon = 0.1
        #block_to_insert, insertion_index = self.choose_block(ansatz)
        for ind, gate in enumerate(ansatz):
            """
            go through the whole ansatz
            ind is the index of gates 
            """
            par_count = 0
            if ind == insert_index:
                """add circuit"""
                new_ansatz.append(block_to_insert)
                """add parameters: for block single qubit is 3, double is 6"""
                if len(block_to_insert) == 3:
                    new_parameters.append([np.random.choice([-1., 1.]) * epsilon for oo in range(3)])
                else:
                    new_parameters.append([np.random.choice([-1., 1.]) * epsilon for oo in range(6)])
            else:
                new_ansatz.append(gate)
                if gate[0] == 'rz' or gate[0] == 'rx':
                    """not new block then add original parameters to new_parameters"""
                    new_parameters.append(parameters[par_count])
                    par_count += 1

        return new_ansatz, new_parameters



    def place_almost_identity(self, indexed_circuit, symbol_to_value):
        block_to_insert, insertion_index = self.choose_block(indexed_circuit)
        new_ansatz,new_parameters = self.insertion(indexed_circuit, symbol_to_value, block_to_insert, insertion_index)
        return new_ansatz,new_parameters


    def place_identities(self,indexed_circuit, symbol_to_value, rate_iids_per_step=1):
        ngates = np.random.exponential(scale=rate_iids_per_step)
        ngates = int(ngates+1)
        #print("Adding {}".format(ngates))
        M_ansatz, M_parameters = self.place_almost_identity(indexed_circuit, symbol_to_value)
        for ll in range(ngates-1):
            M_ansatz, M_parameters = self.place_almost_identity(M_ansatz, M_parameters)
        return M_ansatz, M_parameters

