import pennylane as qml
from pennylane import numpy as np
from QuantumProcess import objective_function_measurement, gradient_measurement, gradient_decent_one_step
import numpy as np



def single_qubit_block(qb):
    return [('rz', qb), ('rx', qb), ('rz', qb)]


def double_qubit_block(c_qb, t_qb):
    return [('cx', c_qb, t_qb), ('rz', c_qb), ('rx', t_qb), ('rx', c_qb), ('rz', t_qb), ('rz', c_qb), ('rx', t_qb), ('cx', c_qb, t_qb)]




def insertion(n_qubits,ansatz, parameters):

    """ function 1:(utils)count cnots"""
    def count_cnots(n_qubits):
        indexed_cnots = {}
        cnots_index = {}
        count = 0
        for control in range(n_qubits):
            for target in range(n_qubits):
                if control != target:
                    indexed_cnots[str(count)] = [control, target]
                    cnots_index[str([control, target])] = count
                    count += 1
        number_of_cnots = len(indexed_cnots)
        return number_of_cnots, cnots_index

    """ function 2:(utils)count gates on qubits """
    def gate_counter_on_qubits(n_qubits, indexed_circuit):
        ngates = {k: [0, 0] for k in range(len(n_qubits))}
        cnot_num, cnot_ind = count_cnots(n_qubits)
        ind1,g=enumerate(indexed_circuit)
        for ind in ind1:
            if ind < cnot_num:
                control, target = cnot_ind[str(ind)]
                ngates[control][1] += 1
                ngates[target][1] += 1
            else:
                qind = (ind - cnot_num) % n_qubits
                ngates[qind][0] += 1
        return ngates

    """function 3:choose where to insert the block(somewhere in the circuit)"""
    def where_to_insert(n_qubits, indexed_circuit):
        if len(indexed_circuit) == n_qubits:
            insertion_index = n_qubits - 1
        else:
            insertion_index = np.squeeze(np.random.choice(range(n_qubits, len(indexed_circuit)), 1))
        return insertion_index

    """function 4:choose target block"""
    def choose_target_bodies(n_qubits, ngates={}, gate_type="one-qubit"):
        """
            gate_type: "one-qubit" or "two-qubit"
            ngates: gate_counter_on_qubits
            Note that selector_temperature could be annealed as energy decreases.. (at beta = 0 we get uniform sampling)
            function that selects qubit according to how many gates are acting on each one in the circuit
        """
        # set selector_temperature
        selector_temperature = 10

        if gate_type == "one-qubit":
            gc = np.array(list(ngates.values()))[:, 0] + 1  #### gives the gate population for each qubit
            probs = np.exp(selector_temperature * (1 - gc / np.sum(gc))) / np.sum(
                np.exp(selector_temperature * (1 - gc / np.sum(gc))))
            return np.random.choice(range(n_qubits), 1, p=probs)[0]
        elif gate_type == "two-qubit":
            gc = np.array(list(ngates.values()))[:, 1] + 1  #### gives the gate population for each qubit
            probs = np.exp(selector_temperature * (1 - gc / np.sum(gc))) / np.sum(
                np.exp(selector_temperature * (1 - gc / np.sum(gc))))
            qubits = np.random.choice(range(n_qubits), 2, p=probs, replace=False)
            return qubits
        else:
            raise NameError("typo code here.")

    """function 5:decide where to insert (place and target qubit(s)) and which block to insert"""
    def choose_block(n_qubits, indexed_circuit):
        """
        randomly choices an identity resolution and index to place it at indexed_circuit.
        """
        ngates = gate_counter_on_qubits(n_qubits, indexed_circuit)
        ### if no qubit is affected by a CNOT in the circuit... (careful, since this might be bias the search if problem is too easy)
        if np.count_nonzero(
                np.array(list(gate_counter_on_qubits(n_qubits, indexed_circuit).values()))[:, 1] < 1) <= n_qubits:
            which_block = np.random.choice([0, 1], p=[.2, .8])
            insertion_index = where_to_insert(n_qubits, indexed_circuit)
        else:
            which_block = np.random.choice([0, 1], p=[.5, .5])
            insertion_index = np.random.choice(max(1, len(indexed_circuit)))
        if which_block == 0:
            # qubit = np.random.choice(self.n_qubits)
            qubit = choose_target_bodies(n_qubits, ngates=ngates, gate_type="one-qubit")
            block_to_insert = single_qubit_block(qubit)
        else:
            qubits = choose_target_bodies(n_qubits, ngates=ngates, gate_type="two-qubit")
            # qubits = np.random.choice(self.n_qubits, 2,replace = False)
            block_to_insert = double_qubit_block(qubits[0], qubits[1])

        return block_to_insert, insertion_index

    """main body"""
    
    new_ansatz=[]
    new_parameters=[]
    epsilon=0.1
    block_to_insert,insertion_index=choose_block(n_qubits,ansatz)
    for ind,gate in enumerate(ansatz):
        """
        go through the whole ansatz
        ind is the index of gates 
        """
        par_count=0
        if ind == insertion_index:
            """add circuit"""
            new_ansatz.append(block_to_insert)
            """add parameters: for block single qubit is 3, double is 6"""
            if len(block_to_insert) ==3:
                new_parameters.append([np.random.choice([-1.,1.])*epsilon for oo in range(3)])
            else:
                new_parameters.append([np.random.choice([-1., 1.]) * epsilon for oo in range(6)])
        else:
            new_ansatz.append(gate)
            if gate[0] == 'rz' or gate[0] == 'rx':
                """not new block then add original parameters to new_parameters"""
                new_parameters.append(parameters[par_count])
                par_count+=1

    return new_ansatz, new_parameters


def simplification(ansatz, parameters):
    new_ansatz = None
    new_parameters = None
    return new_ansatz, new_parameters


def estimation(ansatz, parameters):
    cost = None
    trained_parameters = None
    return cost, trained_parameters


def is_accepted(cost, new_cost):
    return False


def VAns_training(training_steps, Hamiltonian):
    # Initialization
    ansatz = []
    parameters = []
    cost = None
    # Training
    for _ in range(training_steps):
        accept = False
        while not accept:
            new_ansatz, new_parameters = insertion(ansatz, parameters)

            new_ansatz, new_parameters = simplification(new_ansatz, new_parameters)

            new_cost, new_parameters = estimation(new_ansatz, new_parameters)

            new_ansatz, new_parameters = simplification(new_ansatz, new_parameters)

            cost_fn = objective_function_measurement(Hamiltonian, new_ansatz)
            new_cost = cost_fn(new_parameters)
            if is_accepted(cost, new_cost):
                ansatz = new_ansatz
                parameters = new_parameters
