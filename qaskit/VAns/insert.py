import numpy as np


def single_qubit_block(qb):
    return [('rz', qb), ('rx', qb), ('rz', qb)]


def double_qubit_block(c_qb, t_qb):
    return [('cx', c_qb, t_qb), ('rz', c_qb), ('rx', t_qb), ('rx', c_qb), ('rz', t_qb), ('rz', c_qb), ('rx', t_qb),
            ('cx', c_qb, t_qb)]


class IdInserter:
    def __init__(self, n_qubits=3, epsilon=0.1, initialization="epsilon", selector_temperature=5,coupling_set={(0,1), (1,2),(2,3)}):
        """
        Update:add coupling set
        epsilon: perturbation strength
        initialization: how parameters at ientity compilation are perturbated on single-qubit unitary.
                        Options = ["PosNeg", "epsilon"]
        """
        # super(IdInserter, self).__init__(n_qubits=n_qubits)
        self.n_qubits = n_qubits
        self.epsilon = epsilon
        self.init_params = initialization
        self.selector_temperature = selector_temperature
        """
            coupling set
        """
        self.coupling=coupling_set

        #self.qubits = cirq.GridQubit.rect(1, n_qubits)


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

        return number_of_cnots, indexed_cnots

    """ function 2:(utils)count gates on qubits """

    def gate_counter_on_qubits(self, indexed_circuit_1):
        index = []
        indexed_circuit = []
        gatecounter=[]
        for i in indexed_circuit_1:
            index.append(i)

        for i in range(len(index)):
            if index[i][0].lower() == 'cx':
                indexed_circuit.append(index[i][1])
                indexed_circuit.append(index[i][2])
            else:
                indexed_circuit.append(index[i][1])

        #ngates = {k: [0, 0] for k in range(self.n_qubits)}
        #cnot_num, cnot_ind = self.count_cnots()
        for i in range (self.n_qubits):
            gatecounter.append(indexed_circuit.count(i))

        return gatecounter

    """function 3:choose where to insert the block(somewhere in the circuit)"""

    def where_to_insert(self, indexed_circuit):
        if len(indexed_circuit) == self.n_qubits:
            insertion_index = self.n_qubits - 1
        elif len(indexed_circuit)==0:
            insertion_index=0
        else:
            insertion_index = np.squeeze(np.random.choice(range(0, len(indexed_circuit)), 1))
        return insertion_index

    """function 4:choose target block"""

    def choose_target_bodies(self, ngates=None, gate_type="one-qubit"):
        """
            gate_type: "one-qubit" or "two-qubit"
            ngates: gate_counter_on_qubits
            Note that selector_temperature could be annealed as energy decreases.. (at beta = 0 we get uniform sampling)
            function that selects qubit according to how many gates are acting on each one in the circuit
        """
        # set selector_temperature
        if ngates is None:
            ngates = {}
        if gate_type == "one-qubit":
            #print('ngates is',ngates)
            gc = ngates  #### gives the gate population for each qubit
            probs = np.exp(self.selector_temperature * (1 - gc / np.sum(gc))) / np.sum(
                np.exp(self.selector_temperature * (1 - gc / np.sum(gc))))
            #print('p=',probs)
            k=np.random.choice(range(self.n_qubits), 1, p=probs)[0]
            #print('choose',k)
            return k
            #return np.random.choice(range(self.n_qubits), 1, p=probs)[0]
        elif gate_type == "two-qubit":
            gc=ngates
            probs = np.exp(self.selector_temperature * (1 - gc / np.sum(gc))) / np.sum(
                np.exp(self.selector_temperature * (1 - gc / np.sum(gc))))
            flag=1
            qubits_out=None
            while(flag==1):
                qubits_ = np.random.choice(range(self.n_qubits), 2, p=probs, replace=False)
                qubits=tuple(qubits_)
                """try to find whether the random cnot is in the coupling set"""
                if qubits in self.coupling:
                    qubits_out=qubits
                    flag=0
                #print('choose is ',qubits_out)

            return qubits_out
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
                np.array(list(self.gate_counter_on_qubits(indexed_circuit))) < 1) <= self.n_qubits:
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



    def place_almost_identity(self, indexed_circuit, symbol_to_value):
        block_to_insert, insertion_index = self.choose_block(indexed_circuit)
        new_ansatz, new_parameters = insert(indexed_circuit, symbol_to_value, block_to_insert, insertion_index)
        return new_ansatz, new_parameters


    def insertion(self,indexed_circuit, symbol_to_value, rate_iids_per_step=1):
        ngates = np.random.exponential(scale=rate_iids_per_step)
        ngates = int(ngates + 1)
        #print('ngates=',ngates)
        M_ansatz, M_parameters = self.place_almost_identity(indexed_circuit, symbol_to_value)
        for ll in range(ngates - 1):
            M_ansatz, M_parameters = self.place_almost_identity(M_ansatz, M_parameters)
        return M_ansatz, M_parameters



def insert(ansatz1, parameters, block_to_insert, insert_index):
    new_ansatz = []
    new_parameters = []
    epsilon = 0.1
    # block_to_insert, insertion_index = self.choose_block(ansatz)
    block = []
    for i in block_to_insert:
        block.append(i)

    if len(ansatz1)==0:
        new_ansatz += block
        if len(block) == 3:
            new_parameters += ([np.random.choice([-1., 1.]) * epsilon for oo in range(3)])
        else:
            new_parameters += ([np.random.choice([-1., 1.]) * epsilon for oo in range(6)])
    else:
        index = []
        for i in ansatz1:
            index.append(i)

        for ind, gate in enumerate(index):
            """
            go through the whole ansatz
            ind is the index of gates 
            """
            par_count = 0
            if ind == insert_index:
                """add circuit"""
                new_ansatz+=block
                """add parameters: for block single qubit is 3, double is 6"""
                if len(block) == 3:
                    new_parameters+=([np.random.choice([-1., 1.]) * epsilon for oo in range(3)])
                else:
                    new_parameters+=([np.random.choice([-1., 1.]) * epsilon for oo in range(6)])
            new_ansatz.append(gate)
            k=str(gate[0])
            k=k.lower()
            if k == 'rz' or k == 'rx':
                """not new block then add original parameters to new_parameters"""
                new_parameters.append(parameters[par_count])
                par_count += 1
    return new_ansatz, new_parameters



"""
    testing....
"""

#ansatz=[('rx',1)]
#parameters=[0.0]
#idi = IdInserter(n_qubits=4, coupling_set={(0,1),(1,2),(2,3),(3,0)})
#new_ansatz, new_parameters = idi.insertion(ansatz, parameters)
#print('origin_ansatz is',ansatz)
#print('origin_para is',parameters)
#print('new_ansatz is',new_ansatz)
#print('new para is',new_parameters)
