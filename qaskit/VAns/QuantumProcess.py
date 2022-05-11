import pennylane as qml
from pennylane import numpy as np
import multiprocessing
from pennylane._grad import grad as get_gradient


multiproc_quantum_processor = None

noise_model = {'noisy': False}

cumulative_quantum_timer = 0


def linear_fun(x):
    return x


def quantum_time_reset():
    global cumulative_quantum_timer
    cumulative_quantum_timer = 0
    print('Cumulative quantum timer has been reset.')


def get_quantum_time():
    return cumulative_quantum_timer


class encode_none:
    def __init__(self, config: tuple):
        pass

    def __call__(self, par):
        return par[0]

    @staticmethod
    def parameter_num():
        return 1


def set_noise_model(config):
    if multiproc_quantum_processor is not None:
        raise Exception("Please set noise model before quantum multi processor launching.")

    global noise_model
    noise_model = {'noisy': True}
    for noise_type in config:
        noise_model[noise_type] = config[noise_type]


def quantum_processor_launch(wires=4, shots=1000, proc_num=multiprocessing.cpu_count()):
    pool = multiprocessing.Pool()
    queue_in = multiprocessing.Manager().Queue()
    queue_out = multiprocessing.Manager().Queue()

    proc_list = [pool.apply_async(quantum_processor,
                                  (queue_in, queue_out,
                                   qml.device("default.mixed" if noise_model['noisy'] else "default.qubit",
                                              wires=wires, shots=shots), noise_model
                                   )) for i in range(proc_num)]
    global multiproc_quantum_processor
    multiproc_quantum_processor = {'pool': pool, 'reg': queue_in, 'res': queue_out}
    print('Quantum processor launched.')
    print('Noise Model:')
    print(noise_model)


def quantum_processor_terminate():
    import gc
    global multiproc_quantum_processor
    for i in range(multiprocessing.cpu_count()):
        multiproc_quantum_processor['reg'].put('Stop')
    multiproc_quantum_processor['pool'].close()
    multiproc_quantum_processor['pool'].join()
    multiproc_quantum_processor = None
    gc.collect()
    print('Quantum processor terminated.')


def quantum_processor(queue_in, queue_out, quantum_device, noise):
    def lfun(x):
        return x

    ideal_device = qml.device('default.qubit', wires=quantum_device.num_wires)
    # The ideal device is used to determine the exact objective function only.

    while True:
        if not queue_in.empty():
            task_inf = queue_in.get()
            if task_inf == 'Stop':
                return
            coe = task_inf.get('coe', 1)
            observable = task_inf['obs']
            identifier = task_inf['id']
            circuit = task_inf['circ']
            parameters = np.array(task_inf['par'])
            task_type = task_inf['type']
            fun = task_inf.get('fun', lfun)
            ideal = task_inf.get('ideal', False)
            encode = task_inf.get('encode', [encode_none for _ in range(len(parameters))])
            encode_config = task_inf.get('encode_config', [() for _ in range(len(encode))])
            @qml.qnode(ideal_device if ideal else quantum_device)
            def exe_circuit(par):
                index = 0
                parameter_index = 0
                for operation in circuit:
                    if operation[0].upper() == 'RY':
                        qml.RY(encode[index](encode_config[index])(
                            par[parameter_index: parameter_index + encode[index].parameter_num()]),
                            wires=operation[1])
                        if noise['noisy'] and not ideal:
                            if noise.get('depolarizing') is not None:
                                qml.DepolarizingChannel(noise['depolarizing']['single'], wires=operation[1])
                        parameter_index += encode[index].parameter_num()
                        index += 1
                    elif operation[0].upper() == 'RZ':
                        qml.RZ(encode[index](encode_config[index])(
                            par[parameter_index: parameter_index + encode[index].parameter_num()]),
                            wires=operation[1])
                        if noise['noisy'] and not ideal:
                            if noise.get('depolarizing') is not None:
                                qml.DepolarizingChannel(noise['depolarizing']['single'], wires=operation[1])
                        parameter_index += encode[index].parameter_num()
                        index += 1
                    elif operation[0].upper() == 'RX':
                        qml.RX(encode[index](encode_config[index])(
                            par[parameter_index: parameter_index + encode[index].parameter_num()]),
                            wires=operation[1])
                        if noise['noisy'] and not ideal:
                            if noise.get('depolarizing') is not None:
                                qml.DepolarizingChannel(noise['depolarizing']['single'], wires=operation[1])
                        parameter_index += encode[index].parameter_num()
                        index += 1
                    elif operation[0].upper() == 'CX' or operation[0].upper() == 'CNOT':
                        qml.CNOT(wires=[operation[1], operation[2]])
                        if noise['noisy'] and not ideal:
                            if noise.get('depolarizing') is not None:
                                qml.DepolarizingChannel(noise['depolarizing']['multi'], wires=operation[1])
                                qml.DepolarizingChannel(noise['depolarizing']['multi'], wires=operation[2])
                return qml.expval(observable)

            def obj_fn(par):
                return fun(coe * exe_circuit(par))

            if task_type == 'function':
                exp = obj_fn(parameters)
                task_res = {'id': identifier, 'type': task_type, 'res': exp}
                queue_out.put(task_res)
            elif task_type == 'gradient':
                grad = get_gradient(obj_fn)(parameters).reshape(len(parameters))
                task_res = {'id': identifier, 'type': task_type, 'res': grad}
                queue_out.put(task_res)


def objective_function_measurement(Hamiltonian, circuit_in, encode=None, encode_config=None, exact=False):
    """
    Measure the expectation of the Hamiltonian w.r.t the quantum state generated from the ansatz;
    :param Hamiltonian: A dictionary describing the Hamiltonian.
                structure:
                    {
                    'obs': [pennylane observable, ...],
                    'coe': [coefficient(number), ...],
                    'fun': [non-local callable function, ...],
                    'const': a constant(number)
                    }
                obj_fn = sum_{i} Hamiltonian['fun'][i](Hamiltonian['coe'][i] * (<0|U' Hamiltonian['obs'][i] U|0>));
    :param circuit_in: Parameterized quantum circuit U formed as a list of tuples. Each tuple is
                        structured as (type, q0, q1, ...);
                            Available type: 'rz', 'ry', 'cx';
                            Example: [('ry', 0), ('rz', 1), ('cx', 0, 1)];
    :param encode: A list of encode methods structured as a class;
    :param encode_config: A list of tuples representing encode configurations.
    :param exact: Determine the exact objective function value if True.
    :return: Objective function value.
    """
    import time

    if multiproc_quantum_processor is None:
        quantum_processor_launch()

    def exp_mea_multi_proc(parameters):
        """
        :param parameters: Trainable parameters. Note that the dimension must fit the circuit.
        :return: Expectation
        """
        global cumulative_quantum_timer
        time_start = time.time()
        number_left = len(Hamiltonian['obs'])
        for index in range(number_left):
            task_inf = {
                'coe': Hamiltonian['coe'][index],
                'obs': Hamiltonian['obs'][index],
                'fun': Hamiltonian['fun'][index],
                'id': index, 'circ': circuit_in,
                'noise': noise_model,
                'par': parameters, 'type': 'function',
                'ideal': exact
            }
            if encode is not None:
                task_inf['encode'] = encode
                if encode_config is not None:
                    task_inf['encode_config'] = encode_config
            multiproc_quantum_processor['reg'].put(task_inf)
        obj_fn_value = 0
        while number_left != 0:
            if not multiproc_quantum_processor['res'].empty():
                task_res = multiproc_quantum_processor['res'].get()
                obj_fn_value += task_res['res']
                number_left -= 1
        obj_fn_value += Hamiltonian.get('const', 0)
        time_end = time.time()
        cumulative_quantum_timer += time_end - time_start
        return obj_fn_value

    return exp_mea_multi_proc


def gradient_measurement(Hamiltonian, circuit_in, encode=None, encode_config=None):
    """
    Measure the gradient of the expectation of the Hamiltonian w.r.t the quantum state generated from the ansatz;
    :param Hamiltonian: A dictionary describing the Hamiltonian.
                structure:
                    {
                    'obs': [pennylane observable, ...],
                    'coe': [coefficient(number), ...],
                    'fun': [non-local callable function, ...],
                    'const': a constant(number)
                    }
                obj_fn = sum_{i} Hamiltonian['fun'][i](Hamiltonian['coe'][i] * (<0|U' Hamiltonian['obs'][i] U|0>));
    :param circuit_in: Parameterized quantum circuit U formed as a list of tuples. Each tuple is
                        structured as (type, q0, q1, ...);
                            Available type: 'rx', 'ry', 'cx';
                            Example: [('ry', 0), ('rz', 1), ('cx', 0, 1)];
    :param encode: A list of encode methods structured as a class;
    :param encode_config: A list of tuples representing encode configurations.
    :return: Gradient measurement function
    """
    import time

    def grad_mea_multi_proc(parameters):
        """
        :param parameters: Trainable parameters. Note that the dimension must fit the circuit.
        :return: Gradient
        """
        global cumulative_quantum_timer
        time_start = time.time()
        number_left = len(Hamiltonian['obs'])
        for index in range(number_left):
            task_inf = {
                'coe': Hamiltonian['coe'][index],
                'obs': Hamiltonian['obs'][index],
                'fun': Hamiltonian['fun'][index],
                'id': index, 'circ': circuit_in,
                'noise': noise_model,
                'par': parameters, 'type': 'gradient'
            }
            if encode is not None:
                task_inf['encode'] = encode
                if encode_config is not None:
                    task_inf['encode_config'] = encode_config
            multiproc_quantum_processor['reg'].put(task_inf)
        gradient = np.array([0.0 for i in range(len(parameters))]).reshape(len(parameters))
        while number_left != 0:
            if not multiproc_quantum_processor['res'].empty():
                task_res = multiproc_quantum_processor['res'].get()
                gradient += task_res['res']
                number_left -= 1
        time_end = time.time()
        cumulative_quantum_timer += time_end - time_start
        return gradient

    return grad_mea_multi_proc


def gradient_decent_one_step(grad, exp, exp_fun, parameters, fixed_learning_rate=False, learning_rate=10, c=1E-4,
                             rho=0.618):
    """
    Update the trainable parameters by one step;
    :param grad: Gradient of the cost function;
    :param exp: Cost function value;
    :param exp_fun: Callable cost function which take parameters as input;
    :param parameters: Trainable parameters. Note that the dimension must fit the circuit;
    :param fixed_learning_rate: Using fixed learning rate if true;
    :param learning_rate: Initialization of learning rate;
    :param rho: Shrink rate of optimization step size;
    :param c: constant of linear search;
    :return: Updated parameters and the number of measurements to find the learning rate.
    """

    if fixed_learning_rate:
        return {'parameters': parameters - learning_rate * grad, 'measurements': 0, 'learning_rate': learning_rate}
    else:
        measurements = 0

        def threshold(a):
            return exp - c * a * sum(grad ** 2)

        alpha = learning_rate
        new_par = parameters - alpha * grad
        new_exp = exp
        for i in range(2 * len(parameters)):
            new_par = parameters - alpha * grad
            new_exp = exp_fun(new_par)
            measurements += 1
            if new_exp <= threshold(alpha) and new_exp < exp:
                break
            else:
                alpha = alpha * rho

        return {'parameters': new_par, 'measurements': measurements, 'cost': new_exp, 'learning_rate': alpha}


def adam_one_step(grad, exp, parameters, learning_rate=0.1):
    pass