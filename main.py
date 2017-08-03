import utils
import chp_py
import numpy as np
import imp
import time


imp.reload(chp_py)
imp.reload(utils)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Examples of functions to perform different types of simulations, these  #
# are subject to change                                                   #
# NOTE: there are no test cases for these in test.py                      #
# FEEL FREE TO MODIFY the code below to run your own simulations          #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def simulate_qubit_pairs_1D_lattice(n, m):
    """
    Simulates a set of qubits in a 1D lattice of n qubits
    Performs two seperate rounds when applying gates
    to ensure that the depth of the circuit is accurate
    Runs the simulation a total of m times and returns the mean
    and std in the data dictionary
    """
    # Initialize some variables
    d = np.linspace(0, 2 * n, 50, dtype=int)
    k_matrix = np.zeros((m, len(d)))

    # Run the simulation a total of m times
    for i in range(m):
        start = time.time()

        sim = chp_py.CHP_Simulation(n)
        sims = []
        # Add a sim for depth 0
        sims.append(chp_py.CHP_Simulation(n, sim.state))

        for j in range(0, d[-1]):
            gates = utils.get_lattice_gates((n,), j)
            sim.apply_gates(gates)

            if j + 1 in d:
                sims.append(chp_py.CHP_Simulation(n, sim.state))

        # This computes the collision_probability at different
        # depths in parallel
        k_matrix[i, :] = utils.sims_to_collision_probabilty(sims)

        end = time.time()
        print("Time elapsed = {:0.2f}, for sample = {}".format(end - start, i))

    # Average collision_probability over m samples
    k_mean = np.mean(k_matrix, axis=0)
    k_std = np.std(k_matrix, axis=0)

    # Store everything in data
    data = {"d": d,
            "k_mean": k_mean,
            "k_std": k_std,
            "n": n,
            "m": m}

    return data


def simulate_qubit_pairs_2D_lattice(sqrt_n, m):
    """
    Simulates a set of qubits in a 2D square lattice of shape
    sqrt_n x sqrt_n. Performs four seperate rounds when applying gates
    to ensure that the depth of the circuit is accurate
    Runs the simulation a total of m times and returns the mean
    and std in the data dictionary
    """
    # Initialize some variables
    n = np.power(sqrt_n, 2)
    N = 35
    d = np.arange(0, N + 1)
    k_matrix = np.zeros((m, N + 1))

    # Run the simulation a total of m times
    for i in range(m):
        start = time.time()

        sim = chp_py.CHP_Simulation(n)
        sims = []
        # Add a sim for depth 0
        sims.append(chp_py.CHP_Simulation(n, sim.state))

        for j in range(N):
            gates = utils.get_lattice_gates((sqrt_n, sqrt_n), j)
            sim.apply_gates(gates)
            sims.append(chp_py.CHP_Simulation(n, sim.state))

        # This computes the collision_probability at different
        # depths in parallel
        k_matrix[i, :] = utils.sims_to_collision_probabilty(sims)

        end = time.time()
        print("Time elapsed = {:0.2f}, for sample = {}".format(end - start, i))

    # Average collision_probability over m samples
    k_mean = np.mean(k_matrix, axis=0)
    k_std = np.std(k_matrix, axis=0)

    # Store everything in data
    data = {"d": d,
            "k_mean": k_mean,
            "k_std": k_std,
            "n": n,
            "m": m}

    return data


def simulate_qubit_pairs_3D_lattice(cbrt_n, m):
    """
    Simulates a set of qubits in a 3D lattice grid of shape
    cbrt_n x cbrt_n x cbrt_n. Performs six seperate rounds when applying gates
    to ensure that the depth of the circuit is accurate
    Runs the simulation a total of m times and returns the mean
    and std in the data dictionary
    """
    # Initialize some variables
    n = np.power(cbrt_n, 3)
    N = 20
    d = np.arange(0, N + 1)
    k_matrix = np.zeros((m, N + 1))

    # Run the simulation a total of m times
    for i in range(m):
        start = time.time()

        sim = chp_py.CHP_Simulation(n)
        sims = []
        # Add a sim for depth 0
        sims.append(chp_py.CHP_Simulation(n, sim.state))

        for j in range(N):
            gates = utils.get_lattice_gates((cbrt_n, cbrt_n, cbrt_n), j)
            sim.apply_gates(gates)
            sims.append(chp_py.CHP_Simulation(n, sim.state))

        # This computes the collision_probability at different
        # depths in parallel
        k_matrix[i, :] = utils.sims_to_collision_probabilty(sims)

        end = time.time()
        print("Time elapsed = {:0.2f}, for sample = {}".format(end - start, i))

    # Average collision_probability over m samples
    k_mean = np.mean(k_matrix, axis=0)
    k_std = np.std(k_matrix, axis=0)

    # Store everything in data
    data = {"d": d,
            "k_mean": k_mean,
            "k_std": k_std,
            "n": n,
            "m": m}

    return data


def simulate_complete_graph(n, m):
    """
    Simulates qubits in a complete graph of n qubits
    Find k as a function of N = number of gates applied
    Performs simulation a total of m times then returns mean and std
    """
    N = np.linspace(0, 2 * n * np.log(n), 50, dtype=int)
    k_matrix = np.zeros((m, len(N)))

    # Run the simulation a total of m times
    for i in range(m):
        start = time.time()

        sim = chp_py.CHP_Simulation(n)
        sims = []
        # Add a sim for N = 0
        sims.append(chp_py.CHP_Simulation(n, sim.state))

        for j in range(0, N[-1]):
            qubits = np.arange(n)
            np.random.shuffle(qubits)
            q_1, q_2 = qubits[:2]
            gates = utils.get_random_two_qubit_gate(q_1, q_2)
            sim.apply_gates(gates)

            if j + 1 in N:
                sims.append(chp_py.CHP_Simulation(n, sim.state))

        # This computes the collision_probability at different
        # depths in parallel
        k_matrix[i, :] = utils.sims_to_collision_probabilty(sims)

        end = time.time()
        print("Time elapsed = {:0.2f}, for sample = {}".format(end - start, i))

    # Average collision_probability over m samples
    k_mean = np.mean(k_matrix, axis=0)
    k_std = np.std(k_matrix, axis=0)

    # Store everything in data
    data = {"N": N,
            "d": N / n,
            "k_mean": k_mean,
            "k_std": k_std,
            "n": n,
            "m": m}

    return data


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Examples of how to use the functions above to save data, load data,     #
# and create plots of the data. This code is subject to change and these  #
# are the only lines of code that will automatically be executed when you #
# run python main.py                                                      #
# NOTE: there are no test cases for these in test.py                      #
# FEEL FREE TO MODIFY the code below to run your own simulations          #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

folders = ['1D', '2D', '3D', 'CG']
sim_funcs = [simulate_qubit_pairs_1D_lattice,
             simulate_qubit_pairs_2D_lattice,
             simulate_qubit_pairs_3D_lattice,
             simulate_complete_graph]
all_sizes = [[25, 49, 64, 100, 169, 225, 324, 400, 529, 625],
             [25, 49, 64, 100, 169, 225, 324, 400, 529, 625, 784, 900],
             [27, 64, 125, 216, 343, 512, 729, 1000],
             [25, 49, 64, 100, 169, 225, 324, 400, 529, 625]]
modifiers = [np.abs, np.sqrt, np.cbrt, np.abs]
m = 25
a_list = [1, 5, 10, 15, 20]
all_data = dict()

# Run simulations, store and/or load data
for i in range(0, 4):
    folder = folders[i]
    sim_func = sim_funcs[i]
    sizes = all_sizes[i]
    mod = modifiers[i]
    data_list = []

    for n in sizes:
        print("\nStarting {} circuit with {} qubits".format(folder, n))
        start = time.time()

        # Perform simulation, store the data, then save a plot
        # Uncomment the next two lines to run your own simulations
        # data = sim_func(int(mod(n)), m)
        # utils.save_data(data, "/data_{}/data_{}_{}".format(folder, n, m))
        data = utils.load_data("/data_{}/data_{}_{}".format(folder, n, m))
        data_list.append(data)

        end = time.time()
        print("Finished {} circuit with {} qubits".format(folder, n))
        print("Total time elapsed = {}".format(end - start))

    # Add the data from a given folder to all_data
    all_data[folder] = data_list

# All plotting happens here
print("\nStarting plotting software")
utils.plot_k(all_data)
for a in a_list:
    utils.plot_x_star(all_data, a)
print("Finished saving all plots")
