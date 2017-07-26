import numpy as np
import matplotlib.pyplot as plt
import pickle
import multiprocessing as mp
import symplectic
import decompose
import chp_py
import imp
from scipy import stats

imp.reload(symplectic)
imp.reload(decompose)
imp.reload(chp_py)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# These functions are for converting between a 1-D array and an N-D grid  #
# and also getting the neighbors of a point in an N-D grid                #
# DO NOT MODIFY (unless you have a deep understanding of the code)        #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def index_to_coord(index, shape):
    """
    Converts a 1-D index into a coordinate in an array with the given shape
    """
    # Base case, if we have a 1-D grid, return index
    if len(shape) == 1:
        return (index,)

    # Otherwise, recurse
    prod = np.prod(shape[1:])
    c = index // prod
    next_index = index - (prod * c)
    next_shape = shape[1:]
    return((c,) + index_to_coord(next_index, next_shape))


def coord_to_index(coord, shape):
    """
    Converts a coordinate coord in an array with the given shape into an
    index of its corresponding 1-D array
    """
    # Base case, if we have a 1-D grid, return coord[0]
    if len(shape) == 1:
        return coord[0]

    # Otherwise, recurse
    prod = np.prod(shape[1:])
    index = (prod * coord[0])
    next_coord = coord[1:]
    next_shape = shape[1:]
    return(index + coord_to_index(next_coord, next_shape))


def cartesian_product(sequences):
    """
    Returns the cartesian product of the tuples in the list sequences
    """
    # Base case, if we only have one tuple in the list containing an
    # empty tuple
    if len(sequences) == 0:
        return([()])

    # Otherwise, recurse
    return([(i,) + j for j in cartesian_product(sequences[1:])
            for i in sequences[0]])


def get_neighbors_grid(index, shape, diag=True):
    """
    Gets the neighbors of a qubit at the given index organized in a
    grid/array with the given shape. If diag=True then will get all
    neighbors, including on diagonals. If diag=False, it will not include
    neighbors on the diagonals
    """
    # Initialize some variables
    coord = index_to_coord(index, shape)
    sequences = []

    # Create a sequence of tuples, where each tuple as a value of a
    # coordinate plus or minus one
    for i, dim in enumerate(shape):
        c = coord[i]
        sequence = ()

        for j in range(c - 1, c + 2):
            if (0 <= j < dim):
                sequence += (j,)

        sequences.append(sequence)

    # If diag, we take the cartesian_product of sequences to get
    # all of the neighbors, otherwise we can only modify one coordinate
    # at once, so we can iteratively create the list of neighbors
    if diag:
        neighbors_coords = cartesian_product(sequences)
    else:
        neighbors_coords = []
        for i, sequence in enumerate(sequences):
            for j in sequence:
                c = coord[:i] + (j,) + coord[i + 1:]
                neighbors_coords.append(c)

    # Remove duplicates and coord of the original qubit
    neighbors_coords = set(neighbors_coords)
    neighbors_coords.remove(coord)

    # Lastly, convert coords back to indices of qubits
    neighbors = [coord_to_index(c, shape) for c in neighbors_coords]
    return neighbors


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# These functions are for saving and loading data to disk, also includes  #
# a function to store all 2-qubit gates to disk                           #
# DO NOT MODIFY (unless you have a deep understanding of the code)        #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def save_data(data, file_name):
    """
    Stores the Python object data into the data folder with
    the given filename
    """
    with open("data/" + file_name + ".pickle", "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_data(file_name):
    """
    Loads the Python object in the data folder with
    the given filename
    """
    with open("data/" + file_name + ".pickle", "rb") as handle:
        data = pickle.load(handle)

    return data


def store_all_two_qubit_gates():
    m = 2
    qubit_gates = dict()
    qubit_matrices = dict()

    for i in range(symplectic.numberofsymplectic(m)):
        S = symplectic.symplectic(i, m)
        S = decompose.transform_symplectic(S)
        gates = decompose.decompose_state(chp_py.CHP_Simulation(m, S))
        qubit_gates[i] = gates
        qubit_matrices[i] = S

    save_data(qubit_gates, "two_qubit_gates")
    save_data(qubit_matrices, "two_qubit_matrices")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# These functions are for computing the collision probability of a list   #
# of simulations in parallel using a parallel map-reduce scheme           #
# DO NOT MODIFY (unless you have a deep understanding of the code)        #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def g(sim):
    return -sim.log_collision_probability


def sims_to_collision_probabilty(sims):
    """
    Given a list of simulations sims, returns a list of the
    collision_probability for each sim in sims
    """
    pool = mp.Pool()
    result = pool.map(g, sims)
    pool.close()
    return(result)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Make these a global variables, only load them once                      #
# DO NOT MODIFY (unless you have a deep understanding of the code)        #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
two_qubit_gates = load_data("two_qubit_gates")
two_qubit_matrices = load_data("two_qubit_matrices")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# These functions are for applying the 2-qubit gates in different         #
# rounds, so the qubits being affected in each round are deterministic    #
# and one round increases the depth of the circuit by one                 #
# DO NOT MODIFY (unless you have a deep understanding of the code)        #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def get_random_two_qubit_gate(q_1, q_2):
    """
    gets a random symplectic matrix decomposed into gates to apply
    directly to qubits q_1 and q_2
    """
    i = np.random.randint(len(two_qubit_gates))
    gates = two_qubit_gates[i]
    gates = decompose.change_gates(gates, (q_1, q_2))
    return gates


def get_lattice_gates(shape, r):
    """
    Gets a set of gates to apply to a lattice of the given shape
    on the given round r, which corresponds to the depth of the circuit
    NOTE: There are a total of 2 * len(shape) rounds
    """
    # Initialize some variables
    gates = []
    grid = np.arange(np.prod(shape)).reshape(shape)
    r = r % (2 * len(shape))

    # Get the sets of indices that we want
    # Start with all values, then find the one we want to modify
    a = [np.arange(s) for s in shape]
    b = [np.arange(s) for s in shape]
    i = r // len(shape)
    j = r % len(shape)

    # Modify the indices, j is the axis i determines the parity
    a[j] = np.arange(i, shape[j], 2)
    b[j] = np.arange(i + 1, shape[j], 2)

    # Grab the values from the grid
    a = grid[np.ix_(*a)]
    b = grid[np.ix_(*b)]

    # Reshape so they have the same size, basically remove extra values
    # Then flatten both arrays
    c = [np.arange(s) for s in min(a.shape, b.shape)]
    a = a[np.ix_(*c)].flatten()
    b = b[np.ix_(*c)].flatten()

    # Get the final set of coordinats
    coords = [(a[i], b[i]) for i in range(len(a))]

    # Get the gates using those coordinates
    for q_1, q_2 in coords:
        gates.extend(get_random_two_qubit_gate(q_1, q_2))

    return gates

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Examples of plotting functions, these are subject to change             #
# NOTE: there are no test cases for these in test.py                      #
#       some of the functions below will not work unless you have certain #
#       files with specific filenames in the data folder                  #
# FEEL FREE TO MODIFY the code below to create your own plots             #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def save_plot_1D_lattice(data):
    """
    Creates a plot using the data outputted collected from a 1D lattice
    plotting k as a function of d = depth of circuit
    """
    # Initialize some variables
    d = data['d']
    n = data['n']
    k = data['k_mean']
    k_err = data['k_std']
    m = data['m']
    # d_smooth = np.arange(d[-1])
    prediction = n * (1 - np.power(1 / d, 1))

    # plot k as a function of d
    f_1, ax_1 = plt.subplots(figsize=(24, 12))
    ax_1.set_xlabel("d = Depth of Circuit")
    ax_1.set_ylabel("k such that Collision Probability = (1/2)^k")
    ax_1.set_title("Collision Probability in a 1D Lattice, " +
                   "Number of Qubits = " +
                   "{}, Number of Samples = {}".format(n, m))
    ax_1.errorbar(d, k, yerr=k_err, fmt="--o", label="Simulation")
    ax_1.plot(d, prediction, 'r-', label="Prediction: k = n(1 - 1/d)")
    ax_1.axvline(n, color='g',
                 linestyle='--', label="n")
    ax_1.legend(loc=4)

    # Save the figure to the plots folder, then delete it
    f_1.savefig('plots/plots_1D/plot_{}.png'.format(n), dpi=200)
    f_1.clf()


def save_plot_2D_lattice(data):
    """
    Creates a plot using the data outputted collected from 2-qubit
    operations performed on a 2D square lattice
    Plots on a log log scale with error bars
    """
    # Initialize some variables
    d = data['d']
    k = data['k_mean']
    k_err = data['k_std']
    n = data['n']
    sqrt_n = int(np.sqrt(n))
    shape = (sqrt_n, sqrt_n)
    m = data['m']
    prediction_1 = n * (1 - np.power(1 / d, 2))

    # c = np.power(d, 2) * (1 - k / n)
    # p_coef = np.polyfit(d, c, 2)
    # p_string = "{:0.3f}d^2 + {:0.3f}d + {:0.3f}".format(*p_coef)
    # p = np.poly1d(p_coef)(d)
    # prediction_2 = n * (1 - (p * np.power(1 / d, 2)))

    # Add a linear regression in the plot
    # slope, intercept, r_value = stats.linregress(d[1:7], f[1:7])[:3]
    # d_lin = d[:8]
    # linear_reg = intercept + (slope * d_lin)

    # Create a figure plotting our simulation and our prediction
    f_1, ax_1 = plt.subplots(figsize=(24, 12))
    ax_1.set_xlabel("d = Depth of Circuit")
    ax_1.set_ylabel("k such that Collision Probability = (1/2)^k")
    ax_1.set_title("Collision Probability in a 2D lattice, " +
                   "Shape = {}, ".format(shape) +
                   "Number of Qubits = {}, ".format(n) +
                   "Number of Samples = {}".format(m))
    ax_1.errorbar(d, k, yerr=k_err, fmt="--o", label="Simulation")
    ax_1.plot(d, prediction_1, "r-", label="Prediction: k = n(1 - 1/d^2)")
    ax_1.axvline(sqrt_n, color='g',
                 linestyle='--', label="n^(1/2)")

    # ax_1.plot(d, prediction_2, "g-",
    # label="Prediction 2: k = n(1 - p/d^2), p = {}".format(p_string))
    # ax_1.plot(d_lin, linear_reg, "g-",
    #           label="Linear Regression with (m, b, r) = "
    #           + "({:0.2f}, {:0.2f}, {:0.2f})".format(
    #               slope, intercept, r_value))
    ax_1.legend(loc=4)

    # f_2, ax_2 = plt.subplots(figsize=(24, 12))
    # ax_2.plot(d, c, 'bo--', label='Simulation')
    # ax_2.plot(d, p, 'r-',
    #           label='p = polyfit(Simulation) = {}'.format(p_string))
    # ax_2.set_xlabel("d = Depth of Circuit")
    # ax_2.set_ylabel("c such that k = n(1 - c/d^2)")
    # ax_2.set_title("Collision Probability in a 2D Lattice, " +
    #                "Shape = {}, ".format(shape) +
    #                "Number of Qubits = {}, ".format(n) +
    #                "Number of Samples = {}".format(m))
    # ax_2.legend(loc=4)

    # Save the figure to the plots folder, then delete it
    f_1.savefig("plots/plots_2D/plot_{}.png".format(n), dpi=200)
    f_1.clf()

    # f_2.savefig('plots/plots_2D/plot_c_{}.png'.format(n), dpi=200)
    # f_2.clf()


def save_plot_3D_lattice(data):
    """
    Creates a plot using the data outputted collected from a 3D lattice
    plotting k as a function of d = depth of circuit
    """
    # Initialize some variables
    d = data['d']
    n = data['n']
    cbrt_n = int(np.cbrt(n))
    shape = (cbrt_n, cbrt_n, cbrt_n)
    k = data['k_mean']
    k_err = data['k_std']
    m = data['m']
    prediction_1 = n * (1 - 1 * np.power(1 / d, 3))

    # c = np.power(d, 3) * (1 - k / n)
    # p_coef = np.polyfit(d, c, 3)
    # p_string = "{:0.3f}d^3 + {:0.3f}d^2 + {:0.3f}d + {:0.3f}".format(*p_coef)
    # p = np.poly1d(p_coef)(d)
    # prediction_2 = n * (1 - p * np.power(1 / d, 3))

    # plot k as a function of d
    f_1, ax_1 = plt.subplots(figsize=(24, 12))
    ax_1.set_xlabel("d = Depth of Circuit")
    ax_1.set_ylabel("k such that Collision Probability = (1/2)^k")
    ax_1.set_title("Collision Probability in a 3D Lattice, " +
                   "Shape = {}, ".format(shape) +
                   "Number of Qubits = {}, ".format(n) +
                   "Number of Samples = {}".format(m))
    ax_1.errorbar(d, k, yerr=k_err, fmt="--o", label="Simulation")
    ax_1.plot(d, prediction_1, 'r-', label="Prediction: k = n(1-1/d^3)")
    ax_1.axvline(cbrt_n, color='g',
                 linestyle='--', label="n^(1/3)")

    # ax_1.plot(d, prediction_2, 'g-',
    # label="Prediction 2: k = n(1 - p/d^3), p = {}".format(p_string))
    ax_1.legend(loc=4)

    # f_2, ax_2 = plt.subplots(figsize=(24, 12))
    # ax_2.plot(d, c, 'bo--', label='Simulation')
    # ax_2.plot(d, p, 'r-',
    #           label='p = polyfit(Simulation) = {}'.format(p_string))
    # ax_2.set_xlabel("d = Depth of Circuit")
    # ax_2.set_ylabel("c such that k = n(1 - c/d^3)")
    # ax_2.set_title("Collision Probability in a 3D Lattice, " +
    #                "Shape = {}, ".format(shape) +
    #                "Number of Qubits = {}, ".format(n) +
    #                "Number of Samples = {}".format(m))
    # ax_2.legend(loc=4)

    # print("mean value = {}, std = {}".format(np.mean(c), np.std(c)))

    # Save the figure to the plots folder, then delete it
    f_1.savefig('plots/plots_3D/plot_{}.png'.format(n), dpi=200)
    f_1.clf()

    # f_2.savefig('plots/plots_3D/plot_c_{}.png'.format(n), dpi=200)
    # f_2.clf()


def save_plot_complete_graph(data):
    """
    Creates a plot using the data outputted collected from a complete graph
    plotting k as a function of N = number of gates
    """
    # Initialize some variables
    N = data['N']
    n = data['n']
    k = data['k_mean']
    k_err = data['k_std']
    m = data['m']
    # d = N / n
    # prediction_1 = n * (1 - np.power(1 / d, 6))

    # plot k as a function of N
    f_1, ax_1 = plt.subplots(figsize=(24, 12))
    # ax_1.set_ylim([int(0 - .2 * n), int(1.2 * n)])
    ax_1.set_xlabel("N = Number of Gates Applied")
    ax_1.set_ylabel("k such that Collision Probability = (1/2)^k")
    ax_1.set_title("Collision Probability in a Complete Graph, " +
                   "Number of Qubits = " +
                   "{}, Number of Samples = {}".format(n, m))
    ax_1.errorbar(N, k, yerr=k_err, fmt="--o", label="Simulation")
    # ax_1.plot(N, prediction_1, 'r-',
    #           label="Prediction: k = n(1 - 1/N)")
    ax_1.axvline(n * np.log(n), color='g',
                 linestyle='--', label="n ln(n)")

    ax_1.legend(loc=4)

    # Save the figure to the plots folder, then delete it
    f_1.savefig('plots/plots_CG/plot_{}.png'.format(n), dpi=200)
    f_1.clf()


def plot_all_shapes_k():
    # Initialize some variables
    f_1, ax_1 = plt.subplots(figsize=(24, 12))
    ax_1.set_xlabel("d = Depth of Circuit")
    ax_1.set_ylabel("k/n such that Collision Probability = (1/2)^k")
    ax_1.set_title("Collision Probability for Different Sized Circuits, "
                   + "Number of Samples = 25")
    colors = ['b', 'g', 'c', 'm', 'y', 'k']

    for i in range(5, 31, 5):
        m = 25
        n = i * i
        data = load_data("/data_2D/data_{}_{}".format(n, m))

        # Initialize some variables
        d = data['d']
        k = data['k_mean']
        k_err = data['k_std']

        prediction = (1 - np.power(1 / d, 2))
        f = k / n
        f_err = k_err / n

        ax_1.errorbar(d, f, yerr=f_err, fmt="--o", color=colors[i // 5 - 1],
                      label="Simulation for shape ({}, {})".format(i, i))

    ax_1.plot(d, prediction, "r-", label="Prediction: k/n = (1 - 1/d^2)")
    ax_1.legend(loc=4)
    # Save the figure to the plots folder, then delete it
    f_1.savefig('plots/plots_2D/k÷n.png', dpi=200)
    f_1.clf()


def get_d_star(d, k, k_err, n, c):
    """
    Finds d_star such that n - k = c, where n - k is a function of d
    Uses a closed form solution in order to calculate the propogation of error
    Performs a Linear interpolation to find d_star
    """

    # First get the index of the point closest to, but less than c
    i = np.where(n - k < c)[0][0]

    # Get the two points that you will use for the linear interpolation
    x_1, y_1, y_1_err = (d[i - 1], n - k[i - 1], k_err[i - 1])
    x_2, y_2, y_2_err = (d[i], n - k[i], k_err[i])

    # Find closed form for d_star and d_star_err
    d_star = (((x_2 - x_1) * (c - y_1)) / (y_2 - y_1)) + x_1
    partial_y_1 = ((x_2 - x_1) * (c - y_2)) / np.power((y_2 - y_1), 2)
    partial_y_2 = ((x_2 - x_1) * (y_1 - c)) / np.power((y_2 - y_1), 2)
    d_star_err = np.sqrt(np.power(partial_y_1 * y_1_err, 2) +
                         np.power(partial_y_2 * y_2_err, 2))

    return (d_star, d_star_err)


def plot_all_shapes_n_minus_k(c):
    # Initialize some variables
    # f_1 is the original plot
    # f_2 is the zoomed in plot with the values of d_star
    f_1, ax_1 = plt.subplots(figsize=(24, 12))
    ax_1.set_xlabel("d = Depth of Circuit")
    ax_1.set_ylabel("n - k such that Collision Probability = (1/2)^k")
    ax_1.set_title("Collision Probability for Different Sized Circuits, "
                   + "Number of Samples = 25")
    f_2, ax_2 = plt.subplots(figsize=(24, 12))
    ax_2.set_xlabel("d = Depth of Circuit")
    ax_2.set_ylabel("n - k such that Collision Probability = (1/2)^k")
    ax_2.set_title("Collision Probability for Different Sized Circuits, "
                   + "Number of Samples = 25")
    colors = ['b', 'g', 'c', 'm', 'y', 'k']
    d_star_vals = []
    d_star_err_vals = []
    m = 25

    # Collect data and add to the plots
    for i in range(5, 31, 5):
        n = i * i
        data = load_data("/data_2D/data_{}_{}".format(n, m))

        # get parts of data that we need
        d = data['d']
        k = data['k_mean']
        k_err = data['k_std']

        # add our prediction to the original plot
        prediction = n * (np.power(1 / d, 2))
        f = n - k
        f_err = k_err

        ax_1.plot(d, prediction, color=colors[i // 5 - 1],
                  label="Prediction: n - k = {}/d^2".format(n))

        ax_1.errorbar(d, f, yerr=f_err, fmt="--o", color=colors[i // 5 - 1],
                      label="Simulation for shape ({}, {})".format(i, i))

        # get d_star and error for the zoomed in plot
        d_star, d_star_err = get_d_star(d, k, k_err, n, c)
        d_star_vals.append(d_star)
        d_star_err_vals.append(d_star_err)

        ax_2.errorbar(d, f, yerr=f_err, fmt="--o", color=colors[i // 5 - 1],
                      label="Simulation for shape ({}, {})".format(i, i))

    # Add some extra stuff to the zoomed in plot
    ax_2.plot(d, c * np.ones(len(d)), 'r--', label="c = {}".format(c))
    ax_2.errorbar(d_star_vals, c * np.ones(len(d_star_vals)),
                  xerr=d_star_err_vals, fmt="X", color='r',
                  label="d* such that n - k = {}".format(c),
                  capsize=5)
    ax_2.set_ylim([0, 20])
    ax_2.set_xlim([0, 8])

    # Add the legends
    ax_1.legend(loc=1)
    ax_2.legend(loc=3)

    # Save both figures to the plots folder
    f_1.savefig('plots/plots_2D/n-k.png', dpi=200)
    f_2.savefig('plots/plots_2D/n-k_zoom.png', dpi=200)
    f_1.clf()
    f_2.clf()


def plot_d_star(c):
    # Initialize some variables
    d_star_vals = []
    d_star_err_vals = []
    n_vals = []
    m = 25

    # Get the values of d_star and d_star_err
    for i in range(5, 31, 5):
        n = i * i
        data = load_data("/data_2D/data_{}_{}".format(n, m))

        # Initialize some variables
        d = data['d']
        k = data['k_mean']
        k_err = data['k_std']

        d_star, d_star_err = get_d_star(d, k, k_err, n, c)
        d_star_vals.append(d_star)
        d_star_err_vals.append(d_star_err)
        n_vals.append(n)

    # Changing the names of variables for ease
    d_star = np.array(d_star_vals)
    d_star_err = np.array(d_star_err_vals)
    n = np.array(n_vals)
    n_smooth = np.arange(n_vals[0], n_vals[-1])
    prediction = np.sqrt(n_smooth / c)

    # For the log-log-scale
    log_d_star = np.log(d_star)
    log_d_star_err = np.abs(d_star_err / d_star)
    log_n = np.log(n)
    log_n_smooth = np.log(n_smooth)
    log_prediction = np.log(prediction)

    # Add a linear regression to the log-log-scale
    slope, intercept, r_value = stats.linregress(log_n, log_d_star)[:3]
    linear_reg = intercept + (slope * log_n)

    print(slope, intercept, r_value)

    # log-log plot
    f_1, ax_1 = plt.subplots(figsize=(12, 6))
    ax_1.set_xlabel("log(n), n = Number of Qubits")
    ax_1.set_ylabel("log(d*) such that n - k = 10")
    ax_1.set_title(
        "Depth for Different Sized Circuits for n - k = {}, ".format(c)
        + "Number of Samples = 25")
    ax_1.errorbar(log_n, log_d_star, yerr=log_d_star_err,
                  fmt="--o", color="b", label="Simulation")
    ax_1.plot(log_n_smooth, log_prediction,  'r-',
              label="Prediction: log(d*) = log((n/{})^(1/2))".format(c))

    ax_1.plot(log_n, linear_reg, "g-",
              label="Linear Regression with (m, b, r) = "
              + "({:0.3f}, {:0.3f}, {:0.3f})".format(
                  slope, intercept, r_value))

    ax_1.legend(loc=4)
    # Save the figure to the plots folder, then delete it
    f_1.savefig('plots/plots_2D/d_star_log_log.png', dpi=200)
    f_1.clf()

    # Original plot of d* as a function of n
    f_1, ax_1 = plt.subplots(figsize=(12, 6))
    ax_1.set_xlabel("n = Number of Qubits")
    ax_1.set_ylabel("d* such that n - k = 10")
    ax_1.set_title(
        "Depth for Different Sized Circuits for n - k = {}, ".format(c)
        + "Number of Samples = 25")
    ax_1.errorbar(n, d_star, yerr=d_star_err,
                  fmt="--o", color="b", label="Simulation")
    ax_1.plot(n_smooth, prediction,  'r-',
              label="Prediction: d* = (n/{})^(1/2)".format(c))
    ax_1.legend(loc=4)
    # Save the figure to the plots folder, then delete it
    f_1.savefig('plots/plots_2D/d_star.png', dpi=200)
    f_1.clf()


def change_font_size(big):
    """
    Makes the font size of the matplotlib plots
    The font gets bigger if big is true
    """
    if big:
        SMALL_SIZE = 16
        MEDIUM_SIZE = 20
        BIGGER_SIZE = 24
    else:
        SMALL_SIZE = 8
        MEDIUM_SIZE = 10
        BIGGER_SIZE = 12

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
