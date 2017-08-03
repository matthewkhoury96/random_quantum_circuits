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


def plot_k(all_data):
    """
    Plots k as a function of d or N in all types of circuits
    all_data is a dictionary with keys that are folders and values
    that are data_lists, each data_list is a list of dictionaries,
    each dictionary will provide an entire plot of k as a function of d
    For example: all_data['1D'][0] is the data from the 0th simulation
                 of the 1D lattice
    """
    folders = all_data.keys()

    for folder in folders:
        data_list = all_data[folder]
        for data in data_list:
            # Initialize a figure and some variables
            f_1, ax_1 = plt.subplots(figsize=(8, 4))
            n = data['n']
            m = data['m']
            k = data['k_mean']
            k_err = data['k_std']

            if folder != "CG":
                # Create some more variables
                x = data['d']
                w = int(folder[0])
                cutoff = int(np.power(n, 1 / w))
                shape = tuple(cutoff for i in range(w))
                prediction = n * (1 - 1 * np.power(1 / x, w))

                # Create some helper strings for 2D and 3D Lattice
                if w > 1:
                    shape_str = "Shape = {}, ".format(shape)
                    exp_str_pred = "^{}".format(w)
                    exp_str_cutoff = "^(1/{})".format(w)
                else:
                    shape_str = ""
                    exp_str_pred = ""
                    exp_str_cutoff = ""

                # Labels for 1D, 2D, or 3D Lattice
                x_label = "d = Depth of Circuit"
                y_label = "k such that Collision Probability = (1/2)^k"
                title = ("Collision Probability in a " +
                         "{} Lattice\n{}".format(folder, shape_str) +
                         "Number of Qubits = {}, ".format(n) +
                         "Number of Samples = {}".format(m))
                sim_label = "k from Simulations"
                pred_label = ("Prediction: k = " +
                              "n(1 - 1 / d{})".format(exp_str_pred))
                cutoff_label = "n{}".format(exp_str_cutoff)

            else:
                # Create some more variables
                x = data['N']
                cutoff = n * np.log(n)
                prediction = n * (1 - 1 / np.exp(x / n))

                # Labels for a Complete Graph
                x_label = "N = Number of Gates Applied"
                y_label = "k such that Collision Probability = (1/2)^k"
                title = ("Collision Probability in a Complete Graph\n" +
                         "Number of Qubits = " +
                         "{}, Number of Samples = {}".format(n, m))
                sim_label = "k from Simulations"
                pred_label = "Prediction: k = n(1 - 1 / e^(N/n))"
                cutoff_label = "n ln(n)"

            # Add labels to the figures
            ax_1.set_xlabel(x_label)
            ax_1.set_ylabel(y_label)
            ax_1.set_title(title)

            # Add plots to the figure
            ax_1.errorbar(x, k, yerr=k_err, fmt='--o', color='b',
                          label=sim_label, ms=4)
            ax_1.plot(x, prediction, 'r-', label=pred_label)
            ax_1.axvline(cutoff, color='g', linestyle='--',
                         label=cutoff_label)

            # Add the legend and save the plot
            ax_1.legend(loc=4)
            f_1.savefig(
                'plots/plots_{}/k_{}.pdf'.format(folder, n), dpi=200)


def plot_x_star(all_data, a):
    """
    Plots x_star as a function of n in all types of circuits
    all_data is a dictionary with keys that are folders and values
    that are data_lists, each data_list is a list of dictionaries,
    each dictionary will provide a data point on a plot
    For example: all_data['1D'][0] is the data from the 0th simulation
                 of the 1D lattice
    """
    folders = all_data.keys()

    for folder in folders:
        # Initialize a figure
        data_list = all_data[folder]
        f_1, ax_1 = plt.subplots(figsize=(8, 4))

        # The data for the plot will be stored in these lists
        n_vals = []
        x_star_vals = []
        x_star_err_vals = []

        # Extract x_star from the data_list
        for data in data_list:
            k = data['k_mean']
            k_err = data['k_std']
            n = data['n']
            if folder != 'CG':
                # For a lattice x is d
                x = data['d']
                c = n - a
            else:
                # For a complete graph x is N
                x = data['N']
                c = n - (n / a)
            x_star, x_star_err = get_x_star(x, k, k_err, c)
            n_vals.append(n)
            x_star_vals.append(x_star)
            x_star_err_vals.append(x_star_err)

        # Create some more variables
        n = np.array(n_vals)
        x_star = np.array(x_star_vals)
        x_star_err = np.array(x_star_err_vals)

        # Convert to log-log scales
        log_n = np.log(n)
        log_x_star = np.log(x_star)
        log_x_star_err = x_star_err / x_star

        # Add a linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            log_n, log_x_star)
        lin_reg = slope * log_n + intercept

        if folder != "CG":
            # Labels for the 1D, 2D, 3D Lattice
            x_label = "ln(n), n = Number of Qubits"
            y_label = "ln(d*), d* = depth such that k = n - {}".format(a)
            title = "d* in a {} Lattice".format(folder)
            sim_label = "ln(d*) from Simulations"
            w = int(folder[0])

            # Different prediction labels for 1D lattice
            if w > 1:
                pred_label = ("Prediction: ln(d*) = " +
                              "(1/{})(ln(n) - ln({}))".format(w, a))
            else:
                pred_label = ("Prediction: ln(d*) = " +
                              "ln(n) - ln({})".format(a))

            # Create prediction
            prediction = np.power(n / a, 1 / w)
            log_prediction = np.log(prediction)

        else:
            # Labels for the Complete Graph
            x_label = "ln(n), n = Number of Qubits"
            y_label = ("ln(N*), N* = number of gates such that " +
                       "k = n - n/{}".format(a))
            title = "N* in a Complete Graph"
            sim_label = "ln(N*) from Simulations"
            pred_label = "Prediction: ln(N*) = ln(n) + ln(ln({}))".format(a)

            # Create Prediction
            prediction = n * np.log(a)
            log_prediction = np.log(prediction)

        # Add labels to the figures
        ax_1.set_xlabel(x_label)
        ax_1.set_ylabel(y_label)
        ax_1.set_title(title)

        # Add plots to the figure
        ax_1.errorbar(log_n, log_x_star, yerr=log_x_star_err,
                      fmt='--o', color='b', label=sim_label, ms=4)
        ax_1.plot(log_n, log_prediction, 'r-', label=pred_label)
        ax_1.plot(log_n, lin_reg, 'g-',
                  label=("Linear Regression: (m, b, r) = " +
                         "({:1.2f}, {:1.2f}, {:1.2f})".format(
                             slope, intercept, r_value)))

        # Add the legend and save the figure
        ax_1.legend(loc=4)
        f_1.savefig('plots/plots_{}/x_star_{}.pdf'.format(folder, a), dpi=200)


def get_x_star(x, k, k_err, c):
    """
    Finds x_star such that k = c, where k is a function of x
    Uses a closed form solution in order to calculate the propogation of error
    Performs a Linear interpolation to find x_star
    """

    # First get the smallest index i where k[i] > c
    i = np.where(k > c)[0][0]

    # Get the two points that you will use for the linear interpolation
    x_1, y_1, y_1_err = (x[i - 1], k[i - 1], k_err[i - 1])
    x_2, y_2, y_2_err = (x[i], k[i], k_err[i])

    # Find closed form for d_star and d_star_err
    x_star = (((x_2 - x_1) * (c - y_1)) / (y_2 - y_1)) + x_1
    partial_y_1 = ((x_2 - x_1) * (c - y_2)) / np.power((y_2 - y_1), 2)
    partial_y_2 = ((x_2 - x_1) * (y_1 - c)) / np.power((y_2 - y_1), 2)
    x_star_err = np.sqrt(np.power(partial_y_1 * y_1_err, 2) +
                         np.power(partial_y_2 * y_2_err, 2))

    return (x_star, x_star_err)
