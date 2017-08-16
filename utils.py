import numpy as np
import matplotlib.pyplot as plt
import pickle
import multiprocessing as mp
import symplectic
import decompose
import chp_py
import imp
from scipy import stats
plt.rc('font', family='serif')

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
# These functions are for computing the expected value and standard       #
# deviation of the collision probability given a list of values for k     #
# DO NOT MODIFY (unless you have a deep understanding of the code)        #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def collision_probability_mean_and_std(k_matrix, s):
    """
    Given an k_matrix where k_matrix[i,j] is k = -log2(collision_probability)
    for the i^th sample and j^th value of d or N, this function returns four
    arrays mean_log, mean_err_log, std_log, and std_err_log. These are computed
    by dividing the m samples into m/s sets of size s, we then take the
    mean and standard deviation of the m/s sets. We then have mean_log as the
    -log2 of the mean of the means, mean_log_err as the corresponding standard
    error, std_log as the -log2 of the mean of the stds, and std_err_log as the
    corresponding standard error. Calculations here are done to avoid underflow
    """
    # Initialize some variables
    m, x = k_matrix.shape
    mean_log = np.zeros(x)
    std_log = np.zeros(x)
    mean_err_log = np.zeros(x)
    std_err_log = np.zeros(x)

    for i in range(x):
        # k_vec is a set of m samples
        k_vec = k_matrix[:, i]

        # Divide the m samples into m/s sets of size s
        k_sets = k_vec.reshape(m // s, s)
        # Add a small perturbation to avoid a set with
        # zero std to avoid infs and nans
        k_sets[:, -1] += 1e-3

        # mu[i], and sigma[i] are the -log2 of the mean and std
        # of the i^th set in k_sets
        vals = [log_mean_and_std(w) for w in k_sets]
        mu = np.array([val[0] for val in vals])
        sigma = np.array([val[1] for val in vals])

        # a is the -log2 of the mean of the values in mu
        # b is the -log2 of the standard error of the values in mu
        # c is the -log2 of the mean of the values in sigma
        # d is the -log2 of the standard error of the values in sigma
        # NOTE: standard error is std/sqrt(# samples)
        a, b = log_mean_and_std(mu)
        b += (1 / 2) * np.log2(len(mu))
        c, d = log_mean_and_std(sigma)
        d += (1 / 2) * np.log2(len(sigma))

        # a and c are the final values for mean_log and std_log
        mean_log[i] = a
        std_log[i] = c

        # (1/2)^b and (1/2)^d are the errors for the actual
        # mean and std respectively, to compute the errors
        # for the -log2(mean) and -log2(std) we must use  the
        # propogation of uncertainty formula
        mean_err_log[i] = np.power(1 / 2, b - a) * (1 / np.log(2))
        std_err_log[i] = np.power(1 / 2, d - c) * (1 / np.log(2))

    return (mean_log, mean_err_log, std_log, std_err_log)


def log_mean_and_std(w):
    """
    Given a vector w, returns -log2(mean(w)) and -log2(std(w))
    uses log_sum_exp to avoid underflow
    """
    m = len(w)
    mu = np.log2(m) - log_sum_exp(w)
    v = np.hstack((2 * w, 2 * mu * np.ones(m), w + mu - 1))
    c = np.hstack((np.ones(2 * m), -np.ones(m)))
    sigma = ((1 / 2) * np.log2(m)) - ((1 / 2) * log_sum_exp(v, c))

    return(mu, sigma)


def log_sum_exp(v, c=None):
    """
    A modified version of the LogSumExponential function
    Given a vector v = [v_1, v_2, ..., v_n], and returns
    log2((1/2)^v_1 + (1/2)^(v_2) + ... + (1/2)^(v_n))
    avoiding undeflow errors by factoring out the max value in v
    Optional parameter: s is a vector of length n and values of
    co-efficients, in the sum the i^th term will have sign s[i]
    """
    v_star = np.max(v)
    if c is None:
        c = np.ones(len(v))
    result = np.log2(np.sum(c * np.power(1 / 2, v - v_star)))
    result = result - v_star
    return result


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Examples of plotting functions, these are subject to change             #
# NOTE: there are no test cases for these in test.py                      #
#       some of the functions below will not work unless you have certain #
#       files with specific filenames in the data folder                  #
# FEEL FREE TO MODIFY the code below to create your own plots             #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def plot_collision_probability(all_data, s):
    """
    Plots collision probability as a function of d or N and also
    plots collision probability as a function of n
    all_data is a dictionary with keys that are folders and values
    that are data_lists, each data_list is a list of dictionaries,
    each dictionary will provide an entire plot of k as a function of d
    For example: all_data['1D'][0] is the data from the 0th simulation
                 of the 1D lattice
    """
    folders = all_data.keys()

    for folder in folders:
        # Each folder corresponds to a different type of circuit
        data_list = all_data[folder]

        # These values collected here are used to plot cp_mean and cp_std
        # as a function of n for saturated depths
        cp_mean_log_vals = []
        cp_mean_err_log_vals = []
        cp_std_log_vals = []
        cp_std_err_log_vals = []
        n_vals = []
        # The third from the last depth is usually saturated in our simulations
        d_index = -3

        for data in data_list:
            # data contains the k_matrix for a fixed value of n for the circuit
            # type specified by the folder

            # Get the values for the collision_probability for a specific
            # k_matrix, this is used to plot the mean and std of the
            # collision_probability with error bars as a function of d or N
            cp_mean_log, cp_mean_err_log, cp_std_log, cp_std_err_log = (
                collision_probability_mean_and_std(data['k_matrix'], s))

            # Create some helper variables
            n = data['n']
            types = ['cp_mean_log', 'cp_std_log']
            y = [cp_mean_log, cp_std_log]
            y_err = [cp_mean_err_log,
                     cp_std_err_log]
            y_labels = [r"$-\log_2(\langle P_c \rangle)$",
                        r"$-\log_2(\Delta P_c)$"]
            sim_labels = [r"$-\log_2(\langle P_c \rangle)$ from Simulations",
                          r"$-\log_2(\Delta P_c)$ from Simulations"]

            # Add values for cp_mean and cp_std at a saturated depth
            cp_mean_log_vals.append(cp_mean_log[d_index])
            cp_mean_err_log_vals.append(cp_mean_err_log[d_index])
            cp_std_log_vals.append(cp_std_log[d_index])
            cp_std_err_log_vals.append(cp_std_err_log[d_index])
            n_vals.append(n)

            if folder != "CG":
                # Create some more variables
                x = data['d']
                w = int(folder[0])
                cutoff = int(np.rint(np.power(n, 1 / w)))
                shape = tuple(cutoff for i in range(w))
                prediction = n * (1 - 1 * np.power(1 / x, w))

                # Create some helper strings for 2D and 3D Lattice
                if w > 1:
                    shape_str = r"Shape = {}, ".format(shape)
                    exp_str_pred = r"^{}".format(w)
                    cutoff_label = r"$n^{(1/" + str(w) + r")}$"
                else:
                    shape_str = r""
                    exp_str_pred = r""
                    cutoff_label = r"$n$"

                # Labels for 1D, 2D, or 3D Lattice
                x_label = r"$d$ = Depth of Circuit"
                title = (r"Collision Probability in a " +
                         "{} Lattice\n{}".format(folder, shape_str) +
                         "Number of Qubits = {}".format(n))
                pred_label = (r"Prediction: $-\log_2(\langle P_c \rangle) " +
                              "= n(1 - 1 / d{})$".format(exp_str_pred))

            else:
                # Create some more variables
                x = data['N']
                cutoff = n * np.log(n)
                prediction = n * (1 - 1 / np.exp(x / n))

                # Labels for a Complete Graph
                x_label = r"$N$ = Number of Gates Applied"
                title = (r"Collision Probability in a Complete Graph" + "\n" +
                         "Number of Qubits = {}".format(n))
                pred_label = (r"Prediction: $-\log_2( \langle P_c \rangle)" +
                              " = n(1 - 1 / e^(N/n))$")
                cutoff_label = r"$n ln(n)$"

            # Plotting cp_mean and cp_std as a function of d or N
            f_1, ax_1 = plt.subplots(figsize=(8, 4))
            for i in range(2):
                # Add labels to the figures
                ax_1.set_xlabel(x_label)
                ax_1.set_ylabel(y_labels[i])
                ax_1.set_title(title)

                # Add plots to the figure
                ax_1.errorbar(x, y[i], yerr=y_err[i],
                              fmt='--o', color='b',
                              label=sim_labels[i],
                              zorder=1, ms=3)

                # Add a prediction and cutoff for the mean
                if i < 1:
                    ax_1.plot(x, prediction, 'r-', label=pred_label)
                    ax_1.axvline(cutoff, color='g', linestyle='--',
                                 label=cutoff_label)

                # Add the legend and save the plot
                ax_1.legend(loc=4)
                f_1.savefig(
                    'plots/plots_{}/{}/{}.pdf'.format(folder, n,  types[i]),
                    dpi=200)
                ax_1.cla()

        # Create a plot that overplots cp_mean and cp_std as a function of n
        # at a fixed saturated depth with error bars
        f_1, ax_1 = plt.subplots(figsize=(8, 4))
        # Add labels to the figure, note that we re-use some of the labels
        # that we created above
        ax_1.set_xlabel(r"$n$ = Number of Qubits")
        ax_1.set_title("Steady State " + title.split("\n")[0])
        ax_1.errorbar(n_vals, cp_mean_log_vals,
                      yerr=cp_mean_err_log_vals,
                      fmt='x', color='b',
                      label=sim_labels[0],
                      ms=5, capsize=5)
        ax_1.errorbar(n_vals, cp_std_log_vals,
                      yerr=cp_std_err_log_vals,
                      fmt='x', color='r',
                      label=sim_labels[1],
                      ms=5, capsize=5)
        # Also add the line f(n) = n to the plot
        ax_1.plot(n_vals, n_vals, color='g', zorder=1,
                  label=r"$f(n)=n$")

        # Add the legend and save the plot
        ax_1.legend(loc=4)
        f_1.savefig(
            'plots/plots_{}/all_n/steady_state_cp.pdf'.format(
                folder))
        ax_1.cla()


def plot_x_star(all_data, a, s):
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
        # Each folder corresponds to a different type of circuit
        # and we will extract data for one plot from each folder
        data_list = all_data[folder]

        # The data for the plot will be stored in these lists
        n_vals = []
        x_star_vals = []
        x_star_err_vals = []

        # Extract a single coordinate (n, x_star) from the data_list
        for data in data_list:
            # get the cp_mean and its error from a specific k_matrix
            cp_mean_log, cp_mean_err_log = (collision_probability_mean_and_std(
                data['k_matrix'], s)[:2])

            n = data['n']
            if folder != 'CG':
                # For a lattice x is d, c = n - a
                x = data['d']
                c = n - a
            else:
                # For a complete graph x is N, c = n - n/a
                x = data['N']
                c = n - (n / a)

            # Interpolate the value of x_star from the data, and add it
            # with its error to the lists
            x_star, x_star_err = get_x_star(x, cp_mean_log, cp_mean_err_log, c)
            n_vals.append(n)
            x_star_vals.append(x_star)
            x_star_err_vals.append(x_star_err)

        # Create some more variables, here we are re-using names from
        # above to make things easier
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
            x_label = r"$\ln(n), n$ = Number of Qubits"
            y_label = (r"$\ln(d^*), d^*$ = depth such that" + "\n" +
                       r"$-\log_2(\langle P_c \rangle) = n - {}$".format(a))
            title = r"$d^*$ in a {} Lattice".format(folder)
            sim_label = r"$\ln(d^*)$ from Simulations"
            w = int(folder[0])

            # Different prediction labels for 1D lattice
            if w > 1:
                pred_label = (r"Prediction: $\ln(d^*) = " +
                              r"(1/{})(\ln(n) - \ln({}))$".format(w, a))
            else:
                pred_label = (r"Prediction: $\ln(d^*) = " +
                              r"\ln(n) - \ln({})$".format(a))

            # Create prediction
            prediction = np.power(n / a, 1 / w)
            log_prediction = np.log(prediction)

        else:
            # Labels for the Complete Graph
            x_label = r"$\ln(n), n$ = Number of Qubits"
            y_label = (r"$\ln(N^*), N*$ = number of gates such that" + "\n" +
                       r"$-\log_2(\langle P_c \rangle) = n - n/{}$".format(a))
            title = r"$N^*$ in a Complete Graph"
            sim_label = r"$\ln(N^*)$ from Simulations"
            pred_label = (r"Prediction: $\ln(N^*) = \ln(n) + " +
                          r"\ln(\ln({}))$".format(a))

            # Create Prediction
            prediction = n * np.log(a)
            log_prediction = np.log(prediction)

        # Create a figure to plot x_star as a function of n
        f_1, ax_1 = plt.subplots(figsize=(8, 4))

        # Add labels to the figures
        ax_1.set_xlabel(x_label)
        ax_1.set_ylabel(y_label)
        ax_1.set_title(title)

        # Add plots to the figure
        ax_1.errorbar(log_n, log_x_star, yerr=log_x_star_err,
                      fmt='--o', color='b', label=sim_label,
                      zorder=1, ms=3)
        ax_1.plot(log_n, log_prediction, 'r-', label=pred_label)
        ax_1.plot(log_n, lin_reg, 'g-',
                  label=(r"Linear Regression: $(m, b, r) = " +
                         r"({:1.2f}, {:1.2f}, {:1.2f})$".format(
                             slope, intercept, r_value)))

        # Add the legend and save the figure
        ax_1.legend(loc=4)
        f_1.savefig(
            'plots/plots_{}/all_n/x_star_{}.pdf'.format(folder, a), dpi=200)
        ax_1.cla()


def get_x_star(x, f, f_err, c):
    """
    Finds x_star such that f = c, where f is a function of x
    Uses a closed form solution in order to calculate the propogation of error
    Performs a Linear interpolation to find x_star
    """

    # First get the smallest index i where k[i] > c
    i = np.where(f > c)[0][0]

    # Get the two points that you will use for the linear interpolation
    x_1, y_1, y_1_err = (x[i - 1], f[i - 1], f_err[i - 1])
    x_2, y_2, y_2_err = (x[i], f[i], f_err[i])

    # Find closed form for d_star and d_star_err
    x_star = (((x_2 - x_1) * (c - y_1)) / (y_2 - y_1)) + x_1
    partial_y_1 = ((x_2 - x_1) * (c - y_2)) / np.power((y_2 - y_1), 2)
    partial_y_2 = ((x_2 - x_1) * (y_1 - c)) / np.power((y_2 - y_1), 2)
    x_star_err = np.sqrt(np.power(partial_y_1 * y_1_err, 2) +
                         np.power(partial_y_2 * y_2_err, 2))

    return (x_star, x_star_err)
