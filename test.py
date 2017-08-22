import symplectic
import decompose
import chp_py
import slow_sim
import utils
import imp
import numpy as np

imp.reload(symplectic)
imp.reload(decompose)
imp.reload(chp_py)
imp.reload(slow_sim)
imp.reload(utils)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This file contains test cases for all of the relevant functions in      #
# symplectic.py, decompose.py, chp_py.py, and utils.py                    #
# DO NOT MODIFY (unless you have a deep understanding of the code)        #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def test_single_symplectic(i, n):
    """
    Tests a single symplectic matrix
    """
    M = symplectic.symplectic(i, n)
    L = symplectic.get_lambda(n)
    result = (M.T @ L @ M) % 2
    result = ((result == L).all())
    return result


def test_symplectic():
    """
    Test cases for the functions in symplectic.py
    Returns True if passed all test cases, otherwise returns False
    """
    print("Testing: symplectic.py")
    for n in range(1, 6):
        for j in range(500):
            i = np.random.randint(symplectic.numberofsymplectic(n))
            result = test_single_symplectic(i, n)

            if not result:
                print("Failed: found a bad symplectic with (n,i) = "
                      + str((n, i)))
                return False

    print("Passed: symplectic.py passed all tests\n")
    return True


def test_decompose():
    """
    Test cases for the functions in decompose.py
    Returns True if passed all test cases, otherwise returns False
    """
    print("Testing: decompose.py")

    for n in range(1, 6):
        top = np.hstack((np.zeros((n, n)), np.identity(n)))
        bottom = np.hstack((np.identity(n), np.zeros((n, n))))
        L = np.vstack((top, bottom))

        for j in range(500):
            # Get a random symplectic
            i = np.random.randint(symplectic.numberofsymplectic(n))
            S = symplectic.symplectic(i, n)
            S = decompose.transform_symplectic(S)

            # Make sure the transformation worked
            result = (S.T @ L @ S) % 2
            result = ((result == L).all())
            if not result:
                print("Failed: could not transform symplectic " +
                      "matrix with (n, i)= "
                      + str((n, i)))
                return False

            # Make sure we can decompose it
            # Applying the decomposed gates to the identity should give us S
            gates = decompose.decompose_state(chp_py.CHP_Simulation(n, S))
            new_sim = chp_py.CHP_Simulation(n)
            new_sim.apply_gates(gates)
            result = (new_sim.state == S).all()

            if not result:
                print("Failed: found a bad decomposition for " +
                      "symplectic matrix with (n, i)= "
                      + str((n, i)))
                return False

    print("Passed: decompose.py passed all tests\n")
    return True


def test_chp_py():
    """
    Test cases for the functions in chp_py.py
    Returns True if passed all test cases, otherwise returns False
    """
    print("Testing: chp_py.py")
    # n = number of qubits, m = size of qubit gate
    for n in range(1, 50):
        # Create two simulations
        # sim1 uses decomposed gates and sim2 uses matrix multiplication
        # apply 100 random gates to each one
        sim1 = chp_py.CHP_Simulation(n)
        sim2 = chp_py.CHP_Simulation(n)

        for m in range(1, min(n, 5)):
            for j in range(100):
                # Get a random symplectic of size 2m x 2m
                i = np.random.randint(symplectic.numberofsymplectic(m))
                S = symplectic.symplectic(i, m)
                S = decompose.transform_symplectic(S)

                # Get m random qubits
                qubits = np.arange(m)
                np.random.shuffle(qubits)
                qubits = qubits[:m]

                # Get the gates and matrix represention of S
                gates = decompose.decompose_state(chp_py.CHP_Simulation(m, S))
                gates = decompose.change_gates(gates, qubits)
                M = decompose.symplectic_to_matrix(S, n, qubits)

                sim1.apply_gates(gates)
                sim2.state = (sim2.state @ M) % 2
                result = (sim1.state == sim2.state).all()

                if not result:
                    print("Failed: found two simulations with different " +
                          "states for n = " + str(n))
                    return False

    print("Passed: chp_py.py passed all tests\n")
    return True


def test_collision_probability():
    """
    Test cases for the collision probability algorithm
    Returns True if passed all test cases, otherwise returns False
    """
    print("Testing: collision probability algorithm")
    # n = number of qubits, m = size of qubit gate
    for n in range(1, 10):
        # sim1 is fast and sim2 is slow, as sim2 uses exponential
        # storage and runtime
        sim1 = chp_py.CHP_Simulation(n)
        sim2 = slow_sim.Slow_Simulation(n)

        for m in range(1, min(n, 5)):
            for j in range(100):
                # Get a random symplectic of size 2m x 2m
                i = np.random.randint(symplectic.numberofsymplectic(m))
                S = symplectic.symplectic(i, m)
                S = decompose.transform_symplectic(S)

                # Get m random qubits
                qubits = np.arange(m)
                np.random.shuffle(qubits)
                qubits = qubits[:m]

                # Get the gates and matrix represention of S
                gates = decompose.decompose_state(chp_py.CHP_Simulation(m, S))
                gates = decompose.change_gates(gates, qubits)

                decompose.apply_gates(gates, sim1)
                decompose.apply_gates(gates, sim2)

                k_1 = int(-sim1.log_collision_probability)
                k_2 = np.round(-np.log2(sim2.collision_probability))
                result = (k_1 == k_2)

                if not result:
                    print("Failed: collision probability algorithm returned " +
                          "incorrect result with n = "
                          + str(n) + " and m = " + str(m))
                    return False

    print("Passed: collision probability algorithm passed all tests\n")
    return True


def gates_to_coords(gates):
    """
    This is a helper function for test_utils(). Given a set of gates, it
    extracts the coordinates of the qubits that the gates were applied to
    """
    # Get gates and extract the coordinates, this is a bit tricky
    coords = []
    [coords.append(g[1]) for g in gates if g[1] not in coords]
    coords = [sorted((coords[2 * i], coords[2 * i + 1])) for i
              in range(len(coords) // 2)]
    coords = sorted((i, j) for (i, j) in coords)
    return coords


def test_utils():
    """
    Test cases for the functions in utils.py
    Returns True if passed all test cases, otherwise returns False
    """
    print("Testing: utils.py")

    # Test the indexing functions with a random size 4-D array
    shape = tuple(np.random.randint(1, 11, size=4))
    grid = np.arange(np.prod(shape)).reshape(shape)

    for i in range(np.prod(shape)):
        coord = utils.index_to_coord(i, shape)
        result = (i == grid[coord])

        if not result:
            print("Failed: incorrect coordinate for index i = "
                  + str(i) + " with shape = " + str(shape))
            return False

        index = utils.coord_to_index(coord, shape)
        result = (i == index)

        if not result:
            print("Failed: incorrect index for coord = "
                  + str(coord) + " with shape = " + str(shape))
            return False

    # Test the neighbor function on a few points on an easy
    # 3D grid of shape 3 x 3 x 3
    shape = (3, 3, 3)
    test_indices = [0, 4, 13, 16, 24, 26]
    diag_answers = {0: [1, 3, 4, 9, 10, 12, 13],
                    4: [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12,
                        13, 14, 15, 16, 17],
                    13: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                         14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                         25, 26],
                    16: [3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 17, 21, 22, 23,
                         24, 25, 26],
                    24: [12, 13, 15, 16, 21, 22, 25],
                    26: [13, 14, 16, 17, 22, 23, 25]}
    no_diag_answers = {0: [1, 3, 9],
                       4: [1, 3, 5, 7, 13],
                       13: [4, 10, 12, 14, 16, 22],
                       16: [7, 13, 15, 17, 25],
                       24: [15, 21, 25],
                       26: [17, 23, 25]}

    for i in test_indices:
        neighbors = utils.get_neighbors_grid(i, shape, True)
        result = (sorted(neighbors) == diag_answers[i])

        if not result:
            print("Failed: incorrect neighbors (including diagonals) for i = "
                  + str(i) + " with shape = " + str(shape))
            return False

        neighbors = utils.get_neighbors_grid(i, shape, False)
        result = (sorted(neighbors) == no_diag_answers[i])

        if not result:
            print("Failed: incorrect neighbors (not including diagonals) for "
                  + "i = " + str(i) + " with shape = " + str(shape))
            return False

    # Test the get_lattice_gates function on a 1D, 2D, and 3D grid

    # 1D grid of 10 qubits
    shape = (10,)
    answers = {0: [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)],
               1: [(1, 2), (3, 4), (5, 6), (7, 8)]}

    for i in range(2):
        gates = utils.get_lattice_gates(shape, i)
        coords = gates_to_coords(gates)
        result = (coords == answers[i])

        if not result:
            print("Failed: found the wrong set of gates in round i = "
                  + str(i) + " with shape = " + str(shape))
            return False

    # 2D grid of shape 5 x 5
    shape = (5, 5)
    answers = {0: [(0, 5), (1, 6), (2, 7), (3, 8),
                   (4, 9), (10, 15), (11, 16),
                   (12, 17), (13, 18), (14, 19)],
               1: [(0, 1), (2, 3), (5, 6), (7, 8),
                   (10, 11), (12, 13), (15, 16),
                   (17, 18), (20, 21), (22, 23)],
               2: [(5, 10), (6, 11), (7, 12), (8, 13),
                   (9, 14), (15, 20), (16, 21),
                   (17, 22), (18, 23), (19, 24)],
               3: [(1, 2), (3, 4), (6, 7), (8, 9),
                   (11, 12), (13, 14), (16, 17),
                   (18, 19), (21, 22), (23, 24)]}

    for i in range(4):
        gates = utils.get_lattice_gates(shape, i)
        coords = gates_to_coords(gates)
        result = (coords == answers[i])

        if not result:
            print("Failed: found the wrong set of gates in round i = "
                  + str(i) + " with shape = " + str(shape))
            return False

    # 3D grid of shape 3 x 4 x 5
    shape = (2, 3, 4)
    answers = {0: [(0, 12), (1, 13), (2, 14), (3, 15),
                   (4, 16), (5, 17), (6, 18), (7, 19),
                   (8, 20), (9, 21), (10, 22), (11, 23)],
               1: [(0, 4), (1, 5), (2, 6), (3, 7),
                   (12, 16), (13, 17), (14, 18), (15, 19)],
               2: [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9),
                   (10, 11), (12, 13), (14, 15), (16, 17),
                   (18, 19), (20, 21), (22, 23)],
               3: [],
               4: [(4, 8), (5, 9), (6, 10), (7, 11), (16, 20),
                   (17, 21), (18, 22), (19, 23)],
               5: [(1, 2), (5, 6), (9, 10), (13, 14), (17, 18),
                   (21, 22)]}

    for i in range(5):
        gates = utils.get_lattice_gates(shape, i)
        coords = gates_to_coords(gates)
        result = (coords == answers[i])

        if not result:
            print("Failed: found the wrong set of gates in round i = "
                  + str(i) + " with shape = " + str(shape))
            return False

    # Test collision_probability_mean_and_std function for
    # small values of k
    for i in range(100):
        # Initialize some variables
        m, x = (25, 100)
        s = 5
        k_matrix = np.random.randint(20, size=(m, x))
        mean_ans = np.zeros(x)
        std_ans = np.zeros(x)
        mean_err_ans = np.zeros(x)
        std_err_ans = np.zeros(x)

        for j in range(x):
            # k_vec is a set of m samples
            k_vec = k_matrix[:, j]

            # Divide the m samples into m/s sets of size s
            k_sets = k_vec.reshape(m // s, s).astype(np.float64)
            # Add a small perturbation to avoid a set with
            # zero std to avoid infs and nans
            k_sets[:, -1] += 1e-3

            # Exponentiate the k_sets
            p_sets = np.power(1 / 2, k_sets)

            # mu[i] and sigma[i] are the mean and std of the
            # i^th set in p_sets
            mu = np.mean(p_sets, axis=1)
            sigma = np.std(p_sets, axis=1)

            # a is the mean of the values in mu
            # b is the std of the values in mu
            # c is the mean of the values in sigma
            # d is the std of the values in mu
            a = np.mean(mu)
            b = np.std(mu)
            c = np.mean(sigma)
            d = np.std(sigma)

            # Convert to the log scale and use propogation of
            # error formula for the errors of the logs
            mean_ans[j] = -np.log2(a)
            std_ans[j] = -np.log2(c)
            mean_err_ans[j] = b / (a * np.log(2))
            std_err_ans[j] = d / (c * np.log(2))

        mean, mean_err, std, std_err = (
            utils.collision_probability_mean_and_std(k_matrix, s))
        result = ((abs(mean - mean_ans) < 1e-6).all() and
                  (abs(mean_err - mean_err_ans < 1e-6)).all() and
                  (abs(std - std_ans) < 1e-6).all() and
                  (abs(std_err - std_err_ans < 1e-6)).all())
        if not result:
            print("Failed: found incorrect values for mean and standard " +
                  "deviation of the collision probability")
            return False

    print("Passed: utils.py passed all tests\n")
    return True


def test_stored_two_qubit_gates():
    """
    Test cases for the stored two_qubit_gates
    Returns True if passed all test cases, otherwise returns False
    """
    print("Testing: stored two qubit gates")
    utils.store_all_two_qubit_gates()
    two_qubit_gates = utils.load_data("two_qubit_gates")
    two_qubit_matrices = utils.load_data("two_qubit_matrices")

    matrices = set()
    for i in two_qubit_gates:
        # Make sure the gates and matrices have the same effect
        gates = two_qubit_gates[i]
        M = two_qubit_matrices[i]

        sim1 = chp_py.CHP_Simulation(2)
        sim1.apply_gates(gates)

        # Also store the matrices in a set
        M_string = str(M.flatten())
        matrices.add(M_string)

        result = (sim1.state == M).all()
        if not result:
            print("Failed: two qubit gates do not agree with "
                  + "two qubit matrix for i = " + str(i))
            return False

    # Make sure we have the right number of unique matrices
    result = (len(matrices) == symplectic.numberofsymplectic(2))
    if not result:
        print("Failed: did not find the correct number of unique "
              + "two qubit matrices")
        return False

    print("Passed: stored two qubit gates passed all tests\n")
    return True


def run_all_tests():
    print("")
    if all((test_symplectic(), test_decompose(), test_chp_py(),
            test_collision_probability(), test_utils(),
            test_stored_two_qubit_gates())):
        print("Passed: all tests")
    else:
        print("Failed: certain tests, see output above")


run_all_tests()
