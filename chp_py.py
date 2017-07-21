import numpy as np
import symplectic
import decompose
import imp

imp.reload(symplectic)
imp.reload(decompose)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This file contains a class CHP_Simulation() which is used to run        #
# simulations with Clifford Gates in polynomial time                      #
# It also includes certain built in functions to easily apply sets of     #
# gates as well as compute the Collision Probability                      #
# DO NOT MODIFY (unless you have a deep understanding of the code)        #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class CHP_Simulation(object):
    """ A CHP Simulation as described in
    https://arxiv.org/pdf/quant-ph/0406196.pdf
    Both inspired by and tested using Scott Aarson's version
    of this code written in C, can be found
    http://www.scottaaronson.com/chp/
    """

    def __init__(self, n, state=None):
        """
        Initialize the simulation
        """

        # By default we start in the computational basis state
        self.n = n
        self.state = np.identity(2 * n, dtype=np.int8)

        # Otherwise, use the state that is already given if it is the right
        # shape
        if state is not None:
            if state.shape != (2 * n, 2 * n):
                raise ValueError("state must be a matrix of shape "
                                 + str((2 * n, 2 * n)))
            self.state = np.copy(state).astype(np.int8)

    def __str__(self):
        return np.array2string(self.state)

    @property
    def stabilizers(self):
        """
        Prints a string such that the state matrix is represented by
        its stabilizer states instead of binary values
        """
        string_map_matrix = {(0, 0): "I", (1, 0): "X",
                             (1, 1): "Y", (0, 1): "Z"}
        result = ""
        for i in range(2 * self.n):
            if i == self.n:
                result += "-" * (self.n) + "\n"

            for j in range(self.n):
                (x_ij, z_ij) = (self.state[i, j], self.state[i, j + self.n])
                result += string_map_matrix[(x_ij, z_ij)]

            result += "\n"

        print(result[:-1])

    @property
    def log_collision_probability(self):
        """
        Returns the log2 of the collision probability of the vector
        that is stabilized by our stabilizer state self.state
        See notes for details/proof of why this is correct
        """
        # Get the n generators for the 2^n stabilizers
        # and put them into the colums of a matrix A and then
        # create the matrix B with the generators described in the notes
        A = self.state[self.n:, :].T
        B = np.vstack((np.zeros((self.n, self.n)), np.identity(self.n)))
        C = np.hstack((A, B))

        # k = number of pivots in REF(C) = rank(C)
        k = decompose.get_rank(C)
        # k = decompose.row_wise_gaussian_elimination_pivots_fast(C)

        # Collision probaiblity is 1 / (2^(k - n))
        # so log2(collision_probability) = n - k
        return(self.n - k)

    @property
    def collision_probability(self):
        """
        Returns the actual collision_probability
        NOTE: this number may round down to 0 for large n
        """
        return(np.power(2.0, self.log_collision_probability))

    def apply_cnot(self, a, b):
        """
        Apply CNOT from control a to target b
        """
        self.state[:, b] = (self.state[:, b] + self.state[:, a]) % 2
        self.state[:, self.n + a] = ((self.state[:, self.n + a]
                                      + self.state[:, self.n + b]) % 2)

    def apply_hadamard(self, a):
        """
        Apply Hadamard on qubit a
        """
        self.state[:, [a, self.n + a]] = self.state[:, [self.n + a, a]]

    def apply_phase(self, a):
        """
        Apply Phase on qubit a
        """
        self.state[:, self.n + a] = ((self.state[:, self.n + a]
                                      + self.state[:, a]) % 2)

    def apply_z(self, a):
        """
        Apply Z gate on qubit a
        """
        self.apply_phase(a)
        self.apply_phase(a)

    def apply_x(self, a):
        """
        Apply X gate on qubit a
        """
        self.apply_hadamard(a)
        self.apply_z(a)
        self.apply_hadamard(a)

    def apply_y(self, a):
        """
        Apply Y gate on qubit a
        """
        self.apply_phase(a)
        self.apply_z(a)
        self.apply_x(a)
        self.apply_phase(a)

    def apply_random_symplectic(self, qubits):
        """
        Generates a random symplectic gate and then applies it
        to the qubits in the list qubits
        """
        # Here m is the number of qubits that the gate will be applied to
        # while n is the total number of qubits in the simulation
        m = len(qubits)

        # Generate a random symplectic matrix that is
        # symplectic with L = direct_sum_{j=1}^n X
        i = np.random.randint(symplectic.numberofsymplectic(m))
        S = symplectic.symplectic(i, m)

        # Convert this symplectic matrix to one that is symplectic
        # with L = [[0, I], [I, 0]]
        S = decompose.transform_symplectic(S)

        # Lastly, apply this to our state
        self.apply_symplectic(S, qubits)

    def apply_symplectic(self, S, qubits):
        """
        Applies a symplectic matrix that is symplectic with
        L = [[0, I], [I, 0]] to the qubits in the list qubits
        """
        # Approach 1: convert the 2m x 2m symplectic matrix S to a 2n x 2n
        # matrix that acts on the corresponding columns in qubits
        # M = decompose.symplectic_to_matrix(S, self.n, qubits)
        # self.state = (self.state @ M) % 2

        # Approach 2: decompose the 2m x 2m symplectic matrix into a
        # series of {C, H, P} gates, then apply those
        # NOTE: this is actually much faster in practice for large n
        m = len(qubits)
        gates = decompose.decompose_state(CHP_Simulation(m, S))
        gates = decompose.change_gates(gates, qubits)
        decompose.apply_gates(gates, self)

    def apply_gates(self, gates, qubits=None):
        """
        Applies a set of gates to the given qubits
        """
        if qubits is not None:
            gates = decompose.change_gates(gates, qubits)
        decompose.apply_gates(gates, self)
