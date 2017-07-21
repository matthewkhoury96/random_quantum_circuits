import numpy as np
import symplectic
import decompose
import imp

imp.reload(symplectic)
imp.reload(decompose)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This file contains a class to perform a naive simulation of a quantum   #
# computer that takes exponential runtime and storage                     #
# The main purpose of this file is to test the collision_probability      #
# algorithm used in the CHP_Simulation() class in chp_py                  #
# DO NOT MODIFY (unless you have a deep understanding of the code)        #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class Slow_Simulation(object):
    """
    A simulation of a quantum computer that uses exponential storage
    and exponential runtime to apply a single matrix
    Limited specifically to Clifford Gates
    """

    # Some static variables will be helpful

    def __init__(self, n):
        """
        Initialize the simulation
        """

        # By default we start in the computational basis state
        self.n = n
        self.state = np.zeros(np.power(2, n))
        self.state[0] = 1
        self.I = np.identity(2)
        self.H = np.power(2, -0.5) * np.array([[1, 1], [1, -1]])
        self.P = np.array([[1, 0], [0, 0 + 1j]])

    @property
    def collision_probability(self):
        return np.sum(np.power(abs(self.state), 4))

    def apply_cnot(self, a, b):
        """
        Apply CNOT from control a to target b
        """
        nn = int(np.power(2, self.n))
        M = np.zeros((nn, nn))
        bit_flip = {'1': '0', '0': '1'}

        for i in range(nn):
            col = np.zeros(nn)
            bits = np.binary_repr(i, width=self.n)
            if bits[a] == '1':
                new_bits = bits[:b] + bit_flip[bits[b]] + bits[b + 1:]
                col[int(new_bits, 2)] = 1
            else:
                col[i] = 1

            M[:, i] = col

        self.state = M @ self.state

    def apply_hadamard(self, a):
        """
        Apply Hadamard on qubit a
        """
        M = 1
        for i in range(a):
            M = np.kron(M, self.I)

        M = np.kron(M, self.H)

        for i in range(a + 1, self.n):
            M = np.kron(M, self.I)

        self.state = M @ self.state

    def apply_phase(self, a):
        """
        Apply Phase on qubit a
        """
        M = 1
        for i in range(a):
            M = np.kron(M, self.I)

        M = np.kron(M, self.P)

        for i in range(a + 1, self.n):
            M = np.kron(M, self.I)

        self.state = M @ self.state
