import numpy as np

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This file contains a set of functions used to generate a uniformly      #
# random symplectic matrix that is symplectic with respect with           #
# L = direct_sum_{j=1}^n X                                                #
# The only function you will likely need to use is symplectic(i, n)       #
# DO NOT MODIFY (unless you have a deep understanding of the code)        #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def test_gram_schmidt(symp_basis):
    """
    Tests to make sure we have a symplectic basis
    where symp_basis = (v_1, w_1, ..., v_n, w_n)
    """
    n = symp_basis.shape[0] // 2

    for j in range(n):
        for k in range(n):
            v_j = symp_basis[:, 2 * j]
            v_k = symp_basis[:, 2 * k]
            w_j = symp_basis[:, 2 * j + 1]
            w_k = symp_basis[:, 2 * k + 1]
            if (inner_prod(v_j, w_k) != (j == k)):
                return False
            if not(inner_prod(v_j, v_k) == inner_prod(w_j, w_k) == 0):
                return False
    return True


def get_lambda(n):
    """
    Creates a 2n x 2n lambda matrix L
    NOTE: This lambda matrix is NOT the one conventionally used for
          symplectic matrices
    """
    x = np.array([[0, 1], [1, 0]])
    L = np.array([[0, 1], [1, 0]])
    for i in range(n - 1):
        L = direct_sum(L, x) % 2
    return L


def direct_sum(a, b):
    """
    Returns direct sum of matrices a and b
    """
    m, n = a.shape
    p, q = b.shape
    top = np.hstack((a, np.zeros((m, q))))
    bottom = np.hstack((np.zeros((p, n)), b))
    return np.vstack((top, bottom))


def numberofcosets(n):
    """
    Returns the number of different cosets
    """
    x = np.power(2, 2 * n - 1) * (np.power(2, 2 * n) - 1)
    return x


def numberofsymplectic(n):
    """
    Returns the number of symplectic group elements
    """
    x = 1
    for j in range(1, n + 1):
        x = x * numberofcosets(j)
    return x


def inner_prod(v, w):
    """
    Returns Symplectic inner product over F_2 using lambda from get_lambda()
    """
    n = len(v) // 2
    return (v.T @ get_lambda(n) @ w) % 2


def get_binary_repr(i, w):
    """
    Returns the first w bits in the binary representaion of integer i
    NOTE: this function returns the lowest order bits as the first elements
          of the array and the higher order bits as the last elements
          so if i = 2 = 00...0010, then this program returns the array
          arr = [0, 1, 0, 0, ..., 0, 0]
    """
    string = np.binary_repr(i, width=w)[::-1]
    arr = np.array(list(string), dtype=int)[:w]
    return arr


def symplectic_gram_schmidt_step(omega):
    """
    Performs a single step of the Symplectic Gram Schmidt algorithm
    Returns: a tuple (w, omega) or (None, omega) where omega has been
             modified and will have one or two columns removed
    See paper for details on how this works
    """
    # By convention v is always the vector in the first column of omega
    v = omega[:, 0]

    # Look for a symplectic pair (v, w)
    for i in range(1, len(omega[0])):
        w = omega[:, i]
        if inner_prod(v, w) == 1:
            # Case 1: We found a symplectic pair (v, w) so we return
            # w, we then remove v and w from omega and modify its columns
            omega = np.delete(omega, [0, i], axis=1)
            for j in range(len(omega[0])):
                f = omega[:, j]
                omega[:, j] = (f + (inner_prod(v, f) * w) +
                               (inner_prod(w, f) * v)) % 2
            return (w, omega)

    # Case 2: there is no symplectic pair, so remove v
    # and return the rest of the vectors
    omega = np.delete(omega, 0, axis=1)
    return (None, omega)


def symplectic_gram_schmidt(omega):
    """
    Performs the Symplectic Gram Schmidt algorithm on the colum vectors
    in omega
    Returns: symplectic basis symp_basis = (v_1, w_1, ..., v_n, w_n)
             Where v_1 is the first column in omega by convention
    """

    # initialize some variables
    n = omega.shape[0] // 2
    symp_basis = np.zeros((2 * n, 2 * n))

    # Iteratively update the symp_basis by finding symplectic pairs
    # NOTE: we will need a total of m steps, n for the n symplectic
    # pairs (v, w) and one to remove the (m - 2*n) extra vectors
    j = 0
    while j < n:
        v = omega[:, 0]
        w, omega_prime = symplectic_gram_schmidt_step(omega)
        if w is not None:
            # Fill in the symp_basis when we find a symplectic pair
            # and ONLY increment j whenever we update symp_basis
            symp_basis[:, 2 * j] = v
            symp_basis[:, 2 * j + 1] = w
            j += 1
        omega = omega_prime

    return symp_basis


def symplectic(i, n):
    """
    Returns the ith symplectic matrix of size 2n x 2n
    See https://arxiv.org/pdf/1406.2170.pdf for details on how this works
    """

    # Step 1: initialize s and k, and set i <- (i/s)
    s = np.power(2, 2 * n) - 1
    k = (i % s) + 1
    i //= s

    # Step 2: find v_1
    v_1 = get_binary_repr(k, 2 * n)

    # Step 3: symplectic Gram-Schmidt, see paper for details on this
    omega = np.hstack((v_1[:, None], np.identity(2 * n)))
    symp_basis = symplectic_gram_schmidt(omega)

    # Step 4: create b and c
    bits = get_binary_repr(i, 2 * n - 1)
    b = bits[0:n]
    c = np.ones(n)
    c[1:] = bits[n:]
    # For ease, let d = [b_0, c_0, b_1, c_1, ..., b_n, c_n]
    d = np.ravel(np.column_stack((b, c)))

    # Step 5: create w_1_prime
    w_1_prime = (symp_basis @ d) % 2

    # Step 6: create g using w_1_prime
    g = symp_basis
    g[:, 1] = w_1_prime

    # Step 6.5: I had to add this to make it work, does NOT agree w/ paper ?!?!
    g = symplectic_gram_schmidt(g)

    # Step 7: recurse and modify g if necessary, then return
    if n != 1:
        next_g = symplectic(i >> (2 * n - 1), n - 1)
        g = (g @ direct_sum(np.identity(2), next_g)) % 2

    return g.astype(np.int8)
