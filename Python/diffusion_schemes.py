import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


def theta_scheme_iteration(values, dx, dt, D, L, theta=1):
    """

    Arguments:
        values {numpy array} -- Values of function to iterate
        dx {float} -- Space (price) subinterval
        dt {float} -- Time subinterval
        D {float} -- Diffusion constant
        L {float} -- Latent liquidity, imposes Von Neumann boundary conditions

    Keyword Arguments:
        theta {float} -- Weight of implicit scheme in the theta-scheme solver ;
        theta belongs to [0,1] and is 1 by default (fully implicit scheme)
    """

    # Finite difference adimensional constant
    alpha = D * dt/(dx*dx)
    N = len(values)
    # Compute the scheme iteration matrix with a sparse representation
    # (see documentation)
    main_diagonal = np.full(N, -2*alpha)
    secondary_diagonal = np.full(N-1, alpha)

    # Add Von Neumann boundary conditions
    main_diagonal[0] = -alpha
    main_diagonal[N-1] = -alpha
    boundary_terms = np.zeros(N)
    boundary_terms[0] = alpha * L * dx
    boundary_terms[N-1] = - alpha * L * dx

    # Build matrices and solve
    implicit_diagonals = [1-theta * main_diagonal,
                          -theta*secondary_diagonal, -theta*secondary_diagonal]
    explicit_diagonals = [1+(1-theta) * main_diagonal,
                          (1-theta)*secondary_diagonal,
                          (1-theta)*secondary_diagonal]

    implicit_matrix = diags(implicit_diagonals, [0, 1, -1], format='csr')
    explicit_matrix = diags(explicit_diagonals, [0, 1, -1]).toarray()

    return spsolve(implicit_matrix,
                   explicit_matrix.dot(values)
                   + boundary_terms)
