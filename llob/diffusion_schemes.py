import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


def theta_scheme_iteration(values, dx, dt, D, L, theta=1):
    """
    :param values: Values of function to iterate
    :type values: array
    :param dx: Size of price subinterval
    :type float: float
    :param dt: Size of price time
    :type float: float
    :param D: Diffusion constant
    :type float: float
    :param L: Latent liquidity : imposed slope at boundaries
    :type float: float
    :param theta: Weight of the implicit scheme in the theta-scheme solver
        theta belongs to [0,1] and is 1 by default (fully implicit scheme)
    :type theta: float, optional
    :returns: Updated density array
    :rtype: array
    """

    # Finite difference adimensional constant
    alpha = D * dt/(dx*dx)

    Nx = len(values)

    # Compute the scheme iteration matrix with a sparse representation
    main_diagonal = np.full(Nx, -2*alpha)
    secondary_diagonal = np.full(Nx-1, alpha)

    # Add Von Neumann boundary conditions
    main_diagonal[0] = -alpha
    main_diagonal[Nx-1] = -alpha
    boundary_terms = np.zeros(Nx)
    boundary_terms[0] = alpha * L * dx
    boundary_terms[Nx-1] = - alpha * L * dx

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
