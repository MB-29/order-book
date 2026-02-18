"""
PDE integration schemes for diffusion equations.
"""

import numpy as np
import numpy.typing as npt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


def theta_scheme_iteration(
    values: npt.NDArray[np.float64],
    dx: float,
    dt: float,
    D: float,
    L: float,
    theta: float = 1.0,
) -> npt.NDArray[np.float64]:
    """
    Perform one iteration of the theta-scheme for the diffusion equation.

    Solves the 1D diffusion equation using a theta-weighted implicit/explicit
    finite difference scheme with Von Neumann boundary conditions.

    Args:
        values: Current values of the function to iterate.
        dx: Spatial grid spacing.
        dt: Time step size.
        D: Diffusion constant.
        L: Latent liquidity (imposed slope at boundaries).
        theta: Weight of implicit scheme (0=explicit, 1=fully implicit).

    Returns:
        Updated density array after one time step.
    """
    # Finite difference adimensional constant
    alpha = D * dt / (dx * dx)
    Nx = len(values)

    # Compute the scheme iteration matrix with a sparse representation
    main_diagonal = np.full(Nx, -2 * alpha)
    secondary_diagonal = np.full(Nx - 1, alpha)

    # Add Von Neumann boundary conditions
    main_diagonal[0] = -alpha
    main_diagonal[Nx - 1] = -alpha
    boundary_terms = np.zeros(Nx)
    boundary_terms[0] = alpha * L * dx
    boundary_terms[Nx - 1] = -alpha * L * dx

    # Build matrices and solve
    implicit_diagonals = [
        1 - theta * main_diagonal,
        -theta * secondary_diagonal,
        -theta * secondary_diagonal,
    ]
    explicit_diagonals = [
        1 + (1 - theta) * main_diagonal,
        (1 - theta) * secondary_diagonal,
        (1 - theta) * secondary_diagonal,
    ]

    implicit_matrix = diags(implicit_diagonals, [0, 1, -1], format="csr")
    explicit_matrix = diags(explicit_diagonals, [0, 1, -1]).toarray()

    return spsolve(implicit_matrix, explicit_matrix.dot(values) + boundary_terms)
