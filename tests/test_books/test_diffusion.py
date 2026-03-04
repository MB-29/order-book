"""
Tests for theta_scheme_iteration diffusion solver.
"""

import numpy as np

from llob.books.diffusion_schemes import theta_scheme_iteration


class TestThetaSchemeBasics:
    """Tests for basic theta_scheme_iteration functionality."""

    def test_output_shape(self):
        """Output should have same shape as input."""
        Nx = 50
        values = np.linspace(-10, 10, Nx)
        dx = 20.0 / (Nx - 1)
        dt = 0.01
        D = 0.5
        L = 10.0

        result = theta_scheme_iteration(values, dx, dt, D, L)

        assert result.shape == values.shape
        assert len(result) == Nx

    def test_fully_implicit_scheme(self):
        """theta=1 should use fully implicit scheme."""
        Nx = 50
        values = np.linspace(-10, 10, Nx)
        dx = 20.0 / (Nx - 1)
        dt = 0.1  # Larger timestep okay for implicit
        D = 0.5
        L = 10.0

        # Should not blow up even with larger timestep
        result = theta_scheme_iteration(values, dx, dt, D, L, theta=1.0)

        assert np.all(np.isfinite(result))

    def test_explicit_scheme(self):
        """theta=0 should use explicit scheme."""
        Nx = 50
        values = np.linspace(-10, 10, Nx)
        dx = 20.0 / (Nx - 1)
        # Small timestep for stability: dt < dx^2 / (2*D)
        dt = dx**2 / (4 * 0.5)  # Safety factor of 2
        D = 0.5
        L = 10.0

        result = theta_scheme_iteration(values, dx, dt, D, L, theta=0.0)

        assert np.all(np.isfinite(result))

    def test_crank_nicolson_scheme(self):
        """theta=0.5 should use Crank-Nicolson scheme."""
        Nx = 50
        values = np.linspace(-10, 10, Nx)
        dx = 20.0 / (Nx - 1)
        dt = 0.01
        D = 0.5
        L = 10.0

        result = theta_scheme_iteration(values, dx, dt, D, L, theta=0.5)

        assert np.all(np.isfinite(result))


class TestThetaSchemeBoundaryConditions:
    """Tests for boundary condition handling."""

    def test_von_neumann_boundary_slope(self):
        """Boundary conditions should impose slope L."""
        Nx = 100
        L = 10.0
        D = 0.5
        dx = 20.0 / (Nx - 1)
        dt = 0.001

        # Start with linear profile (steady state)
        X = np.linspace(-10, 10, Nx)
        values = -L * X

        # Run several iterations
        result = values.copy()
        for _ in range(10):
            result = theta_scheme_iteration(result, dx, dt, D, L)

        # For linear profile with matching boundary conditions,
        # solution should remain approximately linear
        # Check slope at boundaries
        left_slope = (result[1] - result[0]) / dx
        right_slope = (result[-1] - result[-2]) / dx

        # Slopes should be close to -L (since density = -L*x)
        assert np.isclose(left_slope, -L, rtol=0.1)
        assert np.isclose(right_slope, -L, rtol=0.1)

    def test_steady_state_linear_profile(self):
        """Linear profile with matching L should be steady state."""
        Nx = 100
        L = 10.0
        D = 0.5
        dx = 20.0 / (Nx - 1)
        dt = 0.001

        # Linear profile is steady state for diffusion with linear BC
        X = np.linspace(-10, 10, Nx)
        initial = -L * X

        result = initial.copy()
        for _ in range(50):
            result = theta_scheme_iteration(result, dx, dt, D, L)

        # Should remain close to initial linear profile
        np.testing.assert_array_almost_equal(result, initial, decimal=1)


class TestThetaSchemeDiffusionBehavior:
    """Tests for diffusion behavior."""

    def test_diffusion_smooths_perturbation(self):
        """Diffusion should smooth out perturbations."""
        Nx = 100
        L = 10.0
        D = 0.5
        dx = 20.0 / (Nx - 1)
        dt = 0.001

        # Start with linear profile plus perturbation
        X = np.linspace(-10, 10, Nx)
        linear = -L * X
        perturbation = 5.0 * np.exp(-(X**2))  # Gaussian bump
        values = linear + perturbation

        initial_perturbation_energy = np.sum((values - linear) ** 2)

        # Run diffusion
        result = values.copy()
        for _ in range(100):
            result = theta_scheme_iteration(result, dx, dt, D, L)

        final_perturbation_energy = np.sum((result - linear) ** 2)

        # Perturbation energy should decrease
        assert final_perturbation_energy < initial_perturbation_energy

    def test_higher_D_faster_diffusion(self):
        """Higher diffusion constant should cause faster smoothing."""
        Nx = 100
        L = 10.0
        dx = 20.0 / (Nx - 1)
        dt = 0.0001  # Small for explicit stability

        X = np.linspace(-10, 10, Nx)
        linear = -L * X
        perturbation = 5.0 * np.exp(-(X**2))
        initial = linear + perturbation

        # Run with low D
        result_low_D = initial.copy()
        for _ in range(100):
            result_low_D = theta_scheme_iteration(result_low_D, dx, dt, D=0.1, L=L)

        # Run with high D
        result_high_D = initial.copy()
        for _ in range(100):
            result_high_D = theta_scheme_iteration(result_high_D, dx, dt, D=0.5, L=L)

        # Higher D should give result closer to linear (more smoothed)
        deviation_low_D = np.sum((result_low_D - linear) ** 2)
        deviation_high_D = np.sum((result_high_D - linear) ** 2)

        assert deviation_high_D < deviation_low_D


class TestThetaSchemeNumericalStability:
    """Tests for numerical stability."""

    def test_implicit_stable_with_large_timestep(self):
        """Fully implicit scheme should be stable with large timesteps."""
        Nx = 50
        D = 0.5
        L = 10.0
        dx = 20.0 / (Nx - 1)
        # Large timestep that would be unstable for explicit
        dt = 10 * dx**2 / (2 * D)

        X = np.linspace(-10, 10, Nx)
        values = -L * X

        # Should not blow up
        result = theta_scheme_iteration(values, dx, dt, D, L, theta=1.0)

        assert np.all(np.isfinite(result))
        assert np.max(np.abs(result)) < 1e6  # Reasonable bound

    def test_preserves_approximate_mass(self):
        """Diffusion should approximately preserve integral (mass)."""
        Nx = 100
        L = 10.0
        D = 0.5
        dx = 20.0 / (Nx - 1)
        dt = 0.001

        X = np.linspace(-10, 10, Nx)
        values = -L * X + 5.0 * np.exp(-(X**2))

        initial_mass = np.sum(values) * dx

        result = values.copy()
        for _ in range(50):
            result = theta_scheme_iteration(result, dx, dt, D, L)

        final_mass = np.sum(result) * dx

        # Mass should be approximately conserved
        # (some boundary flux expected due to Von Neumann BC)
        assert abs(final_mass - initial_mass) < abs(initial_mass) * 0.5
