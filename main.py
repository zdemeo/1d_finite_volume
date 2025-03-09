#  1D Implicit Finite Volume Heat Transfer Solver
#  Fully implicit. Unconditionally stable, so choose an appropriate value for dt.
#  Source terms use a linear approximation.
#  Reference: An Introduction to Computational Fluid Dynamics. Versteeg et al.

import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt


def setup_matrix(k, pc, dt, dx, x_len, Ta, Tb):
    """
Setup matrix diagonals representing the thermal system.
    :param k: Thermal conductivity (W/m*K)
    :param pc: Volumetric specific heat (J/m^3*K)
    :param dt: Timestep (s)
    :param dx: Grid step size in x (m)
    :param x_len: Total number of nodes
    :param Ta: Leftmost boundary condition (deg. C)
    :param Tb: Rightmost boundary condition (deg. C)
    :return: Matrix diagonals and source term.
    """
    # Construct tridiagonal matrix
    alpha = np.zeros(x_len)  # Upper diagonal
    D = np.zeros(x_len)  # Main diagonal
    beta = np.zeros(x_len)  # Lower diagonal
    Su = np.zeros(x_len)  # Linearized source term

    # Iterate through domain and setup nodal coefficients and source terms
    for x in range(x_len):
        # Heat transfer coefficient for node to the West of the ith node
        aw = 0
        if x > 0:
            aw = k / dx

        # Heat transfer coefficient for node to the East of the ith node
        ae = 0
        if x < x_len - 1:
            ae = k / dx

        # Transient term
        ap0 = pc * (dx / dt)

        # Linear approximation of source term
        Sp = 0
        if x == 0:  # Check if we're at a boundary
            if Ta is not None:  # Check if adiabatic
                Sp = -2 * k / dx
                Su[x] = Ta * 2 * k / dx
        elif x == x_len - 1:
            if Tb is not None:
                Sp = -2 * k / dx
                Su[x] = Tb * 2 * k / dx

        # Heat transfer coefficient for the ith node
        ap = aw + ae + ap0 - Sp

        # Diagonals for tridiagonal system
        alpha[x] = -ae
        D[x] = ap
        beta[x] = -aw

    # Add leading and trailing zeros to upper and lower diagonals
    # alpha = np.insert(alpha, 0, 0.0)[:-1]
    # beta = np.append(beta, 0.0)[1:]
    alpha = alpha[:-1]
    beta = beta[1:]

    return alpha, D, beta, Su


def solve_tridiagonal_system(upper, main_diag, lower, b):
    """
Solves a system of Ax=b, where A is a tridiagonal matrix.
    :param upper: Upper diagonal
    :param main_diag: Main diagonal
    :param lower: Lower diagonal
    :param b: Right-hand-side of the equation
    :return: x, where x is a vector
    """
    temp = linalg.lapack.dgtsv(lower, main_diag, upper, b)

    # Only want x
    return temp[3]


def main(dt):
    """
Main loop.
    :param dt: Timestep in seconds. NOTE: Implicit solver is unconditionally stable!
    :return temperature: List of temperatures for each dt.
    """

    # Print out every timestep
    verbose = False

    t_max = 100.0  # Total simulation time, seconds
    #dt = 2  # Timestep, seconds. NOTE: Implicit solver is unconditionally stable!

    k = 175  # Thermal conductivity, W/m*K
    p = 2770  # Density, kg/m^3
    c = 986  # Specific heat, J/kg*K

    x_len = 100  # Number of nodes
    dx = 0.001  # Distance between nodes, m

    initial_temperature = 300.0  # deg. C

    # Boundary conditions. NoneType means adiabatic boundary.
    Ta = 25.0  # Western boundary, deg. C
    Tb = 100.0  # Eastern boundary, deg. C

    n_iterations = int(t_max / dt) + 1

    pc = p * c  # Volumetric specific heat, J/m^3*K

    # Initialize grid
    temperature = np.zeros((n_iterations, x_len)) + initial_temperature
    temperature = temperature.tolist()

    # Generate diagonals for sparse matrix
    alpha, D, beta, Su = setup_matrix(k, pc, dt, dx, x_len, Ta, Tb)

    # Start solving for each timestep
    C = np.zeros(x_len)
    for t in range(1, n_iterations):
        for x in range(x_len):
            C[x] = pc * dx / dt * temperature[t - 1][x] + Su[x]

        temperature[t] = solve_tridiagonal_system(alpha, D, beta, C)

        if verbose:
            print('t=%d' % (t * dt,), temperature[t])

    return temperature


if __name__ == '__main__':
    t = main(2)

    plt.plot(t[20])
    plt.show()
