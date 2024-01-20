import numpy as np
# PDE solver only for Neumann boundary condition.
# shenyao jin, 01/2024, shenyaojin@mines.edu
# conflict with pdesolver_parabolic.py
def TDMAsolver(a, b, c, d):
    '''
    TDMA solver, a b c d can be NumPy array type or Python list type.
    refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    input:
        a : left   (i-1) coefficients
        b : center ( i ) coefficients
        c : right  (i+1) coefficients
        d : right hand vector of, e.g., heat distribution values at time t
    output:
        xc: Solution of, e.g., heat distribution vector at time t+∆t

        Note: a,b,c all have the same size. The first element of a and last
        element of b are not used.
    '''
    nf = len(d)
    aa, ab, ac, ad = map(np.array, (a, b, c, d))

    ## . . Forward sweep
    for it in range(1, nf):
        mc = ac[it]/bc[it-1]
        bc[it] = bc[it] - mc*cc[it-1] # . . Overwriting b
        dc[it] = dc[it] - mc*dc[it-1] # . . Overwriting d

    ## . . Backsubstitution sweep
    xc = bc
    xc[-1] = dc[-1]/bc[-1]

    for il in range(nf-2, -1, -1):
        xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]

    return xc

def solve2ddiffusion_neumann_beforeshutin(U, k1, dx, dy, dt, k2, pres):
    nx, ny = np.size(U, 0), np.size(U, 1)
    # iteration:
    U1 = np.zeros((nx, ny))

    # parameters in y direction
    a = (k1 * dt) / (2 * dx ** 2)
    b = (k1 * dt) / (2 * dy ** 2)
    c = (k2 * dt) / 2


    # set up coefficients in X
    ax = -a * np.ones(nx)
    bx = (1 + 2 * a) * np.ones(nx)
    cx = -a * np.ones(nx)

    ax[-1] = 1
    cx[0]  = -1
    bx[0]  = 1
    bx[-1] = -1

    # set up coefficients in y
    ay = -b * np.ones(ny)
    by = (1 + 2 * b) * np.ones(ny)
    cy = -b * np.ones(ny)

    ay[-1] = 1
    cy[0]  = -1
    by[0] = 1
    by[-1] = -1

    # set up solution matrix
    dx = np.zeros(nx)
    dy = np.zeros(ny)

    # solve first for update in x direction
    for iy in range(ny):
