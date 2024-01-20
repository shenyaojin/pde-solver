import numpy as np

# the original definition of PDE solver. Based on Jeff's code.

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
    nf = len(d) # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d)) # copy arrays

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

def Solve_2D_Convection_Diffusion_homogeneous_Thomas(U,K,vx,vy,dx,dy,dt):
    # this function is used to simulate after the shut in period.
    # with force term
    '''Set up tridiagonal matrix and call Thomas Algorithm
    usage: x = Solve_2D_Convection_Diffusion_homogeneous_Thomas(U,K,vx,vy,dx,dy,dt,LB,RB,BB,TB):
    input:
        U: heat solution at time step n   (nx)
        K : thermal diffusivity
        vx: convection velocity in x-direction
        vy: convection velocity in y-direction
        dx: spatial sampling in x-direction
        dy: spatial sampling in y-direction
        dt: temporal sampling
        LB: Left   boundary condition (Dirchelet)
        RB: Right  boundary condition (Dirchelet)
        BB: Bottom boundary condition (Dirchelet)
        TB: Top    boundary condition (Dirchelet)
    output:
        u: heat solution at time step n+1 (nx,ny)
    depends on:
        TDMAsolver

    written by Jeff Shragge, jshragge@mines.edu, 10/2019
    '''
    ## . . Get dimensions
    nx,ny = np.size(U,0),np.size(U,1)
    U1 = np.zeros((nx,ny))

    ## Enforce boundary condition
    # adding boundary condition to
    U[0,:] = U[1,:]
    U[-1,:] = U[-2,:]
    U[:,0] = U[:,1]
    U[:,-1] = U[:,-2]


    U1[0,:] = U1[1,:]
    U1[-1,:] = U1[-2,:]
    U1[:,0] = U1[:,1]
    U1[:,-1] = U1[:,-2]

    ## . . Define diffusivity and Courant numbers
    AX = K *dt/(dx*dx) ## Diffusivity x: alpha
    AY = K *dt/(dy*dy) ## Diffusivity y: alpha
    CX = vx*dt/ dx     ## Courant x : C
    CY = vy*dt/ dy     ## Courant y : C

    ## . . Set up coefficients in X
    ax =-(CX+2*AX)*np.ones(nx-2)
    bx =  4*(1+AX)*np.ones(nx-2)
    cx = (CX-2*AX)*np.ones(nx-2)

    ## . . Set up coefficients in Y
    ay =-(CY+2*AY)*np.ones(ny-2)
    by =  4*(1+AY)*np.ones(ny-2)
    cy = (CY-2*AY)*np.ones(ny-2)

    ## . . Treat ends
#     ax[0]=cx[nx-1]=ay[0]=cy[ny-1]=0

    ## . . Set up known Solution matrix
    dx = np.zeros((nx-2))
    dy = np.zeros((ny-2))

    ## . . Solve first for update in x direction
    for iy in range(1,ny-1):
        dx[:] =  -(CY-2*AY)*U[1:nx-1,iy+1]+\
                   4*(1-AY)*U[1:nx-1,iy  ]+\
                 +(CY+2*AY)*U[1:nx-1,iy-1]

        dx[0   ]+=+(CY+2*AY)*U[0   ,iy]
        dx[nx-3]+=-(CY-2*AY)*U[nx-1,iy]

        U1[1:nx-1,iy] = TDMAsolver(ax, bx, cx, dx)

    ## . . Solve second for update in yu direction
    for ix in range(1,nx-1):
        dy[:] =  -(CX-2*AX)*U1[ix+1,1:ny-1]+\
                   4*(1-AX)*U1[ix  ,1:ny-1]+\
                 +(CX+2*AX)*U1[ix-1,1:ny-1]

        dy[0   ]+=+(CX+2*AX)*U[ix,0   ]
        dy[ny-3]+=-(CX-2*AX)*U[ix,ny-1]

        U[ix,1:ny-1] = TDMAsolver(ay, by, cy, dy)

    return U

def Solve_Convection_Diffusion_homogeneous_Thomas(U,K,vx,dx,dt,LB,RB):
    # 1D convection solution function.
    '''Set up tridiagonal matrix and call Thomas Algorithm
    usage: x = Solve_Tridiagonal(a,b,c,u):
    input:
        u: heat solution at time step n   (nx)
        K : thermal diffusivity
        vx: convection velocity
        dx: spatial sampling
        dt: temporal sampling
        LB: Left  boundary condition (Dirchelet)
        RB: Right boundary condition (Dirchelet)
    output:
        u: heat solution at time step n+1 (nx)
    depends on:
        TDMAsolver

    written by Jeff Shragge, jshragge@mines.edu, 10/2019
    '''
    n = len(U)

    AA = K *dt/(dx*dx) ## Diffusivity: alpha
    CC = vx*dt/ dx     ## Courant :C

    ## . . Set up coefficients
    aa =-(CC+2*AA)*np.ones(n-2)
    bb =  4*(1+AA)*np.ones(n-2)
    cc = (CC-2*AA)*np.ones(n-2)

    ## . . Set up known solution matrrix
    dd = np.zeros((n-2))
    dd[:] = -(CC-2*AA)*U[2:n  ]+\
              4*(1-AA)*U[1:n-1]+\
            +(CC+2*AA)*U[0:n-2]
    dd[0   ]+=+(CC+2*AA)*U[0   ]
    dd[n-3]+=-(CC-2*AA)*U[n-1]
    tmp = TDMAsolver(aa, bb, cc, dd)
    U[1:n-1] = tmp

    return U

def Solve_2D_Convection_Diffusion_heterogeneous_Thomas(U,K,vx,vy,dx,dy,dt,F, TB, RB, LB, BB):
    # simulation of after shut in period
    '''Set up tridiagonal matrix and call Thomas Algorithm
    usage: x = Solve_2D_Convection_Diffusion_homogeneous_Thomas(U,K,vx,vy,dx,dy,dt,LB,RB,BB,TB):
    input:
        U: heat solution at time step n   (nx)
        K : thermal diffusivity
        vx: convection velocity in x-direction
        vy: convection velocity in y-direction
        dx: spatial sampling in x-direction
        dy: spatial sampling in y-direction
        dt: temporal sampling
    output:
        u: heat solution at time step n+1 (nx,ny)
        source: the pressure source serve as boundary condition.
    depends on:
        TDMAsolver

    based on code by Jeff Shragge, jshragge@mines.edu, 10/2019
    modified by Shenyao, shenyaojin@mines.edu, 10/2023
    '''
    ## . . Get dimensions
    nx,ny = np.size(U,0),np.size(U,1)
    U1 = np.zeros((nx,ny))

    ## Enforce boundary condition
    U[0,:] = TB
    U[-1,:] = BB
    U[:,0] = LB
    U[:,-1] = RB
    # Also forcing the boundary condition in the center part of the figure
    U[0, 50] = F[nx//2, ny//2]

    ## . . Define diffusivity and Courant numbers
    AX = K *dt/(dx*dx) ## Diffusivity x: alpha
    AY = K *dt/(dy*dy) ## Diffusivity y: alpha
    CX = vx*dt/ dx     ## Courant x : C
    CY = vy*dt/ dy     ## Courant y : C

    ## . . Set up coefficients in X
    ax =-(CX+2*AX)*np.ones(nx-2)
    bx =  4*(1+AX)*np.ones(nx-2)
    cx = (CX-2*AX)*np.ones(nx-2)

    ## . . Set up coefficients in Y
    ay =-(CY+2*AY)*np.ones(ny-2)
    by =  4*(1+AY)*np.ones(ny-2)
    cy = (CY-2*AY)*np.ones(ny-2)

    ## . . Treat ends
#     ax[0]=cx[nx-1]=ay[0]=cy[ny-1]=0

    ## . . Set up known Solution matrix
    dx = np.zeros((nx-2))
    dy = np.zeros((ny-2))

    ## . . Solve first for update in x direction
    for iy in range(1,ny-1):
        dx[:] =  -(CY-2*AY)*U[1:nx-1,iy+1]+\
                   4*(1-AY)*U[1:nx-1,iy  ]+\
                 +(CY+2*AY)*U[1:nx-1,iy-1]

        dx[0   ]+=+(CY+2*AY)*U[0   ,iy]
        dx[nx-3]+=-(CY-2*AY)*U[nx-1,iy]
        # force term
        # dx[:] += 2 * dt * F[1:nx-1,iy]

        U1[1:nx-1,iy] = TDMAsolver(ax, bx, cx, dx)

    ## . . Solve second for update in yu direction
    for ix in range(1,nx-1):
        dy[:] =  -(CX-2*AX)*U1[ix+1,1:ny-1]+\
                   4*(1-AX)*U1[ix  ,1:ny-1]+\
                 +(CX+2*AX)*U1[ix-1,1:ny-1]

        dy[0   ]+=+(CX+2*AX)*U[ix,0   ]
        dy[ny-3]+=-(CX-2*AX)*U[ix,ny-1]
        # dy[:] += 2 * dt * F[ix, 1:ny-1]

        U[ix,1:ny-1] = TDMAsolver(ay, by, cy, dy)

    return U

# modify the function
def Solve_2D_Convection_Diffusion_heterogeneous_Thomas_resorvoir(U, K, vx, vy, dx, dy, dt, K2, Pres):
    '''Set up tridiagonal matrix and call Thomas Algorithm
    pde solver before shut-in
    usage: x = Solve_2D_Convection_Diffusion_homogeneous_Thomas(U,K,vx,vy,dx,dy,dt,LB,RB,BB,TB):
    input:
        U: heat solution at time step n   (nx)
        K : thermal diffusivity
        vx: convection velocity in x-direction
        vy: convection velocity in y-direction
        dx: spatial sampling in x-direction
        dy: spatial sampling in y-direction
        dt: temporal sampling
        BC: Neumann boundary condition. Set all to zero.
    output:
        u: heat solution at time step n+1 (nx,ny)
        K2: connectivity of force term
        Pres: The pressure of the reservoir
    depends on:
        TDMAsolver

    based on code by Jeff Shragge, jshragge@mines.edu, 10/2019
    modified by Shenyao, shenyaojin@mines.edu, 11/2023
    '''
    ## . . Get dimensions
    nx, ny = np.size(U, 0), np.size(U, 1)
    U1 = np.zeros((nx, ny))
    F = K2 * (Pres - U)
    ## Enforce boundary condition: Neumann
    U[1, :] = U[0, :]
    U[-2, :] = U[-1, :]
    U[:, 1] = U[:, 0]
    U[:, -2] = U[:, -1]

    ## enforce center to zero
    U[nx//2, ny//2] = 0

    ## . . Define diffusivity and Courant numbers
    AX = K * dt / (dx * dx)  ## Diffusivity x: alpha
    AY = K * dt / (dy * dy)  ## Diffusivity y: alpha
    CX = vx * dt / dx  ## Courant x : C
    CY = vy * dt / dy  ## Courant y : C

    # change to np array to use reloaded "*"
    CX = np.array(CX)
    CY = np.array(CY)
    U1 = np.array(U1)
    U = np.array(U)

    ## . . Set up coefficients in X
    ax =-(CX+2*AX)*np.ones(nx-2)
    bx =  4*(1+AX)*np.ones(nx-2)
    cx = (CX-2*AX)*np.ones(nx-2)

    ## . . Set up coefficients in Y
    ay =-(CY+2*AY)*np.ones(ny-2)
    by =  4*(1+AY)*np.ones(ny-2)
    cy = (CY-2*AY)*np.ones(ny-2)
    ## . . Treat ends
    #     ax[0]=cx[nx-1]=ay[0]=cy[ny-1]=0

    ## . . Set up known Solution matrix
    dx = np.zeros((nx - 2))
    dy = np.zeros((ny - 2))

    # define the value of F
    # F = kappa2 * (P_res - P)

    ## . . Solve first for update in x direction
    for iy in range(1, ny - 1):
        dx[:] = -(CY - 2 * AY) * U[1:nx - 1, iy + 1] + \
                4 * (1 - AY) * U[1:nx - 1, iy] + \
                +(CY + 2 * AY) * U[1:nx - 1, iy - 1]
        # force term
        dx[:]+= 2 * dt * F[1:nx - 1, iy]

        dx[0] += +(CY + 2 * AY) * U[0, iy]
        dx[nx - 3] += -(CY - 2 * AY) * U[nx - 1, iy]

        U1[1:nx - 1, iy] = TDMAsolver(ax, bx, cx, dx)

    ## . . Solve second for update in yu direction
    for ix in range(1, nx - 1):
        dy[:] = -(CX - 2 * AX) * U1[ix + 1, 1:ny - 1] + \
                4 * (1 - AX) * U1[ix, 1:ny - 1] + \
                +(CX + 2 * AX) * U1[ix - 1, 1:ny - 1]
        # force term
        dy[:]+= 2 * dt * F[ix, 1:ny - 1]
        dy[0] += +(CX + 2 * AX) * U[ix, 0]
        dy[ny - 3] += -(CX - 2 * AX) * U[ix, ny - 1]

        U[ix, 1:ny - 1] = TDMAsolver(ay, by, cy, dy)

    return U

def Solve_2D_Convection_Diffusion_heterogeneous_Thomas_Neumann(U,K,vx,vy,dx,dy,dt, K2, Pres):
    # the convection function to simulate the shut in period
    # using Neumann condition to be more close to the actual situation
    '''Set up tridiagonal matrix and call Thomas Algorithm
    usage: x = Solve_2D_Convection_Diffusion_homogeneous_Thomas(U,K,vx,vy,dx,dy,dt,LB,RB,BB,TB):
    input:
        U: heat solution at time step n   (nx)
        K : thermal diffusivity
        vx: convection velocity in x-direction
        vy: convection velocity in y-direction
        dx: spatial sampling in x-direction
        dy: spatial sampling in y-direction
        dt: temporal sampling
    output:
        u: heat solution at time step n+1 (nx,ny)
        source: the pressure source serve as boundary condition.
    depends on:
        TDMAsolver

    based on code by Jeff Shragge, jshragge@mines.edu, 10/2019
    modified by Shenyao, shenyaojin@mines.edu, 10/2023
    '''
    ## . . Get dimensions
    ## . . Get dimensions
    nx, ny = np.size(U, 0), np.size(U, 1)
    U1 = np.zeros((nx, ny))
    F = K2 * (Pres - U)
    ## Enforce boundary condition: Neumann
    U[1, :] = U[0, :]
    U[-2, :] = U[-1, :]
    U[:, 1] = U[:, 0]
    U[:, -2] = U[:, -1]

    ## . . Define diffusivity and Courant numbers
    AX = K * dt / (dx * dx)  ## Diffusivity x: alpha
    AY = K * dt / (dy * dy)  ## Diffusivity y: alpha
    CX = vx * dt / dx  ## Courant x : C
    CY = vy * dt / dy  ## Courant y : C

    # change to np array to use reloaded "*"
    CX = np.array(CX)
    CY = np.array(CY)
    U1 = np.array(U1)
    U = np.array(U)

    ## . . Set up coefficients in X
    ax = -(CX + 2 * AX) * np.ones(nx - 2)
    bx = 4 * (1 + AX) * np.ones(nx - 2)
    cx = (CX - 2 * AX) * np.ones(nx - 2)

    ## . . Set up coefficients in Y
    ay = -(CY + 2 * AY) * np.ones(ny - 2)
    by = 4 * (1 + AY) * np.ones(ny - 2)
    cy = (CY - 2 * AY) * np.ones(ny - 2)
    ## . . Treat ends
    #     ax[0]=cx[nx-1]=ay[0]=cy[ny-1]=0

    ## . . Set up known Solution matrix
    dx = np.zeros((nx - 2))
    dy = np.zeros((ny - 2))

    # define the value of F
    # F = kappa2 * (P_res - P)

    ## . . Solve first for update in x direction
    for iy in range(1, ny - 1):
        dx[:] = -(CY - 2 * AY) * U[1:nx - 1, iy + 1] + \
                4 * (1 - AY) * U[1:nx - 1, iy] + \
                +(CY + 2 * AY) * U[1:nx - 1, iy - 1]
        # force term
        dx[:] += 2 * dt * F[1:nx - 1, iy]

        dx[0] += +(CY + 2 * AY) * U[0, iy]
        dx[nx - 3] += -(CY - 2 * AY) * U[nx - 1, iy]

        U1[1:nx - 1, iy] = TDMAsolver(ax, bx, cx, dx)

    ## . . Solve second for update in yu direction
    for ix in range(1, nx - 1):
        dy[:] = -(CX - 2 * AX) * U1[ix + 1, 1:ny - 1] + \
                4 * (1 - AX) * U1[ix, 1:ny - 1] + \
                +(CX + 2 * AX) * U1[ix - 1, 1:ny - 1]
        # force term
        dy[:] += 2 * dt * F[ix, 1:ny - 1]
        dy[0] += +(CX + 2 * AX) * U[ix, 0]
        dy[ny - 3] += -(CX - 2 * AX) * U[ix, ny - 1]

        U[ix, 1:ny - 1] = TDMAsolver(ay, by, cy, dy)
    return U
# original func
'''    nx,ny = np.size(U,0),np.size(U,1)
    U1 = np.zeros((nx,ny))

    F = K2 * (Pres - U)
    ## Enforce boundary condition
    U[0,:] = U[1,:]
    U[-1,:] = U[-2,:]
    U[:,0] = U[:,1]
    U[:,-1] = U[:,-2]
    # Also forcing the boundary condition in the center part of the figure

    ## . . Define diffusivity and Courant numbers
    AX = K *dt/(dx*dx) ## Diffusivity x: alpha
    AY = K *dt/(dy*dy) ## Diffusivity y: alpha
    CX = vx*dt/ dx     ## Courant x : C
    CY = vy*dt/ dy     ## Courant y : C

    ## . . Set up coefficients in X
    ax =-(CX+2*AX)*np.ones(nx-2)
    bx =  4*(1+AX)*np.ones(nx-2)
    cx = (CX-2*AX)*np.ones(nx-2)

    ## . . Set up coefficients in Y
    ay =-(CY+2*AY)*np.ones(ny-2)
    by =  4*(1+AY)*np.ones(ny-2)
    cy = (CY-2*AY)*np.ones(ny-2)

    ## . . Treat ends
#     ax[0]=cx[nx-1]=ay[0]=cy[ny-1]=0

    ## . . Set up known Solution matrix
    dx = np.zeros((nx-2))
    dy = np.zeros((ny-2))

    ## . . Solve first for update in x direction
    for iy in range(1,ny-1):
        dx[:] =  -(CY-2*AY)*U[1:nx-1,iy+1]+\
                   4*(1-AY)*U[1:nx-1,iy  ]+\
                 +(CY+2*AY)*U[1:nx-1,iy-1]

        dx[0   ]+=+(CY+2*AY)*U[0   ,iy]
        dx[nx-3]+=-(CY-2*AY)*U[nx-1,iy]
        # force term
        dx[:] += 2 * dt * F[1:nx-1,iy]

        U1[1:nx-1,iy] = TDMAsolver(ax, bx, cx, dx)

    ## . . Solve second for update in yu direction
    for ix in range(1,nx-1):
        dy[:] =  -(CX-2*AX)*U1[ix+1,1:ny-1]+\
                   4*(1-AX)*U1[ix  ,1:ny-1]+\
                 +(CX+2*AX)*U1[ix-1,1:ny-1]

        dy[0   ]+=+(CX+2*AX)*U[ix,0   ]
        dy[ny-3]+=-(CX-2*AX)*U[ix,ny-1]
        dy[:] += 2 * dt * F[ix, 1:ny-1]

        U[ix,1:ny-1] = TDMAsolver(ay, by, cy, dy)
        '''
