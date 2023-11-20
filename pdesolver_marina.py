import numpy as np
from pde_solver.pdesolver_parabolic import TDMAsolver

def Solve_2D_Convection_Diffusion_heterogeneous_Thomas(U,K,vx,vy,dx,dy,dt,Source):
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
        source: the pressure source serve as boundary condition. (in the center of the array)
    depends on:
        TDMAsolver

    based on code by Jeff Shragge, jshragge@mines.edu, 10/2019
    modified by Shenyao, shenyaojin@mines.edu, 10/2023
    '''
    ## . . Get dimensions
    nx,ny = np.size(U,0),np.size(U,1)
    U1 = np.zeros((nx,ny))

    # Also forcing the boundary condition in the center part of the figure
    U[nx//2, ny//2] = Source

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