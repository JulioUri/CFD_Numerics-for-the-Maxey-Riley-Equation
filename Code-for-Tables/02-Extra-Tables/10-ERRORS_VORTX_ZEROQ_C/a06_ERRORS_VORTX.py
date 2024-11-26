#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import time
# import matplotlib.pyplot as plt
# import pandas as pd
from tabulate import tabulate
from a03_FIELD0_VORTX import velocity_field_Analytical
from a09_PRTCLE_VORTX import maxey_riley_analytic
from a09_PRTCLE_IMEX4 import maxey_riley_imex
from a09_PRTCLE_DIRK4 import maxey_riley_dirk
from a09_PRTCLE_TRAPZ import maxey_riley_trapezoidal
# from a09_PRTCLE_PRSTH import maxey_riley_Prasath
# from a09_PRTCLE_DTCHE import maxey_riley_Daitche

"""
Created on Thu Feb 01 10:48:56 2024
@author: Julio Urizarna-Carasa
"""

#
###############################################################################
################# Define field and implementation variables ###################
###############################################################################
#
save_plot_to = './OUTPUT/'                # Folder where output is saved
tini         = 0.                         # Initial time
tend         = 1.                         # Final time
omega_value  = 1.                         # Angular velocity of the field 
vel          = velocity_field_Analytical(omega=omega_value) # Flow field
y0           = np.array([1., 0.])         # Initial position
v0           = vel.get_velocity(y0[0], y0[1], tini) # Initial velocity

# Define vector of time and spatial nodes
N          = 101

#
###############################################################################
######################### Define particle variables ###########################
###############################################################################
#
# Define parameters to obtain R (Remember R=(1+2*rho_p/rho_f)/3):
rho_p       = 2.   # - Particle's density
rho_f       = 3.   # - Fluid's density

# Define parameters to obtain S (Remember S=rad_p**2/(3*nu_f*t_scale)):
rad_p        = np.sqrt(3.)  # Particle's radius
nu_f         = 1.           # Kinematic viscocity
t_scale      = 10.          # Time Scale of the flow

#
###############################################################################
################# Define parameters for the numerical schemes #################
###############################################################################
#
# Control constant (Koleva 2005)
c_v          = np.array([2., 5., 10., 15., 20., 30., 50., 100., 200.])

# # Define the order for Daitche's method
# order_Daitche = 3

# # Define number of time Chebyshev nodes in each time interval
# nodes_dt    = 21 # Advised by Prasath et al. (2019)

#
#########################################
# Start loop over the densities vectors #
#########################################
#

# initialise count variable
count         = 0

# Start loop
Trap_err_dic    = dict()
# Daitche_err_dic = dict()
# Prasath_err_dic = dict()
IMEX2_err_dic   = dict()
IMEX4_err_dic   = dict()
DIRK4_err_dic   = dict()
for j in range(0, len(c_v)):    
    Trap_err_v    = np.array([])
    # Daitche_err_v = np.array([])
    # Prasath_err_v = np.array([])
    IMEX2_err_v   = np.array([])
    IMEX4_err_v   = np.array([])
    DIRK4_err_v   = np.array([])
    
    taxis     = np.linspace(tini, tend, N) # Time axis
    dt        = taxis[1] - taxis[0]        # time step
    count    += 1
    
    t0        = time.time()
        
    #
    ################################
    # Calculate reference solution #
    ################################
    #
    Analytic_particle  = maxey_riley_analytic(1, y0, v0, tini, vel,
                                           particle_density    = rho_p,
                                           fluid_density       = rho_f,
                                           particle_radius     = rad_p,
                                           kinematic_viscosity = nu_f,
                                           time_scale          = t_scale)
        
    #
    #######################################################################
    ############# Define parameters for the numerical schemes #############
    #######################################################################
    #
    # Define Uniform grid [0,1]:
    xi_fd_v     = np.linspace(0., 1., N)[:-1]

    # Logarithm map to obtain QUM
    x_fd_v      = -c_v[j] * np.log(1.0 - xi_fd_v)

    Trap_particle    = maxey_riley_trapezoidal(1, y0, v0, vel,
                                              x_fd_v, c_v[j], dt, tini,
                                              particle_density    = rho_p,
                                              fluid_density       = rho_f,
                                              particle_radius     = rad_p,
                                              kinematic_viscosity = nu_f,
                                              time_scale          = t_scale)
    
    IMEX2_particle   = maxey_riley_imex(1, y0, v0, vel,
                                        x_fd_v, c_v[j], dt, tini,
                                        particle_density    = rho_p,
                                        fluid_density       = rho_f,
                                        particle_radius     = rad_p,
                                        kinematic_viscosity = nu_f,
                                        time_scale          = t_scale,
                                        IMEXOrder           = 2,
                                        FDOrder             = 2,
                                        parallel_flag       = False)
    
    IMEX4_particle   = maxey_riley_imex(1, y0, v0, vel,
                                        x_fd_v, c_v[j], dt, tini,
                                        particle_density    = rho_p,
                                        fluid_density       = rho_f,
                                        particle_radius     = rad_p,
                                        kinematic_viscosity = nu_f,
                                        time_scale          = t_scale,
                                        IMEXOrder           = 4,
                                        FDOrder             = 4,
                                        parallel_flag       = False)

    DIRK4_particle   = maxey_riley_dirk(1, y0, v0, vel,
                                       x_fd_v, c_v[j], dt, tini,
                                       particle_density    = rho_p,
                                       fluid_density       = rho_f,
                                       particle_radius     = rad_p,
                                       kinematic_viscosity = nu_f,
                                       time_scale          = t_scale,
                                       parallel_flag       = False)

    # Daitche_particle = maxey_riley_Daitche(1, y0, v0, vel, L_v[ll],
    #                                        order_Daitche,
    #                                        particle_density    = rho_p,
    #                                        fluid_density       = rho_f,
    #                                        particle_radius     = rad_p,
    #                                        kinematic_viscosity = nu_f,
    #                                        time_scale          = t_scale)

    # Prasath_particle = maxey_riley_Prasath(1, y0, v0, vel, N, tini, dt,
    #                                         nodes_dt,
    #                                         particle_density    = rho_p,
    #                                         fluid_density       = rho_f,
    #                                         particle_radius     = rad_p,
    #                                         kinematic_viscosity = nu_f,
    #                                         time_scale          = t_scale)
    
    # Prasath_pos      = np.array([y0])
    for tt in range(1, len(taxis)):
        Analytic_particle.solve(taxis[tt])
    
        # try:
        #     Prasath_particle.update()
        #     Prasath_pos = np.vstack((Prasath_pos,
        #                              Prasath_particle.pos_vec[tt * (nodes_dt-1)]))
        # except:
        #     None
    
        Trap_particle.update()
        IMEX2_particle.update()
        IMEX4_particle.update()
        DIRK4_particle.update()
    
    # try:
    #     Prasath_err     = Prasath_pos - Analytic_particle.pos_vec
    #     Prasath_err_max = max(np.linalg.norm(Prasath_err, ord=2, axis=1))
    #     Prasath_err_v   = np.append(Prasath_err_v, Prasath_err_max)
    # except:
    #     None

    Trap_err      = Trap_particle.pos_vec - Analytic_particle.pos_vec
    Trap_err_max  = max(np.linalg.norm(Trap_err, ord=2, axis=1))
    Trap_err_v    = np.append(Trap_err_v, Trap_err_max)

    IMEX2_err     = IMEX2_particle.pos_vec - Analytic_particle.pos_vec
    IMEX2_err_max = max(np.linalg.norm(IMEX2_err, ord=2, axis=1))
    IMEX2_err_v   = np.append(IMEX2_err_v, IMEX2_err_max)

    IMEX4_err     = IMEX4_particle.pos_vec - Analytic_particle.pos_vec
    IMEX4_err_max = max(np.linalg.norm(IMEX4_err, ord=2, axis=1))
    IMEX4_err_v   = np.append(IMEX4_err_v, IMEX4_err_max)

    DIRK4_err     = DIRK4_particle.pos_vec - Analytic_particle.pos_vec
    DIRK4_err_max = max(np.linalg.norm(DIRK4_err, ord=2, axis=1))
    DIRK4_err_v   = np.append(DIRK4_err_v, DIRK4_err_max)

    # if order_Daitche == 1:
    #     Daitche_particle.Euler(taxis, flag=True)  # First order method
    # elif order_Daitche == 2:
    #     Daitche_particle.AdamBashf2(taxis, flag=True) # Second order method
    # elif order_Daitche == 3:
    #     Daitche_particle.AdamBashf3(taxis, flag=True) # Third order method

    # Daitche_err     = Daitche_particle.pos_vec - Analytic_particle.pos_vec
    # Daitche_err_max = max(np.linalg.norm(Daitche_err, ord=2, axis=1))
    # Daitche_err_v   = np.append(Daitche_err_v, Daitche_err_max)
    
    tf              = time.time()
    
    print("\n   Round number " + str(count) + " finished in " + str(round(tf - t0, 2)) + " seconds.\n")
        
    Trap_err_dic[j]    = Trap_err_v
    # Daitche_err_dic[j] = Daitche_err_v
    # Prasath_err_dic[j] = Prasath_err_v
    IMEX2_err_dic[j]   = IMEX2_err_v
    IMEX4_err_dic[j]   = IMEX4_err_v
    DIRK4_err_dic[j]   = DIRK4_err_v

#
###############################################################################
##################### Create Table with convergence orders ####################
###############################################################################
#
# Create convergence table
mydata = [
          [c_v[0], c_v[1], c_v[2],
           c_v[3], c_v[4], c_v[5],
           c_v[6], c_v[7], c_v[8]],
          [IMEX2_err_dic[0],   IMEX2_err_dic[1],   IMEX2_err_dic[2],
           IMEX2_err_dic[3],   IMEX2_err_dic[4],   IMEX2_err_dic[5],
           IMEX2_err_dic[6],   IMEX2_err_dic[7],   IMEX2_err_dic[8]],
          [Trap_err_dic[0],    Trap_err_dic[1],    Trap_err_dic[2],
           Trap_err_dic[3],    Trap_err_dic[4],    Trap_err_dic[5],
           Trap_err_dic[6],    Trap_err_dic[7],    Trap_err_dic[8]],
          [IMEX4_err_dic[0],   IMEX4_err_dic[1],   IMEX4_err_dic[2],
           IMEX4_err_dic[3],   IMEX4_err_dic[4],   IMEX4_err_dic[5],
           IMEX4_err_dic[6],   IMEX4_err_dic[7],   IMEX4_err_dic[8]],
          [DIRK4_err_dic[0],   DIRK4_err_dic[1],   DIRK4_err_dic[2],
           DIRK4_err_dic[3],   DIRK4_err_dic[4],   DIRK4_err_dic[5],
           DIRK4_err_dic[6],   DIRK4_err_dic[7],   DIRK4_err_dic[8]]]

# create header
head = ["c", "FD2 + IMEX2:", "FD2 + Trap.:", "FD2 + IMEX4:", "FD2 + DIRK4:"]

# Transpose the data using zip
transposed_data = list(zip(*mydata))

print("\nErrors")
print("\n" + tabulate(transposed_data, headers=head, tablefmt="grid"))

with open(save_plot_to + 'Errors.txt', 'w') as file:
    file.write("Errors\n")
    file.write( str(tabulate(transposed_data, headers=head, tablefmt="grid") ))


print("\007")
