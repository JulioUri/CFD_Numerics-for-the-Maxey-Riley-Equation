#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from subprocess import call
from progressbar import progressbar
from a03_FIELD0_DATA1 import velocity_field_Faraday1
from a09_PRTCLE_IMEX4 import maxey_riley_imex
from a09_PRTCLE_TRAPZ import maxey_riley_trapezoidal
from a09_PRTCLE_DTCHE import maxey_riley_Daitche

"""
Created on Tue Jan 30 17:19:11 2024
@author: Julio Urizarna-Carasa
"""

###############################################################################
################# Define field and implementation variables ###################
###############################################################################
#
save_plot_to = './OUTPUT/'                  # Folder where output is saved
vel          = velocity_field_Faraday1(field_boundary=False)    # Flow field
tini         = 0.                           # Initial time
tend         = 5.                           # Final time
L            = 501                          # Time nodes
taxis        = np.linspace(tini, tend, L)   # Time axis
dt           = taxis[1] - taxis[0]          # time step
y0           = np.array([0.02, 0.01])       # Initial position
v0           = vel.get_velocity(y0[0], y0[1], tini) # Initial velocity


#
###############################################################################
######################### Define particle variables ###########################
###############################################################################
#
# Define parameters to obtain R (Remember R=(1+2*rho_p/rho_f)/3):
# - Left plot
rho1_p       = 2.0          # - Particle's density
rho1_f       = 3.0          # - Fluid's density

# - Right plot
rho2_p       = 3.0          # - Particle's density
rho2_f       = 2.0          # - Fluid's density

# Define parameters to obtain S (Remember S=rad_p**2/(3*nu_f*t_scale)):
rad_p        = 3.           # Particle's radius
nu_f         = 1.           # Kinematic viscocity
t_scale      = 10.          # Time Scale of the flow


#
###############################################################################
################# Define parameters for the numerical schemes #################
###############################################################################
#
# Define Uniform grid [0,1):
N           = np.copy(L)  # Number of nodes
xi_fd_v     = np.linspace(0., 1., int(N))[:-1]

# Control constant (Koleva 2005)
c           = 20

# Logarithm map to obtain QUM
x_fd_v      = -c * np.log(1.0 - xi_fd_v)

# Parameter for Daitche's method
order_Daitche = 3

#
###############################################################################
######################### Create classes instances ############################
###############################################################################
#
# Particle classes in the left plot
#  - Class of particle traecjtory calulated wtih Daitche's scheme
Daitche_left  = maxey_riley_Daitche(1, y0, v0, vel, L, order_Daitche,
                                       particle_density    = rho1_p,
                                       fluid_density       = rho1_f,
                                       particle_radius     = rad_p,
                                       kinematic_viscosity = nu_f,
                                       time_scale          = t_scale)

#  - Class of particle trajectory calulated with IMEX4 scheme
IMEX4_particle = maxey_riley_imex(1, y0, v0, vel, x_fd_v, c, dt, tini,
                                     particle_density    = rho1_p,
                                     fluid_density       = rho1_f,
                                     particle_radius     = rad_p,
                                     kinematic_viscosity = nu_f,
                                     time_scale          = t_scale,
                                     IMEXOrder           = 4,
                                     FDOrder             = 4,
                                     parallel_flag       = False)

# Particles classes in the right plot
#  - Class of particle traecjtory calulated wtih Daitche's scheme
Daitche_right  = maxey_riley_Daitche(1, y0, v0, vel, L, order_Daitche,
                                       particle_density    = rho2_p,
                                       fluid_density       = rho2_f,
                                       particle_radius     = rad_p,
                                       kinematic_viscosity = nu_f,
                                       time_scale          = t_scale)

#  - Class of particle trajectory calculated with the Trapezoidal Rule
Trapezoidal_particle = maxey_riley_trapezoidal(1, y0, v0, vel, x_fd_v,
                                                  c, dt, tini,
                                                  particle_density    = rho2_p,
                                                  fluid_density       = rho2_f,
                                                  particle_radius     = rad_p,
                                                  kinematic_viscosity = nu_f,
                                                  time_scale          = t_scale)


#
###############################################################################
##################### Calculate particle trajectories #########################
###############################################################################
#
# Calculate trajectories!
for tt in progressbar(range(1, len(taxis))):
    IMEX4_particle.update()
    Trapezoidal_particle.update()

if order_Daitche == 1:
    Daitche_left.Euler(taxis, flag=True)
    Daitche_right.Euler(taxis, flag=True)
elif order_Daitche == 2:
    Daitche_left.AdamBashf2(taxis, flag=True)
    Daitche_right.AdamBashf2(taxis, flag=True)
elif order_Daitche == 3:
    Daitche_left.AdamBashf3(taxis, flag=True)
    Daitche_right.AdamBashf3(taxis, flag=True)


#
###############################################################################
############# Define limits of the plot and import velocity field #############
###############################################################################
#
# Bounds for Convergence velocity Field
x_left  = 0.015
x_right = 0.035
y_down  = 0.0
y_up    = 0.0125


#
###############################################################################
########### Define spatial grid for the plotting of the Flow Field ############
###############################################################################
#
# Define points where to show the flow field
nx = 20
ny = 21

xaxis = np.linspace(x_left, x_right, nx)
yaxis = np.linspace(y_down, y_up, ny)
X, Y  = np.meshgrid(xaxis, yaxis)


#
###############################################################################
##### Plot plots of figure with Particle's trajectories on velocity field #####
###############################################################################
#
fs   = 7
lw   = 1.2
ms   = 6

u, v = vel.get_velocity(X, Y, taxis[-1])
ux, uy, vx, vy = vel.get_gradient(X, Y, taxis[-1])

markers_on = np.arange(0, L, int(L/10))


#############
# Left plot #
#############

plt.figure(1, layout='tight', figsize=(2.5, 2.15))

plt.quiver(X, Y, u, v)
plt.plot(Daitche_left.pos_vec[:,0], Daitche_left.pos_vec[:,1],
              color='red', linewidth=lw, label="Daitche, 3rd order")
plt.plot(IMEX4_particle.pos_vec[:,0], IMEX4_particle.pos_vec[:,1],
              'x', markeredgewidth=lw, markersize=ms,
              color='green', label=("IMEX 4th order"), markevery=markers_on)

plt.xlabel('$y^{(1)}$', fontsize=fs, labelpad=0.25)
plt.ylabel('$y^{(2)}$', fontsize=fs, labelpad=0.25)
plt.tick_params(axis='both', labelsize=fs)
plt.legend(loc="upper right", fontsize=fs-1, prop={'size':fs-1})
plt.xlim([x_left, x_right])
plt.ylim([y_down, y_up])

plt.savefig(save_plot_to + 'Figure_05a.pdf', format='pdf', dpi=400, bbox_inches='tight')

call(["pdfcrop", save_plot_to + 'Figure_05a.pdf', save_plot_to + 'Figure_05a.pdf'])

##############
# Right plot #
##############

plt.figure(2, layout='tight', figsize=(2.5, 2.15))

plt.quiver(X, Y, u, v)
plt.plot(Daitche_right.pos_vec[:,0], Daitche_right.pos_vec[:,1],
              color='red', linewidth=lw, label="Daitche 3rd order")
plt.plot(Trapezoidal_particle.pos_vec[:,0], Trapezoidal_particle.pos_vec[:,1], 'x',
              markeredgewidth=lw, markersize=ms,
              color='blue', label=("Trap. Rule"), markevery=markers_on)

plt.xlabel('$y^{(1)}$', fontsize=fs, labelpad=0.25)
plt.ylabel('$y^{(2)}$', fontsize=fs, labelpad=0.25)
plt.tick_params(axis='both', labelsize=fs)
plt.legend(loc="upper right", fontsize=fs-1, prop={'size':fs-1})
plt.xlim([x_left, x_right])
plt.ylim([y_down, y_up])

plt.savefig(save_plot_to + 'Figure_05b.pdf', format='pdf', dpi=400, bbox_inches='tight')

call(["pdfcrop", save_plot_to + 'Figure_05b.pdf', save_plot_to + 'Figure_05b.pdf'])

plt.show()