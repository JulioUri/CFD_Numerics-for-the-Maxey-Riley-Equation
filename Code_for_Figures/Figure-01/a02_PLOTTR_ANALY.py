#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from progressbar import progressbar
from a03_FIELD0_ANALY import velocity_field_Analytical
from a09_PRTCLE_ANALY import maxey_riley_analytic
from a09_PRTCLE_IMEX4 import maxey_riley_imex
from a09_PRTCLE_DIRK4 import maxey_riley_dirk

"""
Created on Tue Jan 30 11:17:01 2024
@author: Julio Urizarna-Carasa
"""

###############################################################################
################# Define field and implementation variables ###################
###############################################################################
#
save_plot_to = './VISUAL_OUTPUT/'
tini         = 0.                         # Initial time
tend         = 10.                        # Final time
L            = 101                        # Time nodes
taxis        = np.linspace(tini, tend, L) # Time axis
dt           = taxis[1] - taxis[0]        # time step
omega_value  = 1.                         # Angular velocity of the field 
vel          = velocity_field_Analytical(omega=omega_value) # Flow field
y0           = np.array([1., 0.])         # Initial position
v0           = vel.get_velocity(y0[0], y0[1], tini)  # Initial velocity


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
# Define Uniform grid [0,1]:
N           = np.copy(L)  # Number of nodes
xi_fd_v     = np.linspace(0., 1., int(N))[:-1]

# Control constant (Koleva 2005)
c           = 20

# Logarithm map to obtain QUM
x_fd_v      = -c * np.log(1.0 - xi_fd_v)


#
###############################################################################
######################### Create classes instances ############################
###############################################################################
#
# Particles in the left plot
Analytic_left  = maxey_riley_analytic(1, y0, v0, tini, vel,
                                      particle_density    = rho1_p,
                                      fluid_density       = rho1_f,
                                      particle_radius     = rad_p,
                                      kinematic_viscosity = nu_f,
                                      time_scale          = t_scale)

DIRK_particle  = maxey_riley_dirk(1, y0, v0, vel, x_fd_v, c, dt, tini,
                                      particle_density    = rho1_p,
                                      fluid_density       = rho1_f,
                                      particle_radius     = rad_p,
                                      kinematic_viscosity = nu_f,
                                      time_scale          = t_scale,
                                      parallel_flag       = False)

# Particles in the right plot
Analytic_right = maxey_riley_analytic(1, y0, v0, tini, vel,
                                      particle_density    = rho2_p,
                                      fluid_density       = rho2_f,
                                      particle_radius     = rad_p,
                                      kinematic_viscosity = nu_f,
                                      time_scale          = t_scale)

IMEX4_particle = maxey_riley_imex(1, y0, v0, vel, x_fd_v, c, dt, tini,
                                      particle_density    = rho2_p,
                                      fluid_density       = rho2_f,
                                      particle_radius     = rad_p,
                                      kinematic_viscosity = nu_f,
                                      time_scale          = t_scale,
                                      IMEXOrder           = 4,
                                      FDOrder             = 4,
                                      parallel_flag       = False)


#
###############################################################################
##################### Calculate particle trajectories #########################
###############################################################################
#
# Calculate trajectories!
for tt in progressbar(range(1, len(taxis))):
    Analytic_left.solve(taxis[tt])
    Analytic_right.solve(taxis[tt])
    IMEX4_particle.update()
    DIRK_particle.update()


#
###############################################################################
############# Define limits of the plot and import velocity field #############
###############################################################################
#
# Bounds for Convergence velocity Field
x_left  = -1.5
x_right = 1.5
y_down  = -1.5
y_up    = 1.5


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
X, Y = np.meshgrid(xaxis, yaxis)


#
###############################################################################
##### Plot plots in figure with Particle's trajectories on velocity field #####
###############################################################################
#
fig1 = plt.figure(1, layout='tight')
fig1.set_figwidth(4.2)
fig1.set_figheight(3.6)
fs = 13
lw = 2.2

u, v = vel.get_velocity(X, Y, taxis[-1])
ux, uy, vx, vy = vel.get_gradient(X, Y, taxis[-1])

markers_on = np.arange(0, L, int((L-1)/20))


#############
# Left plot #
#############

plt.quiver(X, Y, u, v)
plt.plot(Analytic_left.pos_vec[:,0], Analytic_left.pos_vec[:,1],
              color='red', linewidth=lw, label="Analytical solution")
plt.plot(DIRK_particle.pos_vec[:,0], DIRK_particle.pos_vec[:,1], 'x', markeredgewidth=lw, markersize=10,
              color='green', label=("DIRK, 4th order"), markevery=markers_on)
plt.xlabel('$y^{(1)}$', fontsize=fs, labelpad=0.25)
plt.ylabel('$y^{(2)}$', fontsize=fs, labelpad=0.25)
plt.tick_params(axis='both', labelsize=fs)
plt.legend(loc="lower left", fontsize=fs, prop={'size':fs-4})
plt.xlim([x_left, x_right])
plt.ylim([y_down, y_up])

plt.savefig(save_plot_to + 'Figure_01a.pdf', format='pdf', dpi=400, bbox_inches='tight')


##############
# Right plot #
##############

fig2 = plt.figure(2, layout='tight', figsize=(3.6, 1.8))
fig2.set_figwidth(4.2)
fig2.set_figheight(3.6)
plt.quiver(X, Y, u, v)
plt.plot(Analytic_right.pos_vec[:,0], Analytic_right.pos_vec[:,1],
              color='red', linewidth=lw, label="Analytical solution")
plt.plot(IMEX4_particle.pos_vec[:,0], IMEX4_particle.pos_vec[:,1], 'x', markeredgewidth=lw, markersize=10,
              color='blue', label=("IMEX 4th order"), markevery=markers_on)    
plt.xlabel('$y^{(1)}$', fontsize=fs, labelpad=0.25)
plt.ylabel('$y^{(2)}$', fontsize=fs, labelpad=0.25)
plt.tick_params(axis='both', labelsize=fs)
plt.legend(loc="lower left", fontsize=fs, prop={'size':fs-4})
plt.xlim([x_left, x_right])
plt.ylim([y_down, y_up])

plt.savefig(save_plot_to + 'Figure_01b.pdf', format='pdf', dpi=400, bbox_inches='tight')

plt.show()