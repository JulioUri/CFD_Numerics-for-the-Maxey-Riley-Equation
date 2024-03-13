#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from subprocess import call
from progressbar import progressbar
from a03_FIELD0_QSCNT import velocity_field_Quiescent
from a09_PRTCLE_QSCNT import maxey_riley_relaxing
from a09_PRTCLE_PRSTH import maxey_riley_Prasath
from a09_PRTCLE_DTCHE import maxey_riley_Daitche

"""
Created on Tue Jan 30 17:56:36 2024
@author: Julio Urizarna-Carasa
"""

###############################################################################
################# Define field and implementation variables ###################
###############################################################################
#
save_plot_to = './OUTPUT/'                  # Folder where output is saved
y0           = np.array([0., 0.])           # Initial position
v0           = np.array([0.1, 0.])          # Initial velocity
tini         = 0.                           # Initial time
tend         = 1.                           # Final time
L            = 101                          # Time nodes
taxis        = np.linspace(tini, tend, L)   # Time axis
dt           = taxis[1] - taxis[0]          # time step
vel          = velocity_field_Quiescent()   # Flow field


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
N             = np.copy(L)  # Number of nodes
xi_fd_v       = np.linspace(0., 1., int(N))[:-1]

# Control constant (Koleva 2005)
c             = 20

# Logarithm map to obtain QUM
x_fd_v        = -c * np.log(1.0 - xi_fd_v)

# Parameter for Prasath et al.
nodes_dt      = 21 # Advised by Prasath et al.

# Parameter for Daitche
order_Daitche = 3

#
###############################################################################
######################### Create classes instances ############################
###############################################################################
#
# Particle classes in the left plot
#  - Class of the analytic solution in the Quiescent flow
Quiescent_left  = maxey_riley_relaxing(1, y0, v0, tini,
                                       particle_density    = rho1_p,
                                       fluid_density       = rho1_f,
                                       particle_radius     = rad_p,
                                       kinematic_viscosity = nu_f,
                                       time_scale          = t_scale)

#  - Class of particle trajectory calulated with Prasath et al.'s scheme
Prasath_particle  = maxey_riley_Prasath(1, y0, v0, vel,
                                        N, tini, dt, nodes_dt,
                                        particle_density    = rho1_p,
                                        fluid_density       = rho1_f,
                                        particle_radius     = rad_p,
                                        kinematic_viscosity = nu_f,
                                        time_scale          = t_scale)

# Particles classes in the right plot
#  - Class of the analytic solution in the Quiescent flow
Quiescent_right = maxey_riley_relaxing(1, y0, v0, tini,
                                       particle_density    = rho2_p,
                                       fluid_density       = rho2_f,
                                       particle_radius     = rad_p,
                                       kinematic_viscosity = nu_f,
                                       time_scale          = t_scale)

#  - Class of particle trajectory calculated with Daiche's scheme
Daitche_particle = maxey_riley_Daitche(1, y0, v0, vel, L,
                                       order_Daitche,
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
Prasath_pos     = np.array([y0])
for tt in progressbar(range(1, len(taxis))):
    Quiescent_left.solve(taxis[tt])
    Quiescent_right.solve(taxis[tt])
    Prasath_particle.update()
    Prasath_pos = np.vstack((Prasath_pos, Prasath_particle.pos_vec[tt * (nodes_dt-1)]))
    
if order_Daitche == 1:
    Daitche_particle.Euler(taxis, flag=True)
elif order_Daitche == 2:
    Daitche_particle.AdamBashf2(taxis, flag=True)
elif order_Daitche == 3:
    Daitche_particle.AdamBashf3(taxis, flag=True)


#
###############################################################################
############# Define limits of the plot and import velocity field #############
###############################################################################
#
# Bounds for Convergence velocity Field
x_left  = -0.002
x_right =  0.03


#
###############################################################################
##### Plot plots of figure with Particle's trajectories on velocity field #####
###############################################################################
#
fs   = 7
lw   = 1.2
ms   = 6

markers_on = np.arange(0, L, int((L-1)/10))


#############
# Left plot #
#############

plt.figure(1, layout='tight', figsize=(2.5, 2.15))

plt.plot(taxis, Quiescent_left.pos_vec[:,0],
              color='red', linewidth=lw, label="Analytical solution")
plt.plot(taxis, Prasath_pos[:,0],
              'x', markeredgewidth=lw, markersize=ms,
              color='green', label=("Prasath et al."), markevery=markers_on)

plt.xlabel('$time$', fontsize=fs, labelpad=0.25)
plt.ylabel('$y^{(1)}$', fontsize=fs, labelpad=0.25)
plt.tick_params(axis='both', labelsize=fs)
plt.legend(loc="lower right", fontsize=fs-1, prop={'size':fs-1})
plt.ylim([x_left, x_right])
plt.grid()

plt.savefig(save_plot_to + 'Figure_03a.pdf', format='pdf', dpi=400, bbox_inches='tight')

call(["pdfcrop", save_plot_to + 'Figure_03a.pdf', save_plot_to + 'Figure_03a.pdf'])

##############
# Right plot #
##############

plt.figure(2, layout='tight', figsize=(2.5, 2.15))

plt.plot(taxis, Quiescent_right.pos_vec[:,0],
              color='red', linewidth=lw, label="Analytical solution")
plt.plot(taxis, Daitche_particle.pos_vec[:,0],
              'x', markeredgewidth=lw, markersize=ms,
              color='blue', label=("Daitche 3rd order"), markevery=markers_on)
   
plt.xlabel('$time$', fontsize=fs, labelpad=0.25)
plt.ylabel('$y^{(1)}$', fontsize=fs, labelpad=0.25)
plt.tick_params(axis='both', labelsize=fs)
plt.legend(loc="lower right", fontsize=fs-1, prop={'size':fs-1})
plt.ylim([x_left, x_right])
plt.grid()

plt.savefig(save_plot_to + 'Figure_03b.pdf', format='pdf', dpi=400, bbox_inches='tight')

call(["pdfcrop", save_plot_to + 'Figure_03b.pdf', save_plot_to + 'Figure_03b.pdf'])

plt.show()