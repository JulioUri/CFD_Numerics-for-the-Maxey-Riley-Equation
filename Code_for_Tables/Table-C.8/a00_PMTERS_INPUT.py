#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy
import multiprocessing as mp
import sys

from a03_FIELD0_BICKL import velocity_field_Bickley
from a03_FIELD0_VORTX import velocity_field_Vortex
from a03_FIELD0_ANALY import velocity_field_Analytical
from a03_FIELD0_DATA1 import velocity_field_Faraday1
from a03_FIELD0_DATA2 import velocity_field_Faraday2
from a03_FIELD0_STEDY import velocity_field_Steady
from a03_FIELD0_QSCNT import velocity_field_Quiescent
from a03_FIELD0_OSCLT import velocity_field_Oscillatory
from a03_FIELD0_DGYRE import velocity_field_DoubleGyre
from a03_FIELD0_CGYRE import velocity_field_ConstantGyre
from a03_FIELD0_OURS1 import velocity_field_Ours

"""
Created on Thu Mar 24 15:47:47 2022
@author: Julio Urizarna Carasa

This file contains some of the parameters needed to run the SOLVER files.
"""



##############################################
# Define repository where data must be saved #
##############################################

# Any Computer:
save_output_to = './01_DATA0_OUTPUT/'



###########################################################
# Define repositories where data must be loaded and saved #
###########################################################

# Any Computer:
load_input_from = './01_DATA0_OUTPUT/'
save_plot_to    = './02_VSUAL_OUTPUT/'



################################
# Decide velocity field to use #
################################

Case_v    = np.array(["Bickley",     "Vortex",                # 0, 1,
                      "Faraday1",    "Faraday2",              # 2, 3,
                      "Analytical",  "Quiescent",             # 4, 5,
                      "Steady",      "Oscillatory",           # 6, 7,
                      "Double Gyre", "Cte Double Gyre",       # 8, 9.
                      "Ours"])                                # 10.


Case_elem = 4



####################
# Define time grid #
####################

# Initial time
tini  = 0.0
tend  = 1.0  # Final time
nt    = 101  # Time nodes


# For experimental flows, the final time cannot go beyond a threshold.
if Case_v[Case_elem] == Case_v[2] and tend > 42.240:
    # Time domain for Faraday field is restricted according to available data.
    tend = 42.240
elif Case_v[Case_elem] == Case_v[3] and tend > 7.64:
    # Time domain for Faraday field is restricted according to available data.
    tend = 7.64



# Create time axis
taxis  = np.linspace(tini, tend, nt)
dt     = taxis[1] - taxis[0]



############################################
# Define particle's and fluid's parameters #
############################################

# Densities:
# - Particle's density
rho_p   = 3.0

# - Fluid's density
rho_f   = 2.0

# Particle's radius
rad_p   = np.sqrt(3.0)

# Kinematic viscocity
nu_f    = 1.0

# Time Scale of the flow
t_scale = 0.25



################################
# Import chosen velocity field #
################################

if Case_v[Case_elem] == Case_v[0]:
    '''Bickley Jet velocity field'''
    vel     = velocity_field_Bickley()
elif Case_v[Case_elem] == Case_v[1]:
    '''Vortex velocity Field'''
    vel     = velocity_field_Vortex()
elif Case_v[Case_elem] == Case_v[2]:
    '''Faraday Velocity Field'''
    vel     = velocity_field_Faraday1(field_boundary=True)
elif Case_v[Case_elem] == Case_v[3]:
    '''Faraday Velocity Field'''
    vel     = velocity_field_Faraday2(field_boundary=True)
elif Case_v[Case_elem] == Case_v[4]:
    '''Convergence velocity Field (also a Vortex)'''
    omega_value = 1.0
    vel     = velocity_field_Analytical(omega=omega_value)
elif Case_v[Case_elem] == Case_v[5]:
    '''Quiescent fluid: null velocity Field'''
    vel     = velocity_field_Quiescent()   
elif Case_v[Case_elem] == Case_v[6]:
    '''Steady fluid: steady velocity Field taking from a snapshot of the Faraday1 at t=0'''
    vel     = velocity_field_Steady(field_boundary=True)
elif Case_v[Case_elem] == Case_v[7]:
    '''Oscillatory flow: f(t) = ( 0.05, sin( lambda * t) )'''
    vel     = velocity_field_Oscillatory()
elif Case_v[Case_elem] == Case_v[8]:
    '''Double Gyre background flow.'''
    vel     = velocity_field_DoubleGyre()
elif Case_v[Case_elem] == Case_v[9]:
    '''Double CONSTANT Gyre background flow.'''
    vel     = velocity_field_ConstantGyre()
elif Case_v[Case_elem] == Case_v[10]:
    '''Our own made-up velocity field.'''
    vel     = velocity_field_Ours()


###################################################
# Function defining limits of the background flow #
###################################################


def plotlims(Case_elem):

    #############################
    # Define limits of the plot #
    #############################

    if Case_v[Case_elem] == Case_v[0]:
        '''Bounds for Bickley Jet velocity field'''
        x_left  = 0.0
        x_right = 20.0
        y_down  = -4.0
        y_up    = 4.0
        vel     = velocity_field_Bickley()
    elif Case_v[Case_elem] == Case_v[1]:
        '''Bounds for Vortex velocity Field'''
        x_left  = -5.0
        x_right = 5.0
        y_down  = -5.0
        y_up    = 5.0
        vel     = velocity_field_Vortex()
    elif Case_v[Case_elem] == Case_v[2]:
        '''Bounds for Data-based field'''
        x_left  = 0.0
        x_right = 0.07039463189473738
        y_down  = 0.0
        y_up    = 0.0524872255355498
        vel     = velocity_field_Faraday1(field_boundary=False)
    elif Case_v[Case_elem] == Case_v[3]:
        '''Bounds for Data-based field'''
        x_left  = 0.0
        x_right = 0.07039463189473738
        y_down  = 0.0
        y_up    = 0.0524872255355498
        vel     = velocity_field_Faraday2(field_boundary=False)
    elif Case_v[Case_elem] == Case_v[4]:
        '''Convergence velocity Field (also a Vortex)'''
        x_left  = -1.5
        x_right = 1.5
        y_down  = -1.5
        y_up    = 1.5
        vel     = velocity_field_Analytical(time_scale=t_scale)
    elif Case_v[Case_elem] == Case_v[5]:
        '''Quiescent fluid: null velocity Field'''
        x_left  = -1.5
        x_right = 1.5
        y_down  = -1.5
        y_up    = 1.5
        vel     = velocity_field_Quiescent()
    elif Case_v[Case_elem] == Case_v[6]:
        '''Bounds for Data-based field'''
        x_left  = 0.0
        x_right = 0.07039463189473738
        y_down  = 0.0
        y_up    = 0.0524872255355498
        vel     = velocity_field_Steady(field_boundary=False)
    elif Case_v[Case_elem] == Case_v[7]:
        '''Bounds for Oscillatory flow'''
        x_left  = -0.02
        x_right = .18
        y_down  = -0.15
        y_up    = .4
        vel     = velocity_field_Oscillatory()
    elif Case_v[Case_elem] == Case_v[8]:
        '''Bounds for the Double Gyre'''
        x_left  = 0.0
        x_right = 2.0
        y_down  = 0.0
        y_up    = 1.0
        vel     = velocity_field_DoubleGyre()
    elif Case_v[Case_elem] == Case_v[9]:
        '''Bounds for the Double Gyre'''
        x_left  = 0.0
        x_right = 2.0
        y_down  = 0.0
        y_up    = 1.0
        vel     = velocity_field_ConstantGyre()
    elif Case_v[Case_elem] == Case_v[10]:
        '''Bounds for our own field'''
        x_left  = -1.5
        x_right =  1.5
        y_down  = -1.5
        y_up    =  1.5
        vel     = velocity_field_Ours()

    return x_left, x_right, y_down, y_up, vel
    


########################################
# Define number of nodes per time step #
########################################

# Nodes in each time step as per Prasath et al. (2019) method #
nodes_dt    = 20



###############################################################
# Definition of the pseudo-space grid for FD as Koleva (2005) #
###############################################################

# Define Uniform grid [0,1]:
xi0_fd      = 0.0
xif_fd      = 1.0
N_fd        = 101  # Number of nodes
xi_fd_v     = np.linspace(xi0_fd, xif_fd, int(N_fd))[:-1]

# Control constant (Koleva 2005)
c           = 20

# Logarithm map to obtain QUM
x_fd_v      = -c * np.log(1.0 - xi_fd_v)



#################################################
# Definition of the pseudo-space grid for Fokas #
#################################################

# Nodes in the frequency domain, k, as per Prasath et al. (2019)
N_fokas     = 101



##############################################
# Decide whether to apply parallel computing #
##############################################

parallel_flag = True
number_cores  = int(sys.argv[1]) if (len(sys.argv) > 1) and (int(sys.argv[1]) < mp.cpu_count()) else int(mp.cpu_count())



########################################
# Define particle's initial conditions #
########################################


# mat    = scipy.io.loadmat('IniCond.mat')
# x0     = mat['X']
# y0     = mat['Y']

# u0, v0 = vel.get_velocity(x0, y0, tini)


# u0     = mat['vx']
# v0     = mat['vy']


# if Case_v[Case_elem] == Case_v[2]:
#     x0, y0 = 0.02,   0.02
#     u0, v0 = vel.get_velocity(x0, y0, tini) #0.0015, 0.0015
# elif Case_v[Case_elem] == Case_v[6]:
#     x0, y0 = 0.04,  0.03
#     u0, v0 = 0.001, 0.001
# elif Case_v[Case_elem] == Case_v[7]:
#     x0, y0 = 0.0, 0.0
#     u0, v0 = vel.get_velocity(x0, y0, tini)
# elif Case_v[Case_elem] == Case_v[5]:
#     x0, y0 = 0.0, 0.0
#     u0, v0 = 1.0, 0.0 
# else:
#     x0, y0 = 1.0, 0.0
#     u0, v0 = vel.get_velocity(x0, y0, tini)



############################################################
# Choose convergence order for Direct Integration and IMEX #
############################################################

# # For Daitche's method, values 1, 2 and 3 are available
order_Daitche = 3

# For IMEX implementation, define the convergence order of the finite
# differences methods. It should either be 2 or 4. Koleva's scheme
# corresponds to 2 and our own corresponds to 4.
order_FD   = 4

# For IMEX implementation, values 1, 2, 3, 4 are available
order_IMEX = 2