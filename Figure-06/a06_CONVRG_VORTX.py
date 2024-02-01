#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import time
import matplotlib.pyplot as plt
from tabulate import tabulate
import multiprocessing as mp
import sys
from a03_FIELD0_ANALY import velocity_field_Analytical
from a09_PRTCLE_ANALY import maxey_riley_analytic
from a09_PRTCLE_IMEX4 import maxey_riley_imex
from a09_PRTCLE_DIRK4 import maxey_riley_dirk
from a09_PRTCLE_TRAPZ import maxey_riley_trapezoidal
from a09_PRTCLE_PRSTH import maxey_riley_Prasath
from a09_PRTCLE_DTCHE import maxey_riley_Daitche

"""
Created on Thu Feb 01 10:48:56 2024
@author: Julio Urizarna-Carasa
"""

def compute_particles(particle_dict, L_vec, pos_ini, order_Daitche):
    taxis     = np.linspace(tini, tend, L_vec[ll]) # Time axis
    
    Trap_err_v    = np.array([])
    Daitche_err_v = np.array([])
    Prasath_err_v = np.array([])
    IMEX2_err_v   = np.array([])
    IMEX4_err_v   = np.array([])
    DIRK_err_v    = np.array([])
    
    Prasath_pos     = np.array([y0])
    for tt in range(1, len(taxis)):
        particle_dict["Analytic"].solve(taxis[tt])
    
        try:
            particle_dict["Analytic"].update()
            Prasath_pos = np.vstack((Prasath_pos,
                                     particle_dict["Analytic"].pos_vec[tt * (nodes_dt-1)]))
        except:
            None
    
        particle_dict["Trapezoidal"].update()
        particle_dict["IMEX2"].update()
        particle_dict["IMEX4"].update()
        particle_dict["DIRK4"].update()
    
    try:
        Prasath_err     = Prasath_pos - particle_dict["Analytic"].pos_vec
        Prasath_err_max = max(np.linalg.norm(Prasath_err, 2, axis=1))
        Prasath_err_v   = np.append(Prasath_err_v, Prasath_err_max)
    except:
        None

    Trap_err      = particle_dict["Trapezoidal"].pos_vec - particle_dict["Analytic"].pos_vec
    Trap_err_max  = np.max(np.linalg.norm(Trap_err, ord=np.inf, axis=1))
    Trap_err_v    = np.append(Trap_err_v, Trap_err_max)

    IMEX2_err     = particle_dict["IMEX2"].pos_vec - particle_dict["Analytic"].pos_vec
    IMEX2_err_max = np.max(np.linalg.norm(IMEX2_err, ord=np.inf, axis=1))
    IMEX2_err_v   = np.append(IMEX2_err_v, IMEX2_err_max)

    IMEX4_err     = particle_dict["IMEX4"].pos_vec - particle_dict["Analytic"].pos_vec
    IMEX4_err_max = np.max(np.linalg.norm(IMEX4_err, ord=np.inf, axis=1))
    IMEX4_err_v   = np.append(IMEX4_err_v, IMEX4_err_max)

    DIRK_err      = particle_dict["DIRK4"].pos_vec - particle_dict["Analytic"].pos_vec
    DIRK_err_max  = np.max(np.linalg.norm(DIRK_err, ord=np.inf, axis=1))
    DIRK_err_v    = np.append(DIRK_err_v, DIRK_err_max)

    if order_Daitche == 1:
        particle_dict["Daitche"].Euler(taxis, flag=True)  # First order method
    elif order_Daitche == 2:
        particle_dict["Daitche"].AdamBashf2(taxis, flag=True) # Second order method
    elif order_Daitche == 3:
        particle_dict["Daitche"].AdamBashf3(taxis, flag=True) # Third order method

    Daitche_err     = particle_dict["Daitche"].pos_vec - particle_dict["Analytic"].pos_vec
    Daitche_err_max = np.max(np.linalg.norm(Daitche_err, ord=np.inf, axis=1))
    Daitche_err_v   = np.append(Daitche_err_v, Daitche_err_max)

    # tf              = time.time()

    # print("- Round number " + str(count) + " finished in " + str(round(tf - t0, 2)) + " seconds.")
    
    # Continue here, trying to figure out how to save the values in parallel. #
    
    Trap_err_dic[j]    = Trap_err_v
    Daitche_err_dic[j] = Daitche_err_v
    Prasath_err_dic[j] = Prasath_err_v
    IMEX2_err_dic[j]   = IMEX2_err_v
    IMEX4_err_dic[j]   = IMEX4_err_v
    DIRK_err_dic[j]    = DIRK_err_v

    ConvTrap[j]    = str(round(np.polyfit(np.log(L_v), np.log(Trap_err_v),   1)[0], 2))
    ConvDIRK4[j]   = str(round(np.polyfit(np.log(L_v), np.log(DIRK_err_v),   1)[0], 2))
    ConvIMEX2[j]   = str(round(np.polyfit(np.log(L_v), np.log(IMEX2_err_v),  1)[0], 2))
    ConvIMEX4[j]   = str(round(np.polyfit(np.log(L_v), np.log(IMEX4_err_v),  1)[0], 2))
    ConvDaitche[j] = str(round(np.polyfit(np.log(L_v), np.log(Daitche_err_v),1)[0], 2))
    ConvPrasath[j] = str(round(np.polyfit(np.log(L_v), np.log(Prasath_err_v),1)[0], 2))




###############################################################################
################# Define field and implementation variables ###################
###############################################################################
#
save_plot_to = './VISUAL_OUTPUT/'
y0           = np.array([1., 0.])         # Initial position
v0           = np.array([0., 1.])         # Initial velocity
tini         = 0.                         # Initial time
tend         = 1.                         # Final time
omega_value  = 1.                         # Angular velocity of the field 
vel          = velocity_field_Analytical(omega=omega_value) # Flow field

# Define vector of time nodes
L_v          = np.array([ 26, 51, 101, 251, 501, 1001]) #, 2501])

#
###############################################################################
######################### Define particle variables ###########################
###############################################################################
#
# Define parameters to obtain R (Remember R=(1+2*rho_p/rho_f)/3):
rho_p       = np.array([2.0, 1.0, 3.0])   # - Particle's density
rho_f       = np.array([3.0, 1.0, 2.0])   # - Fluid's density

# Define parameters to obtain S (Remember S=rad_p**2/(3*nu_f*t_scale)):
rad_p        = 1.           # Particle's radius
nu_f         = 1.           # Kinematic viscocity
t_scale      = 10.          # Time Scale of the flow

#
###############################################################################
################# Define parameters for the numerical schemes #################
###############################################################################
#
# Control constant (Koleva 2005)
c           = 20

# Define the order for Daitche's method
order_Daitche = 3

# Define number of time Chebyshev nodes in each time interval
nodes_dt    = 21 # Advised by Prasath et al. (2019)

#
#########################################
# Start loop over the densities vectors #
#########################################
#
Trap_err_dic    = dict()
Daitche_err_dic = dict()
Prasath_err_dic = dict()
IMEX2_err_dic   = dict()
IMEX4_err_dic   = dict()
DIRK_err_dic    = dict()
ConvTrap        = dict()
ConvDIRK4       = dict()
ConvIMEX2       = dict()
ConvIMEX4       = dict()
ConvDaitche     = dict()
ConvPrasath     = dict()
assert len(rho_p) == len(rho_f), "Length of density vectors do not match."
for j in range(0, len(rho_p)):
    count         = 0
    
    print("-> Starting calculating values of the plot Nº" + str(j+1))
    for ll in range(0, len(L_v)):
        taxis     = np.linspace(tini, tend, L_v[ll]) # Time axis
        dt        = taxis[1] - taxis[0]        # time step
        count    += 1
    
        # t0        = time.time()
        
        #
        ################################
        # Calculate reference solution #
        ################################
        #
        Analy_particle  = maxey_riley_analytic(1, y0, v0, tini, vel,
                                               particle_density    = rho_p[j],
                                               fluid_density       = rho_f[j],
                                               particle_radius     = rad_p,
                                               kinematic_viscosity = nu_f,
                                               time_scale          = t_scale)
        
        #
        #######################################################################
        ############# Define parameters for the numerical schemes #############
        #######################################################################
        #
        # Define Uniform grid [0,1]:
        N           = np.copy(L_v[ll])  # Number of nodes
        xi_fd_v     = np.linspace(0., 1., int(N))[:-1]

        # Logarithm map to obtain QUM
        x_fd_v      = -c * np.log(1.0 - xi_fd_v)
    
        Trap_particle   = maxey_riley_trapezoidal(1, y0, v0, vel,
                                                  x_fd_v, c, dt, tini,
                                                  particle_density    = rho_p[j],
                                                  fluid_density       = rho_f[j],
                                                  particle_radius     = rad_p,
                                                  kinematic_viscosity = nu_f,
                                                  time_scale          = t_scale)
    
        IMEX2_particle   = maxey_riley_imex(1, y0, v0, vel,
                                            x_fd_v, c, dt, tini,
                                            particle_density    = rho_p[j],
                                            fluid_density       = rho_f[j],
                                            particle_radius     = rad_p,
                                            kinematic_viscosity = nu_f,
                                            time_scale          = t_scale,
                                            IMEXOrder           = 2,
                                            FDOrder             = 2,
                                            parallel_flag       = False)
    
        IMEX4_particle   = maxey_riley_imex(1, y0, v0, vel,
                                            x_fd_v, c, dt, tini,
                                            particle_density    = rho_p[j],
                                            fluid_density       = rho_f[j],
                                            particle_radius     = rad_p,
                                            kinematic_viscosity = nu_f,
                                            time_scale          = t_scale,
                                            IMEXOrder           = 4,
                                            FDOrder             = 4,
                                            parallel_flag       = False)

        DIRK_particle   = maxey_riley_dirk(1, y0, v0, vel,
                                           x_fd_v, c, dt, tini,
                                           particle_density    = rho_p[j],
                                           fluid_density       = rho_f[j],
                                           particle_radius     = rad_p,
                                           kinematic_viscosity = nu_f,
                                           time_scale          = t_scale,
                                           parallel_flag       = False)

        Daitche_particle = maxey_riley_Daitche(1, y0, v0, vel, L_v[ll],
                                               order_Daitche,
                                               particle_density    = rho_p[j],
                                               fluid_density       = rho_f[j],
                                               particle_radius     = rad_p,
                                               kinematic_viscosity = nu_f,
                                               time_scale          = t_scale)
    
        Prasath_particle  = maxey_riley_Prasath(1, y0, v0, vel, N, tini, dt,
                                                nodes_dt,
                                                particle_density    = rho_p[j],
                                                fluid_density       = rho_f[j],
                                                particle_radius     = rad_p,
                                                kinematic_viscosity = nu_f,
                                                time_scale          = t_scale)
        
number_cores  = int(sys.argv[1]) if (len(sys.argv) > 1) and (int(sys.argv[1]) < mp.cpu_count()) else int(mp.cpu_count())
if __name__ == '__main__':
    t0 = time.time()
    p = mp.Pool(number_cores)
    result  = [p.apply_async(compute_particles, args=(particle, taxis),
                             callback=save_hdf5,
                             error_callback=lambda x: print("Error")) \
               for particle in progressbar(imex_particle_set)]
    
    p.close()
    p.join()
    tf = time.time()
    print("Calculation took " + str(tf - t0) + " seconds.")

#
###############################################################################
##################### Create Table with convergence orders ####################
###############################################################################
#
# Create convergence table
mydata = [
         ["Prasath et al.",
          ConvPrasath[0], ConvPrasath[1], ConvPrasath[2]],
         ["FD2 + Trap.",
          ConvTrap[0],    ConvTrap[1],    ConvTrap[2]],
         ["FD2 + IMEX2",
          ConvIMEX2[0],   ConvIMEX2[1],   ConvIMEX2[2]],
         ["Datiche, " + order_Daitche + " order",
          ConvDaitche[0], ConvDaitche[1], ConvDaitche[2]],
         ["FD2 + IMEX4",
          ConvIMEX4[0],   ConvIMEX4[1],   ConvIMEX4[2]],
         ["FD2 + DIRK4",
          ConvDIRK4[0],   ConvDIRK4[1],   ConvDIRK4[2]],
         ]

# create header
head = ["R:", str(round((1.+2.*(rho_p[0]/rho_f[0]))/3., 2)),
              str(round((1.+2.*(rho_p[1]/rho_f[1]))/3., 2)),
              str(round((1.+2.*(rho_p[2]/rho_f[2]))/3., 2))]

print(tabulate(mydata, headers=head, tablefmt="grid"))

#
###############################################################################
##### Plot plots in figure with Particle's trajectories on velocity field #####
###############################################################################
#
#############
# Left plot #
#############
fig1 = plt.figure(1, layout='tight')
fig1.set_figwidth(4.2)
fig1.set_figheight(3.6)
fs = 13
lw = 2.2

# plt.plot(L_v[2:4], 1.*L_v[2:4]**(-2.0), '--', color='grey',                              linewidth=1.0)
# plt.text(145, 2.e-4, "$N^{-2}$", color='grey')
plt.plot(L_v[2:4], 1.5*L_v[2:4]**(-2.0), '--', color='grey', linewidth=1.0)
plt.text(135, 1.7e-4, "$N^{-2}$", color='grey')
# plt.plot(L_v[2:4], 1e-1*L_v[2:4]**(-3.0), '--', color='grey',                              linewidth=1.0)
# plt.text(145, 1e-9, "$N^{-3}$", color='grey')
plt.plot(L_v[2:4], 5*L_v[2:4]**(-3.0), '--', color='grey', linewidth=1.0)
plt.text(135, 3e-6, "$N^{-3}$", color='grey')
# plt.plot(L_v[2:4], 5e-3 * L_v[2:4]**(-4.0), '--', color='grey',                              linewidth=1.0)
# plt.text(145, 2e-11, "$N^{-4}$", color='grey')
plt.plot(L_v[2:4], 30 * L_v[2:4]**(-4.0), '--', color='grey', linewidth=1.0)
plt.text(125, 1e-8, "$N^{-4}$", color='grey')

plt.plot(L_v, Prasath_err_dic[0], 'P-', color='orange', label="Prasath et al. (2019)", linewidth=lw)
plt.plot(L_v, Trap_err_dic[0],    'v-', color='green',  label="FD2 + Trap. Rule",      linewidth=lw)
plt.plot(L_v, IMEX2_err_dic[0],   's-', color='violet', label="FD2 + IMEX2",           linewidth=lw)
plt.plot(L_v, Daitche_err_dic[0], 'd-', color='cyan',   label="Daitche, " + str(order_Daitche) + " order", linewidth=lw)
plt.plot(L_v, IMEX4_err_dic[0],   'o-', color='red',    label="FD4 + IMEX4",           linewidth=lw)
plt.plot(L_v, DIRK_err_dic[0],    '+-', color='blue',   label="FD4 + DIRK4",           linewidth=lw)

plt.tick_params(axis='both', labelsize=fs)
plt.xscale("log")
plt.yscale("log")
plt.ylim(1e-17,1e-2)
plt.xlabel('Nodes, N', fontsize=fs, labelpad=0.25)
plt.ylabel('$||\; error\; ||_{\infty}$', fontsize=fs, labelpad=0.25)
plt.legend(loc='lower left', fontsize=6)
plt.grid()

# plt.tight_layout()

plt.savefig(save_plot_to + 'Figure06a.pdf', format='pdf', dpi=500)

#
###############
# Center plot #
###############
#
fig1 = plt.figure(2, layout='tight')
fig1.set_figwidth(4.2)
fig1.set_figheight(3.6)
fs = 13
lw = 2.2

# plt.plot(L_v[2:4], 1.*L_v[2:4]**(-2.0), '--', color='grey',                              linewidth=1.0)
# plt.text(145, 2.e-4, "$N^{-2}$", color='grey')
plt.plot(L_v[2:4], 1.5*L_v[2:4]**(-2.0), '--', color='grey', linewidth=1.0)
plt.text(135, 1.7e-4, "$N^{-2}$", color='grey')
# plt.plot(L_v[2:4], 1e-1*L_v[2:4]**(-3.0), '--', color='grey',                              linewidth=1.0)
# plt.text(145, 1e-9, "$N^{-3}$", color='grey')
plt.plot(L_v[2:4], 5*L_v[2:4]**(-3.0), '--', color='grey', linewidth=1.0)
plt.text(135, 3e-6, "$N^{-3}$", color='grey')
# plt.plot(L_v[2:4], 5e-3 * L_v[2:4]**(-4.0), '--', color='grey',                              linewidth=1.0)
# plt.text(145, 2e-11, "$N^{-4}$", color='grey')
plt.plot(L_v[2:4], 30 * L_v[2:4]**(-4.0), '--', color='grey', linewidth=1.0)
plt.text(125, 1e-8, "$N^{-4}$", color='grey')

plt.plot(L_v, Prasath_err_dic[1], 'P-', color='orange', label="Prasath et al. (2019)", linewidth=lw)
plt.plot(L_v, Trap_err_dic[1],    'v-', color='green',  label="FD2 + Trap. Rule",      linewidth=lw)
plt.plot(L_v, IMEX2_err_dic[1],   's-', color='violet', label="FD2 + IMEX2",           linewidth=lw)
plt.plot(L_v, Daitche_err_dic[1], 'd-', color='cyan',   label="Daitche, " + str(order_Daitche) + " order", linewidth=lw)
plt.plot(L_v, IMEX4_err_dic[1],   'o-', color='red',    label="FD4 + IMEX4",           linewidth=lw)
plt.plot(L_v, DIRK_err_dic[1],    '+-', color='blue',   label="FD4 + DIRK4",           linewidth=lw)

plt.tick_params(axis='both', labelsize=fs)
plt.xscale("log")
plt.yscale("log")
plt.ylim(1e-17,1e-2)
plt.xlabel('Nodes, N', fontsize=fs, labelpad=0.25)
plt.ylabel('$||\; error\; ||_{\infty}$', fontsize=fs, labelpad=0.25)
plt.legend(loc='lower left', fontsize=6)
plt.grid()

# plt.tight_layout()

plt.savefig(save_plot_to + 'Figure06b.pdf', format='pdf', dpi=500)

#
##############
# Right plot #
##############
#
fig1 = plt.figure(2, layout='tight')
fig1.set_figwidth(4.2)
fig1.set_figheight(3.6)
fs = 13
lw = 2.2

# plt.plot(L_v[2:4], 1.*L_v[2:4]**(-2.0), '--', color='grey',                              linewidth=1.0)
# plt.text(145, 2.e-4, "$N^{-2}$", color='grey')
plt.plot(L_v[2:4], 1.5*L_v[2:4]**(-2.0), '--', color='grey', linewidth=1.0)
plt.text(135, 1.7e-4, "$N^{-2}$", color='grey')
# plt.plot(L_v[2:4], 1e-1*L_v[2:4]**(-3.0), '--', color='grey',                              linewidth=1.0)
# plt.text(145, 1e-9, "$N^{-3}$", color='grey')
plt.plot(L_v[2:4], 5*L_v[2:4]**(-3.0), '--', color='grey', linewidth=1.0)
plt.text(135, 3e-6, "$N^{-3}$", color='grey')
# plt.plot(L_v[2:4], 5e-3 * L_v[2:4]**(-4.0), '--', color='grey',                              linewidth=1.0)
# plt.text(145, 2e-11, "$N^{-4}$", color='grey')
plt.plot(L_v[2:4], 30 * L_v[2:4]**(-4.0), '--', color='grey', linewidth=1.0)
plt.text(125, 1e-8, "$N^{-4}$", color='grey')

plt.plot(L_v, Prasath_err_dic[2], 'P-', color='orange', label="Prasath et al. (2019)", linewidth=lw)
plt.plot(L_v, Trap_err_dic[2],    'v-', color='green',  label="FD2 + Trap. Rule",      linewidth=lw)
plt.plot(L_v, IMEX2_err_dic[2],   's-', color='violet', label="FD2 + IMEX2",           linewidth=lw)
plt.plot(L_v, Daitche_err_dic[2], 'd-', color='cyan',   label="Daitche, " + str(order_Daitche) + " order", linewidth=lw)
plt.plot(L_v, IMEX4_err_dic[2],   'o-', color='red',    label="FD4 + IMEX4",           linewidth=lw)
plt.plot(L_v, DIRK_err_dic[2],    '+-', color='blue',   label="FD4 + DIRK4",           linewidth=lw)

plt.tick_params(axis='both', labelsize=fs)
plt.xscale("log")
plt.yscale("log")
plt.ylim(1e-17,1e-2)
plt.xlabel('Nodes, N', fontsize=fs, labelpad=0.25)
plt.ylabel('$||\; error\; ||_{\infty}$', fontsize=fs, labelpad=0.25)
plt.legend(loc='lower left', fontsize=6)
plt.grid()

# plt.tight_layout()

plt.savefig(save_plot_to + 'Figure06c.pdf', format='pdf', dpi=500)

plt.show()

print("\007")