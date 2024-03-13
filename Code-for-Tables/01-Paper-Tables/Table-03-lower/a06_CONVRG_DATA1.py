#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import time
import matplotlib.pyplot as plt
from tabulate import tabulate
from a03_FIELD0_DATA1 import velocity_field_Faraday1
from a09_PRTCLE_IMEX4 import maxey_riley_imex
from a09_PRTCLE_DIRK4 import maxey_riley_dirk
from a09_PRTCLE_TRAPZ import maxey_riley_trapezoidal
from a09_PRTCLE_PRSTH import maxey_riley_Prasath
from a09_PRTCLE_DTCHE import maxey_riley_Daitche

"""
Created on Mon Feb 05 16:31:09 2024
@author: Julio Urizarna-Carasa
"""

#
###############################################################################
################# Define field and implementation variables ###################
###############################################################################
#
save_plot_to = './OUTPUT/'                # Folder where to save output
tini         = 0.                         # Initial time
tend         = 1.                         # Final time
vel          = velocity_field_Faraday1()  # Flow field
y0           = np.array([0.02,  0.01])    # Initial position
v0           = vel.get_velocity(y0[0], y0[1], tini) # Initial velocity


# Define vector of time nodes (The last node is used for the calculation of the reference solution)
L_v          = np.array([ 26, 51, 101, 251, 501, 651])


#
###############################################################################
######################### Define particle variables ###########################
###############################################################################
#
# Define parameters to obtain R (Remember R=(1+2*rho_p/rho_f)/3):
rho_p       = np.array([0., 2., 1., 3., 3.])   # - Particle's density
rho_f       = np.array([1., 3., 1., 2., 1.])   # - Fluid's density

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
# Warning
print("\nRunning this script could take over four hours. \n")

# Start loop
Trap_err_dic    = dict()
Daitche_err_dic = dict()
Prasath_err_dic = dict()
IMEX2_err_dic   = dict()
IMEX4_err_dic   = dict()
DIRK4_err_dic   = dict()
ConvTrap        = dict()
ConvDIRK4       = dict()
ConvIMEX2       = dict()
ConvIMEX4       = dict()
ConvDaitche     = dict()
ConvPrasath     = dict()
assert len(rho_p) == len(rho_f), "Length of density vectors do not match."
for j in range(0, len(rho_p)):
    count         = 0
    
    Trap_err_v    = np.array([])
    Daitche_err_v = np.array([])
    Prasath_err_v = np.array([])
    IMEX2_err_v   = np.array([])
    IMEX4_err_v   = np.array([])
    DIRK4_err_v   = np.array([])
    
    #
    ###########################################################################
    ######################## Calculate reference solution #####################
    ###########################################################################
    #
    print("-> Starting calculating values of the plot Nº" + str(j+1) + "\n")
    print("    *Calculating reference solution with Prasath et al.\n")
    N            = np.copy(L_v[-1])  # Number of nodes
    taxis        = np.linspace(tini, tend, L_v[-1]) # Time axis
    dt           = taxis[1] - taxis[0]        # time step
    Reference_particle  = maxey_riley_Prasath(1, y0, v0, vel, N, tini, dt,
                                           nodes_dt,
                                           particle_density    = rho_p[j],
                                           fluid_density       = rho_f[j],
                                           particle_radius     = rad_p,
                                           kinematic_viscosity = nu_f,
                                           time_scale          = t_scale)
    
    Reference_pos      = np.array([y0])
    for tt in range(1, len(taxis)):        
        Reference_particle.update()
        Reference_pos = np.vstack((Reference_pos,
                                   Reference_particle.pos_vec[tt * (nodes_dt-1)]))
    
    print("\n    *Starting the calculation of the numerical solutions.\n")
    for ll in range(0, len(L_v[:-1])):
        taxis     = np.linspace(tini, tend, L_v[ll]) # Time axis
        dt        = taxis[1] - taxis[0]        # time step
        count    += 1
        
        t0        = time.time()
        
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
    
        Trap_particle    = maxey_riley_trapezoidal(1, y0, v0, vel,
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

        DIRK4_particle   = maxey_riley_dirk(1, y0, v0, vel,
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
    
        Prasath_particle = maxey_riley_Prasath(1, y0, v0, vel, N, tini, dt,
                                               nodes_dt,
                                               particle_density    = rho_p[j],
                                               fluid_density       = rho_f[j],
                                               particle_radius     = rad_p,
                                               kinematic_viscosity = nu_f,
                                               time_scale          = t_scale)
        
        Prasath_pos      = np.array([y0])
        for tt in range(1, len(taxis)):        
            try:
                Prasath_particle.update()
                Prasath_pos = np.vstack((Prasath_pos,
                                         Prasath_particle.pos_vec[tt * (nodes_dt-1)]))
            except:
                None
            
            try:
                Trap_particle.update()
            except:
                None
            
            IMEX2_particle.update()
            IMEX4_particle.update()
            DIRK4_particle.update()
        
        try:
            Prasath_err     = Prasath_pos[-1] - Reference_pos[-1]
            Prasath_err_max = np.linalg.norm(Prasath_err, ord=2, axis=0)
            Prasath_err_v   = np.append(Prasath_err_v, Prasath_err_max)
        except:
            None

        Trap_err      = Trap_particle.pos_vec[-1] - Reference_pos[-1]
        Trap_err_max  = np.linalg.norm(Trap_err, ord=2, axis=0)
        Trap_err_v    = np.append(Trap_err_v, Trap_err_max)

        IMEX2_err     = IMEX2_particle.pos_vec[-1] - Reference_pos[-1]
        IMEX2_err_max = np.linalg.norm(IMEX2_err, ord=2, axis=0)
        IMEX2_err_v   = np.append(IMEX2_err_v, IMEX2_err_max)

        IMEX4_err     = IMEX4_particle.pos_vec[-1] - Reference_pos[-1]
        IMEX4_err_max = np.linalg.norm(IMEX4_err, ord=2, axis=0)
        IMEX4_err_v   = np.append(IMEX4_err_v, IMEX4_err_max)

        DIRK4_err     = DIRK4_particle.pos_vec[-1] - Reference_pos[-1]
        DIRK4_err_max = np.linalg.norm(DIRK4_err, ord=2, axis=0)
        DIRK4_err_v   = np.append(DIRK4_err_v, DIRK4_err_max)

        if order_Daitche == 1:
            Daitche_particle.Euler(taxis, flag=True)  # First order method
        elif order_Daitche == 2:
            Daitche_particle.AdamBashf2(taxis, flag=True) # Second order method
        elif order_Daitche == 3:
            Daitche_particle.AdamBashf3(taxis, flag=True) # Third order method

        Daitche_err     = Daitche_particle.pos_vec[-1] - Reference_pos[-1]
        Daitche_err_max = np.linalg.norm(Daitche_err, ord=2, axis=0)
        Daitche_err_v   = np.append(Daitche_err_v, Daitche_err_max)
        
        tf              = time.time()
        
        print("\n   Round number " + str(count) + " finished in " + str(round(tf - t0, 2)) + " seconds.\n")
        
    Trap_err_dic[j]    = Trap_err_v
    Daitche_err_dic[j] = Daitche_err_v
    Prasath_err_dic[j] = Prasath_err_v
    IMEX2_err_dic[j]   = IMEX2_err_v
    IMEX4_err_dic[j]   = IMEX4_err_v
    DIRK4_err_dic[j]   = DIRK4_err_v

    ConvTrap[j]    = str(round(np.polyfit(np.log(L_v[:-1]), np.log(Trap_err_v),   1)[0], 2))
    ConvDIRK4[j]   = str(round(np.polyfit(np.log(L_v[:-1]), np.log(DIRK4_err_v),  1)[0], 2))
    ConvIMEX2[j]   = str(round(np.polyfit(np.log(L_v[:-1]), np.log(IMEX2_err_v),  1)[0], 2))
    ConvIMEX4[j]   = str(round(np.polyfit(np.log(L_v[:-1]), np.log(IMEX4_err_v),  1)[0], 2))
    ConvDaitche[j] = str(round(np.polyfit(np.log(L_v[:-1]), np.log(Daitche_err_v),1)[0], 2))
    ConvPrasath[j] = str(round(np.polyfit(np.log(L_v[:-1]), np.log(Prasath_err_v),1)[0], 2))


#
###############################################################################
################################## Save data ##################################
###############################################################################
#
for j in range(0, len(rho_p)):
    with open(save_plot_to + 'Data_0'  + str(j+1) +'.txt', 'w') as file:
        file.write( "Parameters:\n" )
        file.write( " - t_0: " + str(tini) + "\n" )
        file.write( " - t_f: " + str(tend) + "\n" )
        file.write( " - y_0: " + str(y0) + "\n" )
        file.write( " - v_0: " + str(v0) + "\n" )
        file.write( " - R: " + str((1.+ 2.*rho_p[j]/rho_f[j]) /3.) + "\n" )
        file.write( " - S: " + str(rad_p**2./(3.*nu_f*t_scale)) + "\n\n" )
        file.write( "Nodes: " + str(L_v) + "\n\n" )
        file.write( "Errors:\n")
        file.write( " - Prasath: " + str(Prasath_err_dic[j]) + "\n" )
        file.write( " - Trap. Rule: " + str(Trap_err_dic[j]) + "\n" )
        file.write( " - IMEX2: " + str(IMEX2_err_dic[j]) + "\n" )
        file.write( " - Daitche: " + str(Daitche_err_dic[j]) + "\n" )
        file.write( " - IMEX4: " + str(IMEX4_err_dic[j]) + "\n" )
        file.write( " - DIKR4: " + str(DIRK4_err_dic[j]) + "\n" )


#
###############################################################################
##################### Create Table with convergence orders ####################
###############################################################################
#
# Create convergence table
mydata = [
         ["Prasath et al.:", ConvPrasath[0], ConvPrasath[1],
          ConvPrasath[2],    ConvPrasath[3], ConvPrasath[4]],
         ["FD2 + Trap.:",    ConvTrap[0],    ConvTrap[1],
          ConvTrap[2],       ConvTrap[3],    ConvTrap[4]],
         ["FD2 + IMEX2:",    ConvIMEX2[0],   ConvIMEX2[1],
          ConvIMEX2[2],      ConvIMEX2[3],   ConvIMEX2[4]],
         ["Datiche, " + str(order_Daitche) + " order:",
          ConvDaitche[0], ConvDaitche[1], ConvDaitche[2], ConvDaitche[3], ConvDaitche[4]],
         ["FD2 + IMEX4:",    ConvIMEX4[0],   ConvIMEX4[1],
          ConvIMEX4[2],      ConvIMEX4[3],   ConvIMEX4[4]],
         ["FD2 + DIRK4:",    ConvDIRK4[0],   ConvDIRK4[1],
          ConvDIRK4[2],      ConvDIRK4[3],   ConvDIRK4[4]],
         ]

# create header
head = ["R:", str(round((1.+2.*(rho_p[0]/rho_f[0]))/3., 2)),
              str(round((1.+2.*(rho_p[1]/rho_f[1]))/3., 2)),
              str(round((1.+2.*(rho_p[2]/rho_f[2]))/3., 2)),
              str(round((1.+2.*(rho_p[1]/rho_f[3]))/3., 2)),
              str(round((1.+2.*(rho_p[1]/rho_f[4]))/3., 2))]

print("\nConvergence rates")
print("\n" + tabulate(mydata, headers=head, tablefmt="grid"))

with open(save_plot_to + 'Convergence_rates.txt', 'w') as file:
    file.write("Convergence rates\n")
    file.write( str(tabulate(mydata, headers=head, tablefmt="grid") ))


''' # Uncomment if interested in getting the plots.

#
###############################################################################
##### Plot plots in figure with Particle's trajectories on velocity field #####
###############################################################################
#
############
# Plot Nº1 #
############
fig1 = plt.figure(1, layout='tight')
fig1.set_figwidth(4.2)
fig1.set_figheight(3.6)
fs = 13
lw = 1.5

# plt.plot(L_v[2:4], 1.5*L_v[2:4]**(-2.0), '--', color='grey', linewidth=1.0)
# plt.text(145, 1.3e-4, "$N^{-2}$", color='grey')
# plt.plot(L_v[2:4], 5*L_v[2:4]**(-3.0), '--', color='grey', linewidth=1.0)
# plt.text(190, 8e-7, "$N^{-3}$", color='grey')
# plt.plot(L_v[2:4], 20 * L_v[2:4]**(-4.0), '--', color='grey', linewidth=1.0)
# plt.text(155, 1e-9, "$N^{-4}$", color='grey')

plt.plot(L_v[:-1], Prasath_err_dic[0], 'P-', color='orange', label="Prasath et al. (2019)", linewidth=lw)
plt.plot(L_v[:-1], Trap_err_dic[0],    'v-', color='green',  label="FD2 + Trap. Rule",      linewidth=lw)
plt.plot(L_v[:-1], IMEX2_err_dic[0],   's-', color='violet', label="FD2 + IMEX2",           linewidth=lw)
plt.plot(L_v[:-1], Daitche_err_dic[0], 'd-', color='cyan',   label="Daitche, " + str(order_Daitche) + " order", linewidth=lw)
plt.plot(L_v[:-1], IMEX4_err_dic[0],   'o-', color='red',    label="FD4 + IMEX4",           linewidth=lw)
plt.plot(L_v[:-1], DIRK4_err_dic[0],   '+-', color='blue',   label="FD4 + DIRK4",           linewidth=lw)

plt.tick_params(axis='both', labelsize=fs)
plt.xscale("log")
plt.yscale("log")
plt.ylim(1e-10,1e-1)
plt.xlabel('Nodes, N', fontsize=fs, labelpad=0.25)
plt.ylabel('$||\; error\; ||_2$', fontsize=fs, labelpad=0.25)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.21), ncol=3, fontsize=6)
plt.grid()

plt.savefig(save_plot_to + 'Figure_01.pdf', format='pdf', dpi=500)


############
# Plot Nº2 #
############
fig1 = plt.figure(2, layout='tight')
fig1.set_figwidth(4.2)
fig1.set_figheight(3.6)
fs = 13
lw = 1.5

# plt.plot(L_v[2:4], 1.5*L_v[2:4]**(-2.0), '--', color='grey', linewidth=1.0)
# plt.text(145, 1.3e-4, "$N^{-2}$", color='grey')
# plt.plot(L_v[2:4], 5*L_v[2:4]**(-3.0), '--', color='grey', linewidth=1.0)
# plt.text(190, 8e-7, "$N^{-3}$", color='grey')
# plt.plot(L_v[2:4], 20 * L_v[2:4]**(-4.0), '--', color='grey', linewidth=1.0)
# plt.text(155, 1e-9, "$N^{-4}$", color='grey')

plt.plot(L_v[:-1], Prasath_err_dic[1], 'P-', color='orange', label="Prasath et al. (2019)", linewidth=lw)
plt.plot(L_v[:-1], Trap_err_dic[1],    'v-', color='green',  label="FD2 + Trap. Rule",      linewidth=lw)
plt.plot(L_v[:-1], IMEX2_err_dic[1],   's-', color='violet', label="FD2 + IMEX2",           linewidth=lw)
plt.plot(L_v[:-1], Daitche_err_dic[1], 'd-', color='cyan',   label="Daitche, " + str(order_Daitche) + " order", linewidth=lw)
plt.plot(L_v[:-1], IMEX4_err_dic[1],   'o-', color='red',    label="FD4 + IMEX4",           linewidth=lw)
plt.plot(L_v[:-1], DIRK4_err_dic[1],   '+-', color='blue',   label="FD4 + DIRK4",           linewidth=lw)

plt.tick_params(axis='both', labelsize=fs)
plt.xscale("log")
plt.yscale("log")
plt.ylim(1e-10,1e-1)
plt.xlabel('Nodes, N', fontsize=fs, labelpad=0.25)
plt.ylabel('$||\; error\; ||_2$', fontsize=fs, labelpad=0.25)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.21), ncol=3, fontsize=6)
plt.grid()

plt.savefig(save_plot_to + 'Figure_02.pdf', format='pdf', dpi=500)

#
############
# Plot Nº3 #
############
#
fig2 = plt.figure(3, layout='tight')
fig2.set_figwidth(4.2)
fig2.set_figheight(3.6)

# plt.plot(L_v[2:4], 1.*L_v[2:4]**(-2.0), '--', color='grey',                              linewidth=1.0)
# plt.text(135, 1.e-4, "$N^{-2}$", color='grey')
# plt.plot(L_v[2:4], 5e-2*L_v[2:4]**(-3.0), '--', color='grey',                              linewidth=1.0)
# plt.text(150, 1e-9, "$N^{-3}$", color='grey')
# plt.plot(L_v[2:4], 8e-3 * L_v[2:4]**(-4.0), '--', color='grey',                              linewidth=1.0)
# plt.text(160, 2e-11, "$N^{-4}$", color='grey')

plt.plot(L_v[:-1], Prasath_err_dic[2], 'P-', color='orange', label="Prasath et al. (2019)", linewidth=lw)
plt.plot(L_v[:-1], Trap_err_dic[2],    'v-', color='green',  label="FD2 + Trap. Rule",      linewidth=lw)
plt.plot(L_v[:-1], IMEX2_err_dic[2],   's-', color='violet', label="FD2 + IMEX2",           linewidth=lw)
plt.plot(L_v[:-1], Daitche_err_dic[2], 'd-', color='cyan',   label="Daitche, " + str(order_Daitche) + " order", linewidth=lw)
plt.plot(L_v[:-1], IMEX4_err_dic[2],   'o-', color='red',    label="FD4 + IMEX4",           linewidth=lw)
plt.plot(L_v[:-1], DIRK4_err_dic[2],   '+-', color='blue',   label="FD4 + DIRK4",           linewidth=lw)

plt.tick_params(axis='both', labelsize=fs)
plt.xscale("log")
plt.yscale("log")
plt.ylim(1e-10,1e-1)
plt.xlabel('Nodes, N', fontsize=fs, labelpad=0.25)
plt.ylabel('$||\; error\; ||_2$', fontsize=fs, labelpad=0.25)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.21), ncol=3, fontsize=6)
plt.grid()

plt.savefig(save_plot_to + 'Figure_03.pdf', format='pdf', dpi=500)

#
############
# Plot Nº4 #
############
#
fig3 = plt.figure(4, layout='tight')
fig3.set_figwidth(4.2)
fig3.set_figheight(3.6)

# plt.plot(L_v[2:4], 1.5*L_v[2:4]**(-2.0), '--', color='grey', linewidth=1.0)
# plt.text(145, 1.5e-4, "$N^{-2}$", color='grey')
# plt.plot(L_v[2:4], 5*L_v[2:4]**(-3.0), '--', color='grey', linewidth=1.0)
# plt.text(175, 1e-6, "$N^{-3}$", color='grey')
# plt.plot(L_v[2:4], 15 * L_v[2:4]**(-4.0), '--', color='grey', linewidth=1.0)
# plt.text(145, 1e-9, "$N^{-4}$", color='grey')

plt.plot(L_v[:-1], Prasath_err_dic[3], 'P-', color='orange', label="Prasath et al. (2019)", linewidth=lw)
plt.plot(L_v[:-1], Trap_err_dic[3],    'v-', color='green',  label="FD2 + Trap. Rule",      linewidth=lw)
plt.plot(L_v[:-1], IMEX2_err_dic[3],   's-', color='violet', label="FD2 + IMEX2",           linewidth=lw)
plt.plot(L_v[:-1], Daitche_err_dic[3], 'd-', color='cyan',   label="Daitche, " + str(order_Daitche) + " order", linewidth=lw)
plt.plot(L_v[:-1], IMEX4_err_dic[3],   'o-', color='red',    label="FD4 + IMEX4",           linewidth=lw)
plt.plot(L_v[:-1], DIRK4_err_dic[3],   '+-', color='blue',   label="FD4 + DIRK4",           linewidth=lw)

plt.tick_params(axis='both', labelsize=fs)
plt.xscale("log")
plt.yscale("log")
plt.ylim(1e-10,1e-1)
plt.xlabel('Nodes, N', fontsize=fs, labelpad=0.25)
plt.ylabel('$||\; error\; ||_2$', fontsize=fs, labelpad=0.25)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.21), ncol=3, fontsize=6)
plt.grid()

plt.savefig(save_plot_to + 'Figure_04.pdf', format='pdf', dpi=500)

#
############
# Plot Nº5 #
############
#
fig3 = plt.figure(5, layout='tight')
fig3.set_figwidth(4.2)
fig3.set_figheight(3.6)

# plt.plot(L_v[2:4], 1.5*L_v[2:4]**(-2.0), '--', color='grey', linewidth=1.0)
# plt.text(145, 1.5e-4, "$N^{-2}$", color='grey')
# plt.plot(L_v[2:4], 5*L_v[2:4]**(-3.0), '--', color='grey', linewidth=1.0)
# plt.text(175, 1e-6, "$N^{-3}$", color='grey')
# plt.plot(L_v[2:4], 15 * L_v[2:4]**(-4.0), '--', color='grey', linewidth=1.0)
# plt.text(145, 1e-9, "$N^{-4}$", color='grey')

plt.plot(L_v[:-1], Prasath_err_dic[4], 'P-', color='orange', label="Prasath et al. (2019)", linewidth=lw)
plt.plot(L_v[:-1], Trap_err_dic[4],    'v-', color='green',  label="FD2 + Trap. Rule",      linewidth=lw)
plt.plot(L_v[:-1], IMEX2_err_dic[4],   's-', color='violet', label="FD2 + IMEX2",           linewidth=lw)
plt.plot(L_v[:-1], Daitche_err_dic[4], 'd-', color='cyan',   label="Daitche, " + str(order_Daitche) + " order", linewidth=lw)
plt.plot(L_v[:-1], IMEX4_err_dic[4],   'o-', color='red',    label="FD4 + IMEX4",           linewidth=lw)
plt.plot(L_v[:-1], DIRK4_err_dic[4],   '+-', color='blue',   label="FD4 + DIRK4",           linewidth=lw)

plt.tick_params(axis='both', labelsize=fs)
plt.xscale("log")
plt.yscale("log")
plt.ylim(1e-10,1e-1)
plt.xlabel('Nodes, N', fontsize=fs, labelpad=0.25)
plt.ylabel('$||\; error\; ||_2$', fontsize=fs, labelpad=0.25)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.21), ncol=3, fontsize=6)
plt.grid()

plt.savefig(save_plot_to + 'Figure_05.pdf', format='pdf', dpi=500)

plt.show()

'''

print("\007")
