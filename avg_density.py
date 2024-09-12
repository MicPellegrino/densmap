"""
    Just produces an average density map for water-glycerol menisci
"""

import densmap as dm

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

from matplotlib import ticker

folder_name = 'WatGly40pEQ'
file_root = 'flow_'

# PARAMETERS TO TUNE
Lx = 159.75000
Lz = 30.63400

n_init = 100
n_fin = 2600

# CREATING MESHGRID
print("Creating meshgrid")
density_array = dm.read_density_file(folder_name+'/'+file_root+'{:05d}'.format(n_init)+'.dat', bin='y')
Nx = density_array.shape[0]
Nz = density_array.shape[1]
hx = Lx/Nx
hz = Lz/Nz
x = hx*np.arange(0.0,Nx,1.0, dtype=float)+0.5*hx
z = hz*np.arange(0.0,Nz,1.0, dtype=float)+0.5*hz

density_array = np.mean(density_array,axis=1)
xcom0 = np.sum(density_array*x)/np.sum(density_array)
icom0 = int(xcom0/hx)

# Section for density computation
# N_low = int(np.floor(40.0/hx))
# N_upp = int(np.ceil(120.0/hx))
N_win = int(30/hx)

X, Z = np.meshgrid(x[icom0-N_win:icom0+N_win+1], z, sparse=False, indexing='ij')

izloc = 10
Xloc, Zloc = np.meshgrid(x[icom0:icom0+N_win], z[0:izloc], sparse=False, indexing='ij')

dt = 12.5

n_dump = 10
print("Producing average density profiles")

# Density profile
density_field_sol = np.zeros( (2*N_win+1, Nz), dtype=float )
density_field_gol = np.zeros( (2*N_win+1, Nz), dtype=float )

xcom_vec = []
icom_vec = []

# Center of mass
for idx in range(n_init, n_fin+1 ):
        
    if idx%n_dump==0 :
        print("Obtainig frame "+str(idx))
        t_label = str(dt*idx)+' ps'
    density_array = dm.read_density_file(folder_name+'/'+file_root+ \
        '{:05d}'.format(idx)+'.dat', bin='y')
    density_array = np.mean(density_array,axis=1)
    xcom = np.sum(density_array*x)/np.sum(density_array)
    xcom_vec.append(xcom)
    icom = int(xcom/hx)
    icom_vec.append(icom)
    ishift = icom-icom0
    density_array_sol = dm.read_density_file(folder_name+'/'+file_root+'SOL_' \
        '{:05d}'.format(idx)+'.dat', bin='y')
    density_array_gol = dm.read_density_file(folder_name+'/'+file_root+'GOL_' \
        '{:05d}'.format(idx)+'.dat', bin='y')
    # density_field_sol += density_array_sol[N_low+ishift:N_upp+ishift,:]
    # density_field_gol += density_array_gol[N_low+ishift:N_upp+ishift,:]
    density_field_sol += density_array_sol[icom-N_win:icom+N_win+1,:]
    density_field_gol += density_array_gol[icom-N_win:icom+N_win+1,:]

density_field_sol /= (n_fin+1-n_init)
density_field_gol /= (n_fin+1-n_init)

density_field_sol_loc = 0.5*(density_field_sol[:,0:izloc]+np.fliplr(density_field_sol[:,-izloc:]))
density_field_gol_loc = 0.5*(density_field_gol[:,0:izloc]+np.fliplr(density_field_gol[:,-izloc:]))
density_field_sol_loc = 0.5*(density_field_sol_loc[-N_win:,:]+np.flipud(density_field_sol_loc[0:N_win,:]))
density_field_gol_loc = 0.5*(density_field_gol_loc[-N_win:,:]+np.flipud(density_field_gol_loc[0:N_win,:]))

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

dmap1 = ax1.pcolormesh(Xloc, Zloc, 1.66054*density_field_sol_loc, cmap=cm.Purples)
ax1.tick_params(axis='both', labelsize=25)
ax1.set_ylabel(r'$z$ [nm]', fontsize=30)
cb1 = plt.colorbar(dmap1,ax=ax1)
tick_locator1 = ticker.MaxNLocator(nbins=5)
cb1.locator = tick_locator1
cb1.update_ticks()
cb1.set_label(r'$\rho_w$ [kg/m$^3$]',fontsize=27.5, labelpad=20)
cb1.ax.tick_params(labelsize=27.5)

dmap2 = ax2.pcolormesh(Xloc, Zloc, 1.66054*density_field_gol_loc, cmap=cm.Greens)
ax2.tick_params(axis='both', labelsize=25)
ax2.set_ylabel(r'$z$ [nm]', fontsize=30)
ax2.set_xlabel(r'$x$ [nm]', fontsize=30)
cb2 = plt.colorbar(dmap2,ax=ax2)
tick_locator2 = ticker.MaxNLocator(nbins=5)
cb2.locator = tick_locator2
cb2.update_ticks()
cb2.set_label(r'$\rho_g$ [kg/m$^3$]',fontsize=27.5, labelpad=20)
cb2.ax.tick_params(labelsize=27.5)

# ax1.pcolormesh(X, Z, density_field_sol, cmap=cm.Purples)
# ax2.pcolormesh(X, Z, density_field_gol, cmap=cm.Greens)

plt.show()