# Let's make it a function callable from command line!

import numpy as np
import densmap as dm

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

folder_name = 'Ca010q65'
file_root = 'flow_SOL_'

# PARAMETERS TO TUNE
Lx = 83.70000
Lz = 22.93078

t_shoot = np.array([10000, 20000, 30000, 42230])

dt = 12.5
# dt = 10
n_shoot = np.array(t_shoot/dt, dtype=int)

# n_shoot = 3861

density_array = dm.read_density_file(folder_name+'/'+file_root+ \
            '{:05d}'.format(n_shoot[0])+'.dat', bin='y')

Nx = density_array.shape[0]
Nz = density_array.shape[1]
hx = Lx/Nx
hz = Lz/Nz
x = hx * ( 0.5+np.arange(0.0,Nx,1.0, dtype=float) )
z = hz * ( 0.5+np.arange(0.0,Nz,1.0, dtype=float) )
X, Z = np.meshgrid(x, z, sparse=False, indexing='ij')

# f, ((ax11,ax12),(ax21,ax22)) = plt.subplots(2,2,sharex=True,sharey=True)

"""

# PLOT ORIGINAL DENSITY
ax11.pcolormesh(X, Z, density_array, cmap=cm.Blues)
ax11.text(0, 35, '(a)', fontsize=25)
ax11.axis('scaled')
ax11.set_title('Density profile @'+str(t_shoot[0]/1000)+' ns', fontsize=25)
ax11.tick_params(axis="both", labelsize=25)
# ax11.set_xlabel('x [nm]', fontsize=20)
ax11.set_ylabel('z [nm]', fontsize=25)

density_array = dm.read_density_file(folder_name+'/'+file_root+ \
            '{:05d}'.format(n_shoot[1])+'.dat', bin='y')

ax12.pcolormesh(X, Z, density_array, cmap=cm.Blues)
ax12.text(0, 35, '(b)', fontsize=25)
ax12.axis('scaled')
ax12.set_title('Density profile @'+str(t_shoot[1]/1000)+' ns', fontsize=25)
ax12.tick_params(axis="both", labelsize=25)
# ax12.set_xlabel('x [nm]', fontsize=20)
# ax12.set_ylabel('z [nm]', fontsize=20)

density_array = dm.read_density_file(folder_name+'/'+file_root+ \
            '{:05d}'.format(n_shoot[2])+'.dat', bin='y')

ax21.pcolormesh(X, Z, density_array, cmap=cm.Blues)
ax21.text(0, 35, '(c)', fontsize=25)
ax21.axis('scaled')
ax21.set_title('Density profile @'+str(t_shoot[2]/1000)+' ns', fontsize=25)
ax21.tick_params(axis="both", labelsize=25)
ax21.set_xlabel('x [nm]', fontsize=25)
ax21.set_ylabel('z [nm]', fontsize=25)

"""

Ly = 4.67650
cell_vol = 0.2*0.2*Ly
rho_fac = 1.66053906660/(cell_vol)

f, (ax22) = plt.subplots()

"""
x_low = 18
x_upp = 30
z_low = 0
z_upp = 4
"""

x_low = 0
x_upp = Lx
z_low = 0
z_upp = Lz

i_low = np.argmin(np.abs(x-x_low))
i_upp = np.argmin(np.abs(x-x_upp))
j_low = np.argmin(np.abs(z-z_low))
j_upp = np.argmin(np.abs(z-z_upp))

X_zoom, Z_zoom = np.meshgrid(x[i_low:i_upp], z[j_low:j_upp], sparse=False, indexing='ij')

density_array = dm.read_density_file(folder_name+'/'+file_root+ \
            '{:05d}'.format(n_shoot[3])+'.dat', bin='y')

bulk_density = dm.detect_bulk_density(density_array,10)
print(bulk_density)
contour = dm.detect_contour(density_array,0.5*bulk_density,hx,hz)
print(contour)

plt.pcolormesh(X_zoom, Z_zoom, density_array[i_low:i_upp,j_low:j_upp]*rho_fac, cmap=cm.Blues)
# plt.pcolormesh(X, Z, density_array*rho_fac, cmap=cm.Blues)
# ax22.text(0, 35, '(d)', fontsize=25)
plt.plot(contour[0],contour[1],'k-')
plt.axis('scaled')
# ax22.set_title('Density profile @'+str(t_shoot[3]/1000)+' ns', fontsize=25)
plt.tick_params(axis="both", labelsize=25)
plt.xlabel('x [nm]', fontsize=25)
plt.ylabel('z [nm]', fontsize=25)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=20)
cbar.ax.get_yaxis().labelpad = 20
cbar.ax.set_ylabel(r'$\rho$ [kg/m$^3$]', fontsize=20, rotation=270)

plt.show()