"""
    A stupid attempt to directly verify GNBC boundary conditions ...
"""

import densmap as dm

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

# Physical parameters
surf_tens = 5.78e-2     # Pa*m
rho_w = 986             # Kg/m^3
visc_ratio = 0.1        # nondim
z_max = 2.0             # nm
U = 3.33                # m/s = nm/ns
Ly = 4.6765             # nm

def pfm(rho) :
    return (2.0/(1-visc_ratio)) * ( rho/rho_w - visc_ratio ) - 1.0
pf_mapping = np.vectorize(pfm)

def gp(rho) :
    return 0.75 * (c**2 - 1.0)
g_prime = np.vectorize(gp)

## Positions and lenghts in nanometers ##
FP = dm.fitting_parameters( par_file='parameters_shear.txt' )
folder_name = FP.folder_name
file_root = 'flow_'
Lx = FP.lenght_x
Lz = FP.lenght_z
vel_x, vel_z = dm.read_velocity_file(folder_name+'/'+file_root+'00100.dat')
Nx = vel_x.shape[0]
Nz = vel_x.shape[1]
hx = Lx/Nx
hz = Lz/Nz
x = hx*np.arange(0.0,Nx,1.0, dtype=float) + 0.5*hx
z = hz*np.arange(0.0,Nz,1.0, dtype=float) + 0.5*hz
X, Z = np.meshgrid(x, z, sparse=False, indexing='ij')
idx_z_max = np.argmin(np.abs(z-z_max))
x_crop = x
z_crop = z[0:idx_z_max]
X_crop, Z_crop = np.meshgrid(x_crop, z_crop, sparse=False, indexing='ij')
bin_volume = Ly*hx*hz   # nm^3
#########################################

vx_avg = np.zeros( vel_x[:,:idx_z_max].shape, dtype=float )
vz_avg = np.zeros( vel_x[:,:idx_z_max].shape, dtype=float )
rho_avg = np.zeros( vel_x[:,:idx_z_max].shape, dtype=float )
n_init = FP.first_stamp
n_fin = FP.last_stamp
n_dump = 10
for idx in range(n_init, n_fin+1) :
    if idx%n_dump==0 :
        print("Obtainig frame "+str(idx))
    v_x, v_z = dm.read_velocity_file(folder_name+'/'+file_root+'{:05d}'.format(idx)+'.dat')
    rho = dm.read_density_file(folder_name+'/'+file_root+'{:05d}'.format(idx)+'.dat', bin='y')
    """
    dx_rho = ( np.roll(rho, 1, axis=1) + np.roll(rho, -1, axis=1) ) / (2.0*hx)
    dz_rho = ( np.roll(rho, 1, axis=0) + np.roll(rho, -1, axis=0) ) / (2.0*hz)
    """
    vx_avg += v_x[:,:idx_z_max]
    vz_avg += v_z[:,:idx_z_max]
    rho_avg += rho[:,:idx_z_max]
vx_avg /= (n_fin-n_init+1)
vz_avg /= (n_fin-n_init+1)
# From nm/ps to nm/ns=m/s
vx_avg *= 1000
vz_avg *= 1000
rho_avg /= (n_fin-n_init+1)
# From amu to amu/nm^3
rho_avg /= bin_volume
# From amu/nm^3 to kg/m^3
rho_avg *= 1.66054

phase_var = pf_mapping(rho_avg)

plt.pcolormesh(X_crop, Z_crop, phase_var, cmap=cm.bone)
plt.colorbar()
plt.xlabel('x [nm]')
plt.ylabel('z [nm]')
plt.title('Phase variable')
plt.show()

plt.pcolormesh(X_crop, Z_crop, vx_avg, cmap=cm.jet)
plt.colorbar()
plt.xlabel('x [nm]')
plt.ylabel('z [nm]')
plt.title('vel. x [m/s]')
plt.show()

plt.pcolormesh(X_crop, Z_crop, vz_avg, cmap=cm.jet)
plt.colorbar()
plt.xlabel('x [nm]')
plt.ylabel('z [nm]')
plt.title('vel. z [m/s]')
plt.show()
