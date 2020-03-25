import densmap as dm

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

folder_name = '20nm/flow_adv_w5'
file_name = 'flow_00300.dat'

# PARAMETERS TO TUNE
Lx = 60.00000       # [nm]
Lz = 35.37240       # [nm]

vel_x, vel_z = dm.read_velocity_file(folder_name+'/'+file_name)

Nx = vel_x.shape[0]
Nz = vel_x.shape[1]
hx = Lx/Nx          # [nm]
hz = Lz/Nz          # [nm]
x = hx*np.arange(0.0,Nx,1.0, dtype=float)
z = hz*np.arange(0.0,Nz,1.0, dtype=float)
X, Z = np.meshgrid(x, z, sparse=False, indexing='ij')

kin_ener = 0.5*(np.multiply(vel_x, vel_x)+np.multiply(vel_z, vel_z))

r_h2o = 0.09584
smoother = dm.smooth_kernel(r_h2o, hx, hz)
smooth_kin_ener = dm.convolute(kin_ener, smoother)

plt.pcolor(X[:,0:int(Nz/2)], Z[:,0:int(Nz/2)], smooth_kin_ener[:,0:int(Nz/2)], cmap=cm.jet)
plt.colorbar()
plt.axis('scaled')
plt.xlim([0,Lx])
plt.ylim([0,Lz/2])
plt.show()
