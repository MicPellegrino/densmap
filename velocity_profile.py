import densmap as dm

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

FP = dm.fitting_parameters( par_file='parameters_shear.txt' )
folder_name = FP.folder_name
file_root = 'flow_'
Lx = FP.lenght_x
Lz = FP.lenght_z

print("Creating meshgrid")
vel_x, vel_z = dm.read_velocity_file(folder_name+'/'+file_root+'00001.dat')
Nx = vel_x.shape[0]
Nz = vel_x.shape[1]
hx = Lx/Nx
hz = Lz/Nz
x = hx*np.arange(0.0,Nx,1.0, dtype=float)+0.5*hx
z = hz*np.arange(0.0,Nz,1.0, dtype=float)+0.5*hz

n_init = FP.first_stamp
n_fin = FP.last_stamp
dt = FP.time_step

# Tunable parameters
n_dump = 10
a = 0.1
x_max = 0.5*Lx + a*0.5*Lz
x_min = 0.5*Lx - a*0.5*Lz
i_max = np.argmin(np.abs(x-x_max))
i_min = np.argmin(np.abs(x-x_min))
U = 0.00300   # nm/ps (SiO2, theta=99deg)
# U = 0.01000     # nm/ps (SiO2, theta=123deg)
# U = 0.01667    # nm/ps (LJ, theta=111deg)

velocity_profile = np.zeros( Nz, dtype=float )

for idx in range(n_init, n_fin+1) :

    if idx%n_dump==0 :
        print("Obtainig frame "+str(idx))

    v_x, v_z = dm.read_velocity_file(folder_name+'/'+file_root+'{:05d}'.format(idx)+'.dat')

    velocity_profile += np.mean(v_x[i_min:i_max+1,], axis=0)

velocity_profile /= (n_fin-n_init+1)

plt.plot(z, velocity_profile, 'k-', linewidth=5.0)
plt.plot(z, np.ones(Nz)*U, 'g--', linewidth=5.0)
plt.plot(z, -np.ones(Nz)*U, 'g--', linewidth=5.0)
plt.ylabel("v [nm/ps]", fontsize=25)
plt.xlabel("z [nm]", fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim([0,Lz])
plt.show()
