import densmap as dm
import numpy as np
import matplotlib.pyplot as plt

file_root = 'flow_'
FP = dm.fitting_parameters( par_file='parameters_test.txt' )
folder_name = FP.folder_name
Lx = FP.lenght_x
Lz = FP.lenght_z

vel_x, vel_z = dm.read_velocity_file(folder_name+'/'+file_root+'00001.dat')
Nx = vel_x.shape[0]
Nz = vel_x.shape[1]
hx = Lx/Nx
hz = Lz/Nz
x = hx*np.arange(0.0,Nx,1.0, dtype=float)+0.5*hx
z = hz*np.arange(0.0,Nz,1.0, dtype=float)+0.5*hz
X, Z = np.meshgrid(x, z, sparse=False, indexing='ij')

n_init = FP.first_stamp
n_fin = FP.last_stamp
dt = FP.time_step

velocity_profile = np.zeros( len(z), dtype=float )

print("Producing averaged profile ")
for idx in range( n_init, n_fin ):
    U, V = dm.read_velocity_file(folder_name+'/'+file_root+'{:05d}'.format(idx)+'.dat')
    velocity_profile = np.add( np.mean(U, axis=0), velocity_profile )

velocity_profile /= n_fin-n_init

n_exclude = 10
p = np.polyfit(z[n_exclude:-n_exclude], velocity_profile[n_exclude:-n_exclude], 2)
print(p)

plt.plot(velocity_profile, z, 'k-', linewidth=2.0)
plt.plot(velocity_profile[n_exclude:-n_exclude:5], z[n_exclude:-n_exclude:5], 'ro', markersize=10.0)
plt.plot(np.polyval(p, z), z, 'b--', linewidth=3.0)
plt.title(r'Couette flow, $q_1$, Ca=0.1', fontsize=25.0)
plt.ylabel(r'$z$ [nm]', fontsize=25.0)
plt.xlabel(r'$u$ [nm/ps]', fontsize=25.0)
plt.xticks(fontsize=20.0)
plt.yticks(fontsize=20.0)
plt.ylim([z[0], z[-1]])
plt.show()
