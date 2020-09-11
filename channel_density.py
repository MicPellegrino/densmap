import densmap as dm

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

file_root = 'flow_'

# Linear flow profile fit
print("[densmap] Obtaning density profile")

FP = dm.fitting_parameters( par_file='parameters_nano.txt' )
folder_name = FP.folder_name

Lx = FP.lenght_x
Lz = FP.lenght_z

rho = dm.read_density_file(folder_name+'/'+file_root+'00001.dat', bin='y')
Nx = rho.shape[0]
Nz = rho.shape[1]
hx = Lx/Nx
hz = Lz/Nz
x = hx*np.arange(0.0,Nx,1.0, dtype=float)
z = hz*np.arange(0.0,Nz,1.0, dtype=float)
X, Z = np.meshgrid(x, z, sparse=False, indexing='ij')

profile_density = np.zeros( len(z), dtype=float )

spin_up_steps = 49
n_init = FP.first_stamp + spin_up_steps
n_fin = FP.last_stamp
dt = FP.time_step

n_dump = 10
print("Producing averaged profile ")
for idx in range( n_init, n_fin ):
    if idx%n_dump==0 :
        print("Obtainig frame "+str(idx))
        t_label = str(dt*idx)+' ps'
    # Time-averaging window
    rho = dm.read_density_file(folder_name+'/'+file_root+'{:05d}'.format(idx)+'.dat', bin='y')
    profile_density = np.add( np.mean(rho, axis=0), profile_density )

profile_density /= n_fin-n_init

z_low = 0.5*(0.579657+0.254902)
z_upp = 0.5*(2.49755+2.1667)

d_so = 0.151
dz = z[1]-z[0]

z_bulk_low = 0.9277
z_bulk_upp = 1.8085

bulk_density = 0.0
n = 0
for i in range(len(z)) :
    if z[i]+0.5*dz > z_bulk_low and z[i]+0.5*dz < z_bulk_upp :
        bulk_density += profile_density[i]
        n += 1
bulk_density /= n

solid_profile = np.zeros(len(z))
for i in range(len(z)) :
    if z[i]+dz < z_low+d_so or z[i] > z_upp-d_so :
        solid_profile[i] = 1.0
    elif z[i] > z_low+d_so and z[i]+dz < z_upp-d_so :
        solid_profile[i] = 0.0
    else :
        if z[i] < 0.5*z[-1] :
            solid_profile[i] = (z_low+d_so-z[i])/dz
        else :
            solid_profile[i] = (z[i]+dz-z_upp+d_so)/dz

SOL_reduced = profile_density/bulk_density
SUB_reduced = solid_profile

depletion_layer = 0.5*dz*np.sum(1.0-SOL_reduced-SUB_reduced)

print("Estimate for the depletion layer:")
print("delta = "+str(depletion_layer)+" nm")

plt.plot(z+0.5*dz, profile_density, 'ko--', linewidth=1.5, markeredgewidth=2.5, markersize=12.5, label='density SOL')
plt.plot(z+0.5*dz, max(profile_density)*solid_profile, 'bx', markeredgewidth=2.5, markersize=12.5, label='density SUB')
plt.plot(z+0.5*dz, bulk_density*np.ones(len(z)), 'b--', linewidth=1.5, label='SOL bulk density')
plt.plot([z_low, z_low], [0.0, 1.25*max(profile_density)], 'r-', linewidth=3.5, label='silica')
plt.plot([z_upp, z_upp], [0.0, 1.25*max(profile_density)], 'r-', linewidth=3.5)
plt.plot([z_low+d_so, z_low+d_so], [0.0, 1.25*max(profile_density)], 'r--', linewidth=3.5, label='oxigen')
plt.plot([z_upp+d_so, z_upp+d_so], [0.0, 1.25*max(profile_density)], 'r--', linewidth=3.5)
plt.plot([z_low-d_so, z_low-d_so], [0.0, 1.25*max(profile_density)], 'r--', linewidth=3.5)
plt.plot([z_upp-d_so, z_upp-d_so], [0.0, 1.25*max(profile_density)], 'r--', linewidth=3.5)
plt.plot()
plt.legend(fontsize=20.0)
plt.xlabel('z [nm]', fontsize=30.0)
plt.ylabel(r'$\rho$ [amu/nm^2]', fontsize=30.0)
plt.xticks(fontsize=30.0)
plt.yticks(fontsize=30.0)
plt.xlim([z[0], z[-1]+dz])
plt.ylim([0.0, 1.25*max(profile_density)])
plt.title('Near-walls density profile', fontsize=30.0)
plt.show()
