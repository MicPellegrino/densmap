import densmap as dm
import numpy as np
import matplotlib.pyplot as plt

print("[densmap] Obtaning depletion layer from density profile")
file_root = 'flow_'
FP = dm.fitting_parameters( par_file='parameters_nano.txt' )
folder_name = FP.folder_name

# Initialization
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

spin_up_steps = 0
n_init = FP.first_stamp + spin_up_steps
n_fin = FP.last_stamp
dt = FP.time_step

# Binning
n_depletion = 10
n_batch = int( (n_fin-n_init+1) / n_depletion )

d_so = 0.151
dz = z[1]-z[0]
# z_low = np.linspace( 0.5*(0.579657+0.254902), 0.5*(0.640951+0.359069), n_fin-n_init+1 )
# z_upp = np.linspace( 0.5*(2.49755+2.16667), 0.5*(2.38725+2.10539), n_fin-n_init+1 )

z_low = np.linspace( 0.5*(0.579657+0.254902), 0.5*(0.86152+0.579675), n_fin-n_init+1 )
z_upp = np.linspace( 0.5*(2.49755+2.16667), 0.5*(2.16667+1.89706), n_fin-n_init+1 )

print("[densmap] Difference between initial and final height")
print("delta_upp = "+str(z_upp[0]-z_upp[-1]))
print("delta_low = "+str(z_low[-1]-z_low[0]))

# Let's assume those are not changing in time
z_bulk_low = 0.9277
z_bulk_upp = 1.8085

depletion_layer = np.zeros(n_depletion)
bulk_density_vec = np.zeros(n_depletion)

for i in range(n_depletion) :
    
    print("Obtaining frame "+str(i*n_batch))

    profile_density = np.zeros( len(z), dtype=float )
    solid_profile_avg = np.zeros(len(z))

    for idx in range( n_init+i*n_batch, n_init+(i+1)*n_batch ) :
        
        # Liquid
        rho = dm.read_density_file(folder_name+'/'+file_root+'{:05d}'.format(idx)+'.dat', bin='y')
        profile_density = np.add( np.mean(rho, axis=0), profile_density )

        # Solid
        solid_profile = np.zeros(len(z))
        for ii in range(len(z)) :
            if z[ii]+dz < z_low[idx-1]+d_so or z[ii] > z_upp[idx-1]-d_so :
                solid_profile[ii] = 1.0
            elif z[ii] > z_low[idx-1]+d_so and z[ii]+dz < z_upp[idx-1]-d_so :
                solid_profile[ii] = 0.0
            else :
                if z[ii] < 0.5*z[-1] :
                    solid_profile[ii] = (z_low[idx-1]+d_so-z[ii])/dz
                else :
                    solid_profile[ii] = (z[ii]+dz-z_upp[idx-1]+d_so)/dz

        solid_profile_avg += solid_profile

    solid_profile_avg /= n_batch
    profile_density /= n_batch

    # Bulk liquid density
    bulk_density = 0.0
    n = 0
    for ii in range(len(z)) :
        if z[ii]+0.5*dz > z_bulk_low and z[ii]+0.5*dz < z_bulk_upp :
            bulk_density += profile_density[ii]
            n += 1
    bulk_density /= n

    SOL_reduced = profile_density/bulk_density
    SUB_reduced = solid_profile_avg

    depletion_layer[i] = 0.5*dz*np.sum(1.0-SOL_reduced-SUB_reduced)
    bulk_density_vec[i] = bulk_density

    # Plotting
    """
    plt.plot(z+0.5*dz, profile_density, 'ko--', linewidth=1.5, markeredgewidth=2.5, markersize=12.5, label='density SOL')
    plt.plot(z+0.5*dz, max(profile_density)*solid_profile_avg, 'bx', markeredgewidth=2.5, markersize=12.5, label='density SUB')
    plt.plot(z+0.5*dz, bulk_density*np.ones(len(z)), 'b--', linewidth=1.5, label='SOL bulk density')
    plt.plot([z_low[idx-1], z_low[idx-1]], [0.0, 1.25*max(profile_density)], 'r-', linewidth=3.5, label='silica')
    plt.plot([z_upp[idx-1], z_upp[idx-1]], [0.0, 1.25*max(profile_density)], 'r-', linewidth=3.5)
    plt.plot([z_low[idx-1]+d_so, z_low[idx-1]+d_so], [0.0, 1.25*max(profile_density)], 'r--', linewidth=3.5, label='oxigen')
    plt.plot([z_upp[idx-1]+d_so, z_upp[idx-1]+d_so], [0.0, 1.25*max(profile_density)], 'r--', linewidth=3.5)
    plt.plot([z_low[idx-1]-d_so, z_low[idx-1]-d_so], [0.0, 1.25*max(profile_density)], 'r--', linewidth=3.5)
    plt.plot([z_upp[idx-1]-d_so, z_upp[idx-1]-d_so], [0.0, 1.25*max(profile_density)], 'r--', linewidth=3.5)
    plt.legend(fontsize=20.0)
    plt.xlabel('z [nm]', fontsize=30.0)
    plt.ylabel(r'$\rho$ [amu/nm^2]', fontsize=30.0)
    plt.xticks(fontsize=30.0)
    plt.yticks(fontsize=30.0)
    plt.xlim([z[0], z[-1]+dz])
    plt.ylim([0.0, 1.25*max(profile_density)])
    plt.title('Near-walls density profile', fontsize=30.0)
    plt.show()
    """

time_vector = dt*np.array(range(n_init,n_fin,n_batch))

# plt.plot(time_vector, depletion_layer, 'ko-', linewidth=1.5, markeredgewidth=2.5, markersize=7.5, label='depletion layer')
plt.plot(time_vector/max(time_vector), depletion_layer, 'ko-', linewidth=1.5, markeredgewidth=2.5, markersize=7.5, label='depletion layer')
plt.legend(fontsize=20.0)
# plt.xlabel('t [ps]', fontsize=20.0)
plt.xlabel(r'$\lambda [-1]$', fontsize=20.0)
plt.ylabel(r'$\delta$ [nm]', fontsize=20.0)
plt.xticks(fontsize=20.0)
plt.yticks(fontsize=20.0)
plt.title("Depletion layer vs collective coordinate")
plt.show()

plt.plot(time_vector/max(time_vector), bulk_density_vec, 'ko-', linewidth=1.5, markeredgewidth=2.5, markersize=7.5, label='bulk density')
plt.legend(fontsize=20.0)
# plt.xlabel('t [ps]', fontsize=20.0)
plt.xlabel(r'$\lambda [-1]$', fontsize=20.0)
plt.ylabel(r'$\rho$ [amu/bin]', fontsize=20.0)
plt.xticks(fontsize=20.0)
plt.yticks(fontsize=20.0)
plt.title("Bulk density vs collective coordinate")
plt.show()

plt.plot(bulk_density_vec, depletion_layer, 'kx', linewidth=1.5, markeredgewidth=2.5, markersize=7.5)
plt.xlabel(r'$\rho$ [amu/bin]', fontsize=20.0)
plt.ylabel(r'$\delta$ [nm]', fontsize=20.0)
plt.xticks(fontsize=20.0)
plt.yticks(fontsize=20.0)
plt.title('Bulk density vs depletion layer')
plt.show()

