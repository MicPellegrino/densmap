import densmap as dm
import numpy as np
import matplotlib.pyplot as plt

rho_bulk = 990      # [kg/m^3]
# rho_bulk = 100      
hydrophobic = 0

FP = dm.fitting_parameters( par_file='parameters_shear.txt' )

folder_name = FP.folder_name
file_root = 'flow_'

Lx = FP.lenght_x
Lz = FP.lenght_z
Ly = 4.68           # [nm]

# CREATING MESHGRID
print("Creating meshgrid")
vel_x, vel_z = dm.read_velocity_file(folder_name+'/'+file_root+'00100.dat')
Nx = vel_x.shape[0]
Nz = vel_x.shape[1]
hx = Lx/Nx
hz = Lz/Nz
x = hx*np.arange(0.0,Nx,1.0, dtype=float)
z = hz*np.arange(0.0,Nz,1.0, dtype=float)
X, Z = np.meshgrid(x, z, sparse=False, indexing='ij')
n_init = FP.first_stamp
n_fin = FP.last_stamp
delta_th = 2.0
z0 = 0.80

v_list = []
d_list = []

v_mean = 0
d_mean = 0

n_transient = 0

n_dump = 10
for i in range(n_init+n_transient, n_fin+1 ) :
    file_name = file_root+'{:05d}'.format(i)+'.dat'
    # In kilos (give it or take...)
    density_array = 1.66*dm.read_density_file(folder_name+'/'+file_name, bin='y')
    # print(np.max(density_array))
    indicator = density_array>0.5*rho_bulk
    volume = np.sum(indicator)*hx*hz*Ly
    bulk_density = dm.detect_bulk_density(density_array, delta_th)
    left_intf, right_intf = dm.detect_interface_int(density_array, 0.5*bulk_density, hx, hz, z0)
    diff = np.abs(left_intf[0]-right_intf[0])
    width = 0
    if hydrophobic :
        width = np.max(diff)
    else :
        width = np.min(diff)
    if i % n_dump == 0 :
        print("Obtainig frame "+str(i))
        print("volume = "+str(volume)+" nm^3")
        print("width = "+str(width)+" nm")
    v_list.append(volume)
    d_list.append(width)
    v_mean += volume
    d_mean += width

v_mean /= (n_fin-(n_init+n_transient)+1)
d_mean /= (n_fin-(n_init+n_transient)+1)

print("MEAN VOLUME = "+str(v_mean)+" nm^3")
print("MEAN WIDTH = "+str(d_mean)+" nm")

plt.plot(v_list, 'b-')
plt.title("Volume")
plt.xlabel("frame")
plt.ylabel("v [nm^3]")
plt.show()

plt.plot(d_list, 'r-')
plt.title("Width")
plt.xlabel("frame")
plt.ylabel("d [nm^3]")
plt.show()