import densmap as dm

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as manimation

mpl.use("Agg")

folder_name = 'N16A02Ca25/'
file_root = 'flow_'

# Output file name
output_file_name = "n16a02-ca25.mp4"

# SUBSTRATE
a = 0.2
n = 16
Lx = 82.80000/4
waven  = 2*np.pi*n/Lx
height = a/waven
phi_0  = 0
h_0    = 3.0
fun_sub = lambda x : height * np.sin(waven*x+phi_0) + h_0

# PARAMETERS TO TUNE
Lx = 82.80000
Lz = 28.00000

n_init = 1
n_fin = 850

# CREATING MESHGRID
print("Creating meshgrid")
density_array = dm.read_density_file(folder_name+'/'+file_root+'00001.dat', bin='y')
Nx = density_array.shape[0]
Nz = density_array.shape[1]
hx = Lx/Nx
hz = Lz/Nz
x = hx*np.arange(0.0,Nx,1.0, dtype=float)
z = hz*np.arange(0.0,Nz,1.0, dtype=float)
X, Z = np.meshgrid(x, z, sparse=False, indexing='ij')

# Testing .vtk output function
"""
vtk_folder = "/home/michele/densmap/TestVtk"
dm.export_scalar_vtk(x, z, hx, hz, 2.5, density_array)
"""

# Section for density computation
N_low = int(np.floor(70.0/hx))
N_upp = int(np.ceil(90.0/hx))

# INITIALIZING SMOOTHING KERNEL
r_mol = 0.39876
smoother = dm.smooth_kernel(r_mol, hx, hz)

dt = 12.5
delta_th = 2.0

n_dump = 10
print("Producing movie of the number density")
FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Meniscus density profile', artist='Michele Pellegrino',
    comment='Just the tracked contour of a shear droplet')
writer = FFMpegWriter(fps=30, metadata=metadata)
fig = plt.figure()
area = []

# Density profile
density_profile = np.zeros( Nz, dtype=float )

# Center of mass
x_com = []
z_com = []
t_com = []
avg_dens = []
print(np.sum(density_array))

with writer.saving(fig, output_file_name, 250):

    t_label = '0.0'

    for idx in range(n_init, n_fin+1 ):
        plt.xlabel('x [nm]')
        plt.ylabel('z [nm]')
        if idx%n_dump==0 :
            print("Obtainig frame "+str(idx))
            t_label = str(dt*idx)+' ps'
        density_array = dm.read_density_file(folder_name+'/'+file_root+ \
            '{:05d}'.format(idx)+'.dat', bin='y')

        # PLOT ORIGINAL DENSITY
        plt.pcolormesh(X, Z, density_array, cmap=cm.Blues, vmax=1000)

        # PLOT SMOOTHED DENSITY
        # smooth_density_array = dm.convolute(density_array, smoother)
        # plt.pcolormesh(X, Z, smooth_density_array, cmap=cm.bone)
        
        # PLOT SUBSTRATE
        fig_substrate, = plt.plot([], [], 'm-', linewidth=1.0)
        fig_substrate.set_data(x, fun_sub(x))

        x_com.append( np.sum(np.multiply(density_array,X))/np.sum(density_array) )
        z_com.append( np.sum(np.multiply(density_array,Z))/np.sum(density_array) )
        t_com.append( (1e-3)*idx*dt )
        # avg_dens.append( np.mean(density_array) )

        """
        bulk_density = dm.detect_bulk_density(smooth_density_array, delta_th)
        indicator = dm.density_indicator(smooth_density_array,0.5*bulk_density)
        if idx%n_dump==0 :
            area.append(hx*hz*np.sum(indicator))
            print("Droplet area = "+str(area[-1])+" nm^2")
        density_profile += np.sum( density_array[N_low:N_upp,:], axis=0 )
        plt.pcolormesh(X, Z, indicator, cmap=cm.bone)
        intf_contour = dm.detect_contour(smooth_density_array, 0.5*bulk_density, hx, hz)
        plt.plot(intf_contour[0,:], intf_contour[1,:], 'r-', linewidth=1.5)
        """

        plt.axis('scaled')
        plt.title('Density profile @'+t_label)
        writer.grab_frame()
        plt.cla()
        plt.clf()

        # Exporting corresponding .vtk
        """
        dm.export_scalar_vtk(x, z, hx, hz, 2.5, density_array, file_name=vtk_folder+"/density_"+str(idx).zfill(5)+".vtk")
        """

mpl.use("TkAgg")

"""
idx = n_fin
density_array = dm.read_density_file(folder_name+'/'+file_root+'{:05d}'.format(idx)+'.dat', bin='y')
# PLOT ORIGINAL DENSITY
plt.pcolormesh(X, Z, density_array, cmap=cm.Blues)
plt.axis('scaled')
plt.show()
"""

plt.plot(t_com, x_com, 'r-', linewidth=5.0, label='x')
plt.plot(t_com, z_com, 'b-', linewidth=5.0, label='z')
plt.legend(fontsize=40.0)
plt.title("COM", fontsize=45.0)
plt.xlabel("t [ns]", fontsize=42.5)
plt.ylabel("pos [nm]", fontsize=42.5)
plt.xticks(fontsize=40.0)
plt.yticks(fontsize=40.0)
plt.xlim([t_com[0],t_com[-1]])
plt.show()

"""
plt.plot(t_com, avg_dens, 'b-', linewidth=5.0)
plt.ylabel("amu/nm^3", fontsize=42.5)
plt.xlabel("ps", fontsize=42.5)
plt.xticks(fontsize=40.0)
plt.yticks(fontsize=40.0)
plt.xlim([t_com[0],t_com[-1]])
plt.show()
"""

# Saving COM position
"""
output_com_file = open('InterfaceTest/com.txt', 'w')
for k in range( len(t_com) ) :
    line = str(t_com[k]).zfill(5)+" "+"{:3.5f}".format(x_com[k])+" "+"{:3.5f}".format(z_com[k])+"\n"
    output_com_file.write(line)
output_com_file.close()
"""

# POST-PROCESSING ...
"""
density_profile /= (n_fin-n_init)
area = np.array(area)
avg_area = np.mean(area)
print("Average droplet area = "+str(avg_area)+" nm^2")
plt.plot(z, density_profile)
plt.show()
Nz_th = int(5.0/hz)
bulk_density = np.mean(density_profile[Nz_th:Nz-Nz_th])
L_hat = (1.0/bulk_density) * sum(density_profile) * hz
print("L_hat = "+str(L_hat)+" nm")
"""
