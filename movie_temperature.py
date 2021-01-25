import densmap as dm

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as manimation

mpl.use("Agg")

FP = dm.fitting_parameters( par_file='parameters_shear.txt' )

folder_name = FP.folder_name
file_root = 'flow_'

Lx = FP.lenght_x
Lz = FP.lenght_z

# CREATING MESHGRID
print("Creating meshgrid")
temp = dm.read_temperature_file(folder_name+'/'+file_root+'00100.dat')
Nx = temp.shape[0]
Nz = temp.shape[1]
hx = Lx/Nx
hz = Lz/Nz
x = hx*np.arange(0.0,Nx,1.0, dtype=float)
z = hz*np.arange(0.0,Nz,1.0, dtype=float)
X, Z = np.meshgrid(x, z, sparse=False, indexing='ij')

n_init = FP.first_stamp
n_fin = FP.last_stamp
dt = FP.time_step

n_dump = 10
print("Producing movie of temperature field")
FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict( title='Shear flow Ca=0.1, Q2', \
    artist='Michele Pellegrino', \
    comment='Results from flow data binning' )
writer = FFMpegWriter(fps=30, metadata=metadata)
fig = plt.figure(figsize=(14.0,7.0))
p_x_list = []
p_z_list = []
kin_ener_list = []
with writer.saving(fig, "temp_q2_ca01_double.mp4", 250):
    t_label = '0.0'
    for idx in range(n_init, n_fin+1):
        plt.xlabel('x [nm]')
        plt.ylabel('z [nm]')
        if idx%n_dump==0 :
            print("Obtainig frame "+str(idx))
            t_label = str(dt*idx)+' ps'
        temperature_array = dm.read_temperature_file(folder_name+'/'+file_root+ \
            '{:05d}'.format(idx)+'.dat')
        dm.export_scalar_vtk(x, z, hx, hz, 2.5, temperature_array, "/home/michele/densmap/TemperatureVTK/temperature"+'{:05d}'.format(idx)+".vtk")
        plt.pcolormesh(X, Z, temperature_array, cmap=cm.plasma)
        plt.axis('scaled')
        plt.title('Temperature profile @'+t_label)
        writer.grab_frame()
        plt.cla()
        plt.clf()
        

mpl.use("TkAgg")
