import vtk
import os
import sys

import densmap as dm
import numpy as np

FP = dm.fitting_parameters( par_file='parameters_shear.txt' )

folder_name = FP.folder_name
file_root = 'flow_'

Lx = FP.lenght_x
Lz = FP.lenght_z

# CREATING MESHGRID
print("Creating meshgrid")
dummy_dens = dm.read_density_file(folder_name+'/'+file_root+'00500.dat', bin='y')
dummy_vel_x, dummy_vel_z = dm.read_velocity_file(folder_name+'/'+file_root+'00500.dat')
Nx = dummy_dens.shape[0]
Nz = dummy_dens.shape[1]
hx = Lx/Nx
hz = Lz/Nz
x = hx*np.arange(0.0,Nx,1.0, dtype=float)
z = hz*np.arange(0.0,Nz,1.0, dtype=float)
X, Z = np.meshgrid(x, z, sparse=False, indexing='ij')

# Manually tune, crop window [nm]
x0_crop = 87.00
x1_crop = 100.0
z0_crop = 0.000
z1_crop = 6.000
# Ny need to be strictly larger than 1
Ny = 2
Ly = 25     # [Å]
hy = Ly/Ny

# Creating mesh
idx_x0 = np.argmin(np.abs(x-x0_crop))
idx_x1 = np.argmin(np.abs(x-x1_crop))
idx_z0 = np.argmin(np.abs(z-z0_crop))
idx_z1 = np.argmin(np.abs(z-z1_crop))
x0_crop = x[idx_x0]
x1_crop = x[idx_x1]
z0_crop = z[idx_z0]
z1_crop = z[idx_z1]
x_crop = x[idx_x0:idx_x1]
z_crop = z[idx_z0:idx_z1]

print("Zoom-in window: ["+str(x0_crop)+","+str(x1_crop)+"]x["+str(z0_crop)+","+str(z1_crop)+"], (dX x dZ)")

# INITIALIZING SMOOTHING KERNEL
p = 2.0
r_mol = p*FP.r_mol
smoother = dm.smooth_kernel(r_mol, hx, hz)

# TIME AVERAGING
n_aver = 50
n_init = FP.first_stamp
n_fin = FP.last_stamp
dt = FP.time_step

### TEST ###

## NB in order to match the .pdb format, the measures need to be given in Ångström

test_dens_array = dm.convolute(dummy_dens[idx_x0:idx_x1, idx_z0:idx_z1], smoother)
test_velx_array = dm.convolute(dummy_vel_x[idx_x0:idx_x1, idx_z0:idx_z1], smoother)
test_velz_array = dm.convolute(dummy_vel_z[idx_x0:idx_x1, idx_z0:idx_z1], smoother)
print("Array size : "+str(test_dens_array.shape))

file_name = "output.vtk"

fvtk = open(file_name, 'w')

fvtk.write("# vtk DataFile Version 3.0\n")
# Header
fvtk.write("Vtk output for binned flow configurations (metric in Ångström)\n")
# File format
fvtk.write("ASCII\n")
# Dataset typr
fvtk.write("DATASET STRUCTURED_POINTS\n")
# Grid
fvtk.write("DIMENSIONS "+str(len(x_crop))+" "+str(Ny)+" "+str(len(z_crop))+"\n")
fvtk.write("ORIGIN "+str(10.0*x_crop[0])+" 0.0 "+str(10.0*z_crop[0])+"\n")
fvtk.write("SPACING "+str(10.0*hx)+" "+str(10.0*hy)+" "+str(10.0*hz)+"\n")
# Data
fvtk.write("CELL_DATA "+str((Ny-1)*(len(x_crop)-1)*(len(z_crop)-1))+"\n")
fvtk.write("POINT_DATA "+str(Ny*len(x_crop)*len(z_crop))+"\n")
# fvtk.write("SCALARS density float 1\n")
# fvtk.write("LOOKUP_TABLE default\n")
fvtk.write("VECTORS velocity float\n")
"""
for i in range(len(x_crop)) :
    for j in range(len(z_crop)) :
        fvtk.write(str(test_dens_array[i,j])+"\n")
"""
for k in range(len(z_crop)) :
    for j in range(Ny) :
        for i in range(len(x_crop)) :
            vx_str = "{:.5f}".format(10*test_velx_array[i,k])
            vz_str = "{:.5f}".format(10*test_velz_array[i,k])
            fvtk.write(vx_str+" 0.00000 "+vz_str+"\n")
            # fvtk.write(str(test_dens_array[i,j])+"\n")

fvtk.close()

############
