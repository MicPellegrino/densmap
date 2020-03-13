import densmap as dm

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

mpl.use('TkAgg')

folder_name = 'flow_20nm'
file_name = 'flow_00100.dat'

# PARAMETERS TO TUNE
Lx = 60.00000       # [nm]
Lz = 35.37240       # [nm]

# ONLY IN CASE IT COMES FROM GMX DENSMAP
################################################################################
# h = 0.02
# Nx = int( np.round(Lx/h) )
# Nz = int( np.round(Lz/h) )
################################################################################

density_array = dm.read_density_file(folder_name+'/'+file_name, bin='y')
# density_array = dm.read_density_file(folder_name+'/'+file_name, bin='n', n_bin_x=Nx, n_bin_z=Nz)

Nx = density_array.shape[0]
Nz = density_array.shape[1]

hx = Lx/Nx          # [nm]
hz = Lz/Nz          # [nm]

# PARAMETERS TO TUNE
# Do not crop (for now)
nx = Nx
nz = Nz

x = hx*np.arange(0.0,nx,1.0, dtype=float)
z = hz*np.arange(0.0,nz,1.0, dtype=float)
X, Z = np.meshgrid(x, z, sparse=False, indexing='ij')

# PARAMETERS TO TUNE
r_h2o = 0.09584

smoother = dm.smooth_kernel(r_h2o, hx, hz)

smooth_density_array = dm.convolute(density_array, smoother)

# PARAMETERS TO TUNE
bulk_density = dm.detect_bulk_density(smooth_density_array, density_th=2.0)
intf_contour = dm.detect_contour(smooth_density_array, 0.5*bulk_density, hx, hz)

# PARAMETERS TO TUNE
# left_branch, right_branch, points_l, points_r = \
#     dm.detect_contact_line(intf_contour, z_min=2.0, z_max=5.0, x_half=12.0)

# PARAMETERS TO TUNE
# foot_l, foot_r, theta_l, theta_r, cot_l, cot_r = \
#     dm.detect_contact_angle(points_l, points_r, order=1)

# FP = dm.fitting_parameters()
# FP.time_step = 2.0
# FP.lenght_x = 23.86485
# FP.lenght_z = 17.68620
# FP.r_mol = 0.09584
# FP.max_vapour_density = 2.0
# FP.substrate_location = 2.0
# FP.bulk_location = 5.0
# FP.simmetry_plane = 12.0
# FP.interpolation_order = 1

# dm.contour_tracking('flow_data3', 200, 300, FP)

# CIRCLE FITTING

# import circle_fit as cf

# z_th = 2.0
# M = len(intf_contour[0,:])
# data_circle_x = []
# data_circle_z = []
# for k in range(M) :
#     if intf_contour[1,k] > z_th :
#         data_circle_x.append(intf_contour[0,k])
#         data_circle_z.append(intf_contour[1,k])
# data_circle_x = np.array(data_circle_x)
# data_circle_z = np.array(data_circle_z)

# xc,zc,R,_ = cf.least_squares_circle(np.stack((data_circle_x, data_circle_z), axis=1))
xc, zc, R, residue = dm.circle_fit(intf_contour, z_th=2.0)
t = np.linspace(0,2*np.pi,250)
circle_x = xc + R*np.cos(t)
circle_z = zc + R*np.sin(t)
print("Circle fit residue = "+str(residue))

################################################################################
#### PLOTTING ##################################################################
################################################################################
# dz = 3.0
# dx_l = dz*cot_l
# dx_r = dz*cot_r
plt.pcolor(X, Z, smooth_density_array, cmap=cm.bone)
plt.colorbar()
plt.plot(intf_contour[0,:], intf_contour[1,:], 'k--', linewidth=2.0)
# plt.plot(data_circle_x, data_circle_z, 'g.', linewidth=2.0)
plt.plot(circle_x, circle_z, 'r-', linewidth=2.0)
# plt.plot(left_branch[0,:], left_branch[1,:], 'r-', right_branch[0,:], right_branch[1,:], 'g-', linewidth=2.0)
# plt.plot(points_r[0,:], points_r[1,:], 'kx', points_l[0,:], points_l[1,:], 'kx')
# plt.plot([points_r[0,0], points_r[0,0]+dx_r], [points_r[1,0],points_r[1,0]+dz] , 'g--',
#     [points_l[0,0], points_l[0,0]+dx_l], [points_l[1,0],points_l[1,0]+dz] , 'r--', linewidth=2.0)
plt.axis('scaled')
plt.xlim([0,Lx])
plt.ylim([0,Lz])
plt.title('Smoothed density output')
plt.show()
