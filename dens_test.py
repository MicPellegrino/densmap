import densmap as dm

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

mpl.use('TkAgg')

# Small droplet
# folder_name = 'flow_data5'
# file_name = 'flow_00400.dat'
# Medium droplet
# folder_name = '20nm/flow_adv_w1/'
# file_name = 'flow_00200.dat'
# Large droplet
# folder_name = '100nm/second_run'
# file_name = 'flow_00001.dat'
# Small shear droplet
# folder_name = 'Shear/flow'
# file_name = 'flow_00634.dat'

# PARAMETERS TO TUNE
# Small droplet
# Lx = 23.86485
# Lz = 17.68620
# Medium droplet
# Lx = 60.00000
# Lz = 35.37240
# Large droplet
# Lx = 300.00000
# Lz = 200.44360
# Small shear droplet
# Lx = 149.85000
# Lz = 51.35000

# Shear droplet
folder_name = 'ShearChar/Q2/'
file_name = 'flow_00300.dat'
Lx = 159.75000
Lz = 30.36600

# ONLY IN CASE IT COMES FROM GMX DENSMAP
"""
h = 0.02
Nx = int( np.round(Lx/h) )
Nz = int( np.round(Lz/h) )
"""

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

# CHANGE IN MAIN DENSMAP.PY !!!
x = hx*np.arange(0.0, nx, 1.0, dtype=float)+0.5*hx
z = hz*np.arange(0.0, nz, 1.0, dtype=float)+0.5*hz
X, Z = np.meshgrid(x, z, sparse=False, indexing='ij')

# PARAMETERS TO TUNE
r_h2o = 0.09584
# r_pro = 0.17
# r_half = 0.5 * ( r_h2o + r_pro )
# alpha = 3.0
alpha = 1.0

smoother = dm.smooth_kernel(alpha*r_h2o, hx, hz)
# smoother = dm.smooth_kernel(alpha*r_half, hz, hz)

smooth_density_array = dm.convolute(density_array, smoother)

# PARAMETERS TO TUNE
# aka 'max_vapour_density'
delta_th = 2.0      

bulk_density = dm.detect_bulk_density(smooth_density_array, delta_th)
intf_contour = dm.detect_contour(smooth_density_array, 0.5*bulk_density, hx, hz)

density_indicator = dm.density_indicator(smooth_density_array, 0.5*bulk_density)

# PARAMETERS TO TUNE
z0 = 0.85
left_branch, right_branch, points_l, points_r = \
    dm.detect_contact_line(intf_contour, z_min=z0, z_max=Lz-z0, x_half=0.5*Lx, m=10, d=10)

# PARAMETERS TO TUNE
# foot_l, foot_r, theta_l, theta_r, cot_l, cot_r = \
#     dm.detect_contact_angle(points_l, points_r, order=2)

# Finding interface contour from bisection-interpolation
left_branch_man, right_branch_man = dm.detect_interface_int(smooth_density_array, 0.5*bulk_density, hx, hz, z0)
"""
M = int(np.ceil(0.5*Nx))
i0 = np.abs(z-z0).argmin()
left_branch_man = np.zeros( (2,Nz-2*i0), dtype=float )
right_branch_man = np.zeros( (2,Nz-2*i0), dtype=float )
for j in range(i0,Nz-i0) :
    left_branch_man[1,j-i0] = z[j]
    right_branch_man[1,j-i0] = z[j]
    for i in range(0,M) :
        if smooth_density_array[i,j] > 0.5*bulk_density :
            left_branch_man[0,j-i0] = \
                    ((smooth_density_array[i,j]-0.5*bulk_density)*x[i-1]+(0.5*bulk_density-smooth_density_array[i-1,j])*x[i])/(smooth_density_array[i,j]-smooth_density_array[i-1,j])
            break
    for i in range(Nx-1, Nx-M+1, -1) :
        if smooth_density_array[i,j] > 0.5*bulk_density :
            right_branch_man[0,j-i0] = \
                    ((smooth_density_array[i,j]-0.5*bulk_density)*x[i+1]+(0.5*bulk_density-smooth_density_array[i+1,j])*x[i])/(smooth_density_array[i,j]-smooth_density_array[i+1,j])
            break
"""

# Test lenght
print( "Lenght contour (automatic) = "+str(len(left_branch[0,:])) )
print( "Lenght contour (manual)    = "+str(len(left_branch_man[0,:])) )
print( "Spacing std (automatic) = "+str( np.std(left_branch[1,1:]-left_branch[1,0:-1]) ) )
print( "Spacing std (manual)    = "+str( np.std(left_branch_man[1,1:]-left_branch_man[1,0:-1]) ) )

"""
FP = dm.fitting_parameters()
FP.time_step = 2.0
FP.lenght_x = 23.86485
FP.lenght_z = 17.68620
FP.r_mol = 0.09584
FP.max_vapour_density = 2.0
FP.substrate_location = 2.0
FP.bulk_location = 5.0
FP.simmetry_planefontsize=20.0 = 12.0
FP.interpolation_order = 1
dm.droplet_tracking('flow_data3', 200, 300, FP)
"""

# CIRCLE FITTING
"""
z_th = 2.5
M = len(intf_contour[0,:])
data_circle_x = []
data_circle_z = []
for k in range(M) :
    if intf_contour[1,k] > z_th :
        data_circle_x.append(intf_contour[0,k])
        data_circle_z.append(intf_contour[1,k])
data_circle_x = np.array(data_circle_x)
data_circle_z = np.array(data_circle_z)
h = 2.5
xc, zc, R, residue = dm.circle_fit(intf_contour, z_th=h)
t = np.linspace(0,2*np.pi,250)
circle_x = xc + R*np.cos(t)
circle_z = zc + R*np.sin(t)
print("Circle fit residue = "+str(residue))
cot_circle = (h-zc)/np.sqrt(R*R-(h-zc)**2)
theta_circle = np.rad2deg( -np.arctan( cot_circle )+0.5*np.pi )
theta_circle = theta_circle + 180*(theta_circle<=0)
print(theta_circle)
"""

################################################################################
#### PLOTTING ##################################################################
################################################################################

"""
plt.pcolor(X, Z, density_array, cmap=cm.bone)
plt.colorbar()
plt.axis('scaled')
plt.xlim([0,Lx])
plt.ylim([0,Lz])
plt.title('Initial density output', fontsize=20.0)
plt.xticks(fontsize=20.0)
plt.yticks(fontsize=20.0)
plt.show()
"""

"""
plt.pcolor(smoother, cmap=cm.bone)
plt.colorbar()
plt.axis('scaled')
plt.title('Kernel', fontsize=30.0)
plt.xticks(fontsize=30.0)
plt.yticks(fontsize=30.0)
plt.xlabel('cells', fontsize=30.0)
plt.ylabel('cells', fontsize=30.0)
plt.show()
"""

"""
white_val = 35.0
zero_thresh = 3.0
for i in range (nx) :
    for j in range (nz) :
        smooth_density_array[i][j] = \
            white_val*(smooth_density_array[i][j]<=zero_thresh) + \
            smooth_density_array[i][j]*(smooth_density_array[i][j]>zero_thresh)
"""

dz = 3.0

# dx_l = dz*cot_l
# dx_r = dz*cot_r
# plt.pcolor(X, Z, smooth_density_array, cmap=cm.bone, vmin=0.0, vmax=white_val)

plt.pcolor(X, Z, smooth_density_array, cmap=cm.bone)
# plt.pcolor(X, Z, density_indicator, cmap=cm.bone)
plt.colorbar()

# plt.plot(data_circle_x, data_circle_z, 'g.', linewidth=2.0)
# plt.plot(circle_x, circle_z, 'g-', linewidth=5.0, label='cylindrical cap')
# plt.plot(intf_contour[0,:], intf_contour[1,:], 'r--', linewidth=3.0, label='half density')

# plt.plot(intf_contour[0,:], intf_contour[1,:], 'r-', linewidth=3.0, label='half density')

plt.plot(left_branch[0,:], left_branch[1,:], 'r-', linewidth=2.5, label="marching squares")

plt.plot(right_branch[0,:], right_branch[1,:], 'r-', linewidth=2.5)

plt.plot(left_branch_man[0,:], left_branch_man[1,:], 'g--', linewidth=2.5, label="bin-wise interpolation")

plt.plot(right_branch_man[0,:], right_branch_man[1,:], 'g--', linewidth=2.5)

# plt.plot(points_r[0,:], points_r[1,:], 'kx', points_l[0,:], points_l[1,:], 'kx')
# plt.plot([points_r[0,0], points_r[0,0]+dx_r], [points_r[1,0],points_r[1,0]+dz] , 'b-',
#    [points_l[0,0], points_l[0,0]+dx_l], [points_l[1,0],points_l[1,0]+dz] , 'b-', linewidth=3.0)
# plt.plot([x[0], x[-1]], [1.8, 1.8], 'k--')

plt.axis('scaled')
plt.xlim([0,Lx])
plt.ylim([0,Lz])
plt.title('Binned and smoothed density output', fontsize=30.0)
plt.legend(fontsize=15.0)

# plt.legend(fontsize=20.0)

plt.xlabel('x [nm]', fontsize=30.0)
plt.ylabel('z [nm]', fontsize=30.0)
plt.xticks(fontsize=30.0)
plt.yticks(fontsize=30.0)

plt.show()
