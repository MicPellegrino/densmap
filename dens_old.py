import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy as sc
from scipy import signal
import math
import skimage as sk
from skimage import measure

"""
    Read density file
"""

# filename = 'densmap_q1.dat'
# filename = 'densmap_q2.dat'
filename = 'densmap_q3.dat'
# filename = 'densmap_q4.dat'
# filename = 'densmap_q5.dat'
# filename = 'hyst.dat'

Lx = 23.86485
Lz = 17.68620
# Lx = 22.50000
# Lz = 17.68500
h = 0.02

Nx = int( np.round(Lx/h) )
Nz = int( np.round(Lz/h) )

idx = 0
density_array = np.zeros( (Nx+1) * (Nz+1), dtype=float)
for line in open(filename, 'r'):
    vals = np.array( [ float(i) for i in line.split() ] )
    density_array[idx:idx+len(vals)] = vals
    idx += len(vals)

density_array = density_array.reshape((Nx+1,Nz+1))
density_array = density_array[1:-1,1:-1]

################################################################################

"""
    Resize density map
"""

density_array = density_array[int(Nx/6):-1,0:int(Nz/2)]
nx = density_array.shape[0]
nz = density_array.shape[1]

################################################################################

"""
    Spacing an meshgrid
"""

x = h*np.arange(0.0,nx,1.0, dtype=float)
z = h*np.arange(0.0,nz,1.0, dtype=float)
X, Z = np.meshgrid(x, z, sparse=False, indexing='ij')

################################################################################

"""
    Compute gradient (optional)
"""

grad_x = np.zeros((nx, nz), dtype=float)
grad_z = np.zeros((nx, nz), dtype=float)
for i in range(1, nx-1) :
    for j in range (1, nz-1) :
        grad_x[i,j] = 0.5*( density_array[i+1,j] - density_array[i-1,j] )/h
        grad_z[i,j] = 0.5*( density_array[i,j+1] - density_array[i,j-1] )/h
grad_norm = np.sqrt( np.power( grad_x, 2 ) + np.power( grad_z, 2 ) )

################################################################################

"""
    Definition of the smoothing kernel
"""

r_h2o = 0.09584
sigma = 2*r_h2o
Nker = int(sigma/h)
print('sigma \t= %f' % sigma)
print('Nker \t= %d' % Nker)

smoother = np.zeros((2*Nker+1, 2*Nker+1))
for i in range(-Nker,Nker+1):
    for j in range(-Nker,Nker+1):
        dist2 = sigma**2-(i*h)**2-(j*h)**2
        if dist2 >= 0:
            smoother[i+Nker][j+Nker] = np.sqrt(dist2)
        else :
            smoother[i+Nker][j+Nker] = 0.0

smoother = smoother / np.sum(np.sum(smoother))

################################################################################

"""
    Convolve smoothing kernel with density map
"""

smooth_dens = sc.signal.convolve2d(density_array, smoother,
    mode='same', boundary='wrap')

################################################################################

"""
    Automatically identify bulk density value
"""

# Minimum threshold number density
n_min = 20
hist, bin_edges = np.histogram(smooth_dens[smooth_dens >= n_min].flatten(),
    bins=150)
bulk_density = bin_edges[np.argmax(hist)]
intf_density = bulk_density/2.0
print('Bulk number density \t\t= %f' % bulk_density)
print('Interface number density \t= %f' % intf_density)

################################################################################

"""
    Finding interface contour
"""

intf_contour = sk.measure.find_contours(smooth_dens, intf_density)
intf_contour = h*(intf_contour[0].transpose())
# print(intf_contour.shape)

################################################################################

"""
    Obtaining the points near the contact line for each angle
"""

h_0 = 2.0
h_1 = 5.0
# h_0 = 0.75
# h_1 = 3.75
x_half = h*nx/2.0
m = 5
d = 10
# d = 20
left_branch_x = []
left_branch_z = []
right_branch_x = []
right_branch_z = []
for i in range(intf_contour.shape[1]) :
    if intf_contour[1,i] > h_0 and intf_contour[1,i] <= h_1 :
        if intf_contour[0,i] <= x_half :
            left_branch_x.append(intf_contour[0,i])
            left_branch_z.append(intf_contour[1,i])
        else :
            right_branch_x.append(intf_contour[0,i])
            right_branch_z.append(intf_contour[1,i])
left_branch = np.empty((2,len(left_branch_x)))
right_branch = np.empty((2,len(right_branch_x)))
left_branch[0,:] = left_branch_x
left_branch[1,:] = left_branch_z
right_branch[0,:] = right_branch_x
right_branch[1,:] = right_branch_z
# idx_l = np.argsort(left_branch[0,:])
# idx_r = np.flip(np.argsort(right_branch[0,:]))
idx_l = np.argsort(left_branch[1,:])
idx_r = np.argsort(right_branch[1,:])
left_branch = left_branch[:,idx_l]
right_branch = right_branch[:,idx_r]
points_l = left_branch[:,0:d*m:d]
points_r = right_branch[:,0:d*m:d]

################################################################################

"""
    Polynomial fitting for the branches
"""

# Higher order polynomials can be used instead
# for example: quadratic with points more spread-out on the branch
p_l = np.polyfit( points_l[1,:] , points_l[0,:] , 1 )
p_r = np.polyfit( points_r[1,:] , points_r[0,:] , 1 )
cot_l = p_l[0]
cot_r = p_r[0]
# p_l = np.polyfit( points_l[1,:] , points_l[0,:] , 2 )
# p_r = np.polyfit( points_r[1,:] , points_r[0,:] , 2 )
# cot_l = (p_l[1]+2.0*p_l[0]*points_l[1,0])
# cot_r = (p_r[1]+2.0*p_r[0]*points_r[1,0])

################################################################################

"""
    Obtaining contact angles
"""

# Obtaining angles
# theta_l = np.rad2deg( np.arctan( tan_l ) )
# theta_l = theta_l + 180*(theta_l<=0)
# theta_r = - np.rad2deg( np.arctan( tan_r ) )
# theta_r = theta_r + 180*(theta_r<=0)
theta_l = np.rad2deg( -np.arctan( cot_l )+0.5*math.pi )
theta_l = theta_l + 180*(theta_l<=0)
theta_r = - np.rad2deg( -np.arctan( cot_r )+0.5*math.pi )
theta_r = theta_r + 180*(theta_r<=0)
print('Left ca: theta_l \t = %f deg' % theta_l)
print('Right ca: theta_r \t = %f deg' % theta_r)
print('Hysteresis: d_theta \t = %f deg' % np.abs(theta_r-theta_l))

################################################################################

"""
    Plotting
"""

dz = 3.0
# dx_l = dz/tan_l
# dx_r = dz/tan_r
dx_l = dz*cot_l
dx_r = dz*cot_r

# plt.hist(smooth_dens[smooth_dens >= n_min].flatten(), bins=150)
# plt.show()

# plt.subplot(2, 1, 1)
# plt.pcolor(X, Z, density_array, cmap=cm.bone)
# plt.colorbar()
# plt.clim([np.min(np.min(density_array)), np.max(np.max(density_array))])
# plt.plot(intf_contour[0,:], intf_contour[1,:], 'k--')
# plt.plot(left_branch[0,:], left_branch[1,:], 'r-')
# plt.plot(right_branch[0,:], right_branch[1,:], 'b-')
# plt.plot(points_l[0,:], points_l[1,:], points_r[0,:], points_r[1,:], 'ko')
# contour_smooth_not = plt.contour(X, Z, density_array, cmap=cm.cividis, linewidths=0.75)
# plt.axis('scaled')
# plt.subplot(2, 1, 2)
plt.pcolor(X, Z, smooth_dens, cmap=cm.bone)
plt.colorbar()
# plt.clim([np.min(np.min(density_array)), np.max(np.max(density_array))])
plt.plot(intf_contour[0,:], intf_contour[1,:], 'k--', linewidth=2.0)
plt.plot(left_branch[0,:], left_branch[1,:], 'r-', right_branch[0,:], right_branch[1,:], 'g-', linewidth=2.0)
plt.plot(points_r[0,:], points_r[1,:], 'kx', points_l[0,:], points_l[1,:], 'kx')
plt.plot([points_r[0,0], points_r[0,0]+dx_r], [points_r[1,0],points_r[1,0]+dz] , 'g--',
    [points_l[0,0], points_l[0,0]+dx_l], [points_l[1,0],points_l[1,0]+dz] , 'r--', linewidth=2.0)
# contour_smooth_yes = plt.contour(X, Z, smooth_dens, cmap=cm.cividis, linewidths=0.75)
plt.axis('scaled')
plt.show()

# plt.matshow(np.flip(cont_points.transpose(), axis=0), cmap=cm.bone)
# plt.show()

# plt.pcolor(X, Z, density_array, cmap=cm.bone)
# plt.colorbar()
# plt.axis('scaled')
# plt.show()

# plt.pcolor(smoother, cmap=cm.bone)
# plt.colorbar()
# plt.axis('equal')
# plt.show()

# plt.pcolor(X, Z, smooth_dens, cmap=cm.bone)
# plt.colorbar()
# plt.axis('scaled')
# plt.show()

# plt.hist(density_array.reshape((nx*nz,1)), bins=int(np.sqrt(nx*nz)))
# plt.show()

# plt.matshow(np.flip(grad_norm.transpose(), axis=0), cmap=cm.bone)
# plt.contour(grad_norm.transpose())
# plt.colorbar()
# plt.show()
