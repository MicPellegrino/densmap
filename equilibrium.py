import densmap as dm
import numpy as np
import math
import scipy as sc
import scipy.special

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

# PARAMETERS TO TUNE
# Setting up (1) dimensions, (2) bins, (3) kernel, (4) thresholds
print('Initialize smoothing kernel')
Lx = 60.00000       # [nm]
Lz = 35.37240       # [nm]
r_h2o = 0.09584
dummy_file = '20nm/flow_sit_f/flow_00125.dat'
density_array = dm.read_density_file(dummy_file, bin='y')
Nx = density_array.shape[0]
Nz = density_array.shape[1]
hx = Lx/Nx
hz = Lz/Nz
nx = Nx
nz = Nz
x = hx*np.arange(0.0,nx,1.0, dtype=float)
z = hz*np.arange(0.0,nz,1.0, dtype=float)
X, Z = np.meshgrid(x, z, sparse=False, indexing='ij')
smoother = dm.smooth_kernel(r_h2o, hx, hz)
density_th=2.0
h=1.85

# Data structures for radius and contact angle
corrugation = ['f','w1','w2','w3','w4','w5']
charges = ['q1','q3','q4','q5']
q_vals = [0.40, 0.67, 0.74, 0.79]
a = np.array([0.00, 0.75, 1.00, 1.25, 1.50, 1.75])
rough_par = dm.rough_parameter(a)
radius_eq = dict()
angle_eq = dict()
for q in charges :
    radius_eq[q] = dict()
    angle_eq[q] = dict()
    for c in corrugation :
        radius_eq[q][c] = 0.0
        angle_eq[q][c] = 0.0

n_max_hydrophobic = 125
n_max_hydrophylic = 375
n_mean = 5

def eq_dens ( filename ) :
    radius, theta = dm.equilibrium_from_density( filename, smoother, density_th, hx, hz, h )
    return radius, theta

# TEST #
# file_name = dummy_file
# eq_dens(file_name)
########

# Master folders
folder_q1 = '20nm'
folder_q3 = '20nm/CHARGE_3'
folder_q4 = '20nm/CHARGE_4'
folder_q5 = '20nm'
folder_names = dict()
for q in charges :
    folder_names[q] = dict()
for i in range(len(corrugation)) :
    if i == 0:
        folder_names['q1'][corrugation[i]] = folder_q1+'/flow_sit_f/'
        folder_names['q3'][corrugation[i]] = folder_q3+'/flat/'
        folder_names['q4'][corrugation[i]] = folder_q4+'/flat/'
        folder_names['q5'][corrugation[i]] = folder_q5+'/flow_adv_f/'
    else :
        folder_names['q1'][corrugation[i]] = folder_q1+'/flow_sit_w'+str(i)+'/'
        folder_names['q3'][corrugation[i]] = folder_q3+'/wave'+str(i)+'/'
        folder_names['q4'][corrugation[i]] = folder_q4+'/wave'+str(i)+'/'
        folder_names['q5'][corrugation[i]] = folder_q5+'/flow_adv_w'+str(i)+'/'

# TEST #
# print(folder_names)
########

for c in corrugation:
    print('Obtaining contact angles: '+c)
    # Q1
    print('-> charge: Q1')
    theta = 0
    r = 0
    for i in range(n_mean) :
        r_temp, theta_temp = eq_dens( \
            folder_names['q1'][c]+'flow_'+'{:05d}'.format(n_max_hydrophobic-i)+'.dat' )
        # TEST #
        # print(folder_names['q1'][c])
        ########
        theta += theta_temp
        r += r_temp
    theta /= n_mean
    r /= n_mean
    radius_eq['q1'][c] = r
    angle_eq['q1'][c] = theta
    # Q3
    print('-> charge: Q3')
    theta = 0
    r = 0
    for i in range(n_mean) :
        r_temp, theta_temp = eq_dens( \
            folder_names['q3'][c]+'flow_'+'{:05d}'.format(n_max_hydrophylic-i)+'.dat' )
        theta += theta_temp
        r += r_temp
    theta /= n_mean
    r /= n_mean
    radius_eq['q3'][c] = r
    angle_eq['q3'][c] = theta
    # Q4
    print('-> charge: Q4')
    theta = 0
    r = 0
    for i in range(n_mean) :
        r_temp, theta_temp = eq_dens( \
            folder_names['q4'][c]+'flow_'+'{:05d}'.format(n_max_hydrophylic-i)+'.dat' )
        theta += theta_temp
        r += r_temp
    theta /= n_mean
    r /= n_mean
    radius_eq['q4'][c] = r
    angle_eq['q4'][c] = theta
    # Q5
    print('-> charge: Q5')
    theta = 0
    r = 0
    for i in range(n_mean) :
        r_temp, theta_temp = eq_dens( \
            folder_names['q5'][c]+'flow_'+'{:05d}'.format(n_max_hydrophylic-i)+'.dat' )
        theta += theta_temp
        r += r_temp
    theta /= n_mean
    r /= n_mean
    radius_eq['q5'][c] = r
    angle_eq['q5'][c] = theta

# TEST #
# print(radius_eq)
# print(angle_eq)
########

"""
radius_eq_list = dict()
angle_eq_list = dict()
for q in charges :
    radius_eq_list[q] = []
    angle_eq_list[q] = []
    for c in corrugation :
        radius_eq_list[q].append(radius_eq[q][c])
        angle_eq_list[q].append(angle_eq[q][c])
"""

radius_eq_list = dict()
angle_eq_list = dict()
for c in corrugation :
    radius_eq_list[c] = []
    angle_eq_list[c] = []
    for q in charges :
        radius_eq_list[c].append(radius_eq[q][c])
        angle_eq_list[c].append(angle_eq[q][c])

sat_arccos = lambda cval : 0.0 if (cval>1.0) else np.rad2deg(np.arccos(cval))

# angle_wenzel = dict()
# for q in charges :
#     angle_wenzel[q] = []
#     for i in range(len(rough_par)):
#         angle_wenzel[q].append( sat_arccos( rough_par[i] * \
#             np.cos(np.deg2rad(angle_eq_list[q][0])) ) )

# plt.plot(rough_par, angle_eq_list['q1'], 'kh--', markersize=10.0, label='Q1')
# plt.plot(rough_par, angle_wenzel['q1'], 'k-', linewidth=0.5)
# plt.plot(rough_par, angle_eq_list['q3'], 'bv--', markersize=10.0, label='Q3')
# plt.plot(rough_par, angle_wenzel['q3'], 'b-', linewidth=0.5)
# plt.plot(rough_par, angle_eq_list['q4'], 'rs--', markersize=10.0, label='Q4')
# plt.plot(rough_par, angle_wenzel['q4'], 'r-', linewidth=0.5)
# plt.plot(rough_par, angle_eq_list['q5'], 'gd--', markersize=10.0, label='Q5')
# plt.plot(rough_par, angle_wenzel['q5'], 'g-', linewidth=0.5)
# plt.legend(fontsize=20.0)
# plt.title('Impact of rougnness on equilibrium c.a. (advancing)', fontsize=20.0)
# plt.xlabel('Roughness parameter [nondim.]', fontsize=20.0)
# plt.ylabel('Contact angle [deg]', fontsize=20.0)
# plt.xticks(fontsize=20.0)
# plt.yticks(fontsize=20.0)
# plt.show()

young_cosine = []
for q in charges:
    young_cosine.append( np.cos(np.deg2rad(angle_eq[q]['f'])) )
wenzel_cosine = dict()
for c in corrugation:
    wenzel_cosine[c] = []
    for q in charges:
        wenzel_cosine[c].append( np.cos(np.deg2rad(angle_eq[q][c]) ) )

plt.plot(young_cosine[1:], wenzel_cosine['f'][1:], 'kh--', markersize=10.0, label='a='+str(a[0]))
plt.plot(young_cosine[1:], wenzel_cosine['w1'][1:], 'gv--', markersize=10.0, label='a='+str(a[1]))
plt.plot(young_cosine[1:], wenzel_cosine['w2'][1:], 'rs--', markersize=10.0, label='a='+str(a[2]))
plt.plot(young_cosine[1:], wenzel_cosine['w3'][1:], 'cd--', markersize=10.0, label='a='+str(a[3]))
plt.plot(young_cosine[1:], wenzel_cosine['w4'][1:], 'mp--', markersize=10.0, label='a='+str(a[4]))
plt.plot(young_cosine[1:], wenzel_cosine['w5'][1:], 'b^--', markersize=10.0, label='a='+str(a[5]))
plt.legend(fontsize=20.0)
plt.title('Verification of Wenzel law', fontsize=20.0)
plt.xlabel(r'$cos\theta_Y$', fontsize=20.0)
plt.ylabel(r'$cos\theta_W$', fontsize=20.0)
plt.xticks(fontsize=20.0)
plt.yticks(fontsize=20.0)
plt.show()

# plt.plot(rough_par, radius_eq_list['q1'], 'kh--', markersize=10.0, label='Q1')
# plt.plot(rough_par, radius_eq_list['q3'], 'bv--', markersize=10.0, label='Q3')
# plt.plot(rough_par, radius_eq_list['q4'], 'rs--', markersize=10.0, label='Q4')
# plt.plot(rough_par, radius_eq_list['q5'], 'gd--', markersize=10.0, label='Q5')
# plt.legend(fontsize=20.0)
# plt.title('Impact of rougnness on equilibrium radius (advancing)', fontsize=20.0)
# plt.xlabel('Roughness parameter [nondim.]', fontsize=20.0)
# plt.ylabel('Base radius [nm]', fontsize=20.0)
# plt.xticks(fontsize=20.0)
# plt.yticks(fontsize=20.0)
# plt.show()
