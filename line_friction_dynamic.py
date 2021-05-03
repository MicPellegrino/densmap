from math import isnan

import numpy as np
import matplotlib.pyplot as plt

import scipy.ndimage as smg
import scipy.signal as sgn
import scipy.optimize as opt

def array_from_file( filename ):
    my_list = []
    with open(filename, 'r') as f:
        for line in f:
            my_list.append(float(line.split()[0]))
    return np.array(my_list)

cos = lambda t : np.cos(np.deg2rad(t))

theta_0 = 37.8
capillary_number = 0.02

fn = 'ShearDynamic/Q4_Ca002'

time = array_from_file(fn+'/time.txt')

angle_tl = array_from_file(fn+'/angle_tl.txt')
angle_br = array_from_file(fn+'/angle_br.txt')
angle_tr = array_from_file(fn+'/angle_tr.txt')
angle_bl = array_from_file(fn+'/angle_bl.txt')

pos_up = array_from_file(fn+'/position_upper.txt')
pos_lw = array_from_file(fn+'/position_lower.txt')
rad_up = array_from_file(fn+'/radius_upper.txt')
rad_lw = array_from_file(fn+'/radius_lower.txt')

position_tl = pos_up-0.5*rad_up
position_br = pos_lw+0.5*rad_lw
position_tr = pos_up+0.5*rad_up
position_bl = pos_lw-0.5*rad_lw

T0 = 19000
T1 = 22500
T2 = 27000

N0 = np.argmin(np.abs(time-T0))
N1 = np.argmin(np.abs(time-T1))
N2 = np.argmin(np.abs(time-T2))

plt.title('Contact lines')
plt.plot(time, position_tl, label='tl')
plt.plot(time, position_br, label='br')
plt.plot(time, position_tr, label='tr')
plt.plot(time, position_bl, label='bl')
plt.plot([T0,T0], [40.0, 110.000], 'k--')
plt.plot([T1,T1], [40.0, 110.000], 'k--')
plt.plot([T2,T2], [40.0, 110.000], 'k--')
plt.legend()
plt.show()

plt.title(r'Contact angles ($\cos\theta_0-\cos\theta$)')
plt.plot(time, cos(theta_0)-cos(angle_tl), label='tl')
plt.plot(time, cos(theta_0)-cos(angle_br), label='br')
plt.plot(time, cos(theta_0)-cos(angle_tr), label='tr')
plt.plot(time, cos(theta_0)-cos(angle_bl), label='bl')
plt.plot([T0,T0], [-0.5, 0.500], 'k--')
plt.plot([T1,T1], [-0.5, 0.500], 'k--')
plt.plot([T2,T2], [-0.5, 0.500], 'k--')
plt.legend()
plt.show()

cos_diff = cos(theta_0)-cos(angle_bl)
cd1  = cos_diff[N0:N1]
cd2  = cos_diff[N1:N2]
pos1 = position_bl[N0:N1]
pos2 = position_bl[N1:N2]

cd1_avg = np.mean( cd1 )
cd2_avg = np.mean( cd2 )

p1 = np.polyfit(time[N0:N1], pos1, 1)
p2 = np.polyfit(time[N1:N2], pos2, 1)

v1 = p1[0]
v2 = p2[0]

mu = 8.77e-4
gamma = 5.78e-2
Ca1 = -capillary_number + (mu*v1*1e3)/gamma
Ca2 = -capillary_number + (mu*v2*1e3)/gamma

print('Ca1 = '+str(Ca1))
print('Ca2 = '+str(Ca2))

mu_f1 = cd1_avg/Ca1
mu_f2 = cd2_avg/Ca2

print('mu_f1 = '+str(mu_f1))
print('mu_f2 = '+str(mu_f2))

plt.title('Receding contact line (bottom-left)')
plt.plot(time, position_bl, 'k--', linewidth=1.25)
plt.plot(time[N0:N1], np.polyval(p1, time[N0:N1]), 'r-', linewidth=2.5)
plt.plot(time[N1:N2], np.polyval(p2, time[N1:N2]), 'b-', linewidth=2.5)
plt.xlabel('t [ps]')
plt.ylabel('x [nm]')
plt.legend()
plt.show()

plt.title('Receding contact angle (cosine difference, left)')
plt.plot(time, cos_diff, 'k--', linewidth=1.0)
plt.plot(time[N0:N1], cd1_avg*np.ones(time[N0:N1].shape), 'r-', linewidth=1.75)
plt.plot(time[N1:N2], cd2_avg*np.ones(time[N1:N2].shape), 'b-', linewidth=1.75)
plt.legend()
plt.show()
