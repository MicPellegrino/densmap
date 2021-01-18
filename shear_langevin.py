import densmap as dm

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

from scipy.optimize import curve_fit
from scipy.signal import detrend

def array_from_file( filename ):
    my_list = []
    with open(filename, 'r') as f:
        for line in f:
            my_list.append(float(line.split()[0]))
    return np.array(my_list)

folder_label = "NeoQ1"

time = array_from_file('/home/michele/densmap/ShearDropModes/'+folder_label+'/time.txt')

mean_angle = np.zeros(5)
std_angle = np.zeros(5)

dt = 12.5
t_0 = 1000    # Q1
# t_0 = 2000      # Q2
# t_0 = 4000    # Q3
# t_0 = 5000    # Q4
# t_0 = 7000    # Q5
idx_0 = np.abs( time-t_0 ).argmin()

tl = array_from_file('/home/michele/densmap/ShearDropModes/'+folder_label+'/angle_bl.txt')[idx_0:]
tr = array_from_file('/home/michele/densmap/ShearDropModes/'+folder_label+'/angle_br.txt')[idx_0:]
bl = array_from_file('/home/michele/densmap/ShearDropModes/'+folder_label+'/angle_tl.txt')[idx_0:]
br = array_from_file('/home/michele/densmap/ShearDropModes/'+folder_label+'/angle_tr.txt')[idx_0:]

# Computing autocorrelation function for the c.l. friction force
theta_q = bl
mean_angle = theta_q.mean()
mean_cos = np.cos(np.deg2rad(mean_angle))
force_q = mean_cos - np.cos(np.deg2rad(theta_q))
acf = np.zeros(force_q.shape)
N = len(force_q)
for k in range(N) :
    # Non-cyclic
    # acf[k] = np.sum(force_q[:N-k]*force_q[k:])/(N-k)
    # Cyclic
    acf[k] = np.sum(force_q*np.roll(force_q, k))/N
    # acf[k] = ( np.sum(tl*np.roll(tl, k)) + np.sum(tr*np.roll(tr, k)) + \
    #     np.sum(bl*np.roll(bl, k)) + np.sum(br*np.roll(br, k)) ) / (4*N)
acf = acf[:len(acf)//2]

tau = time[idx_0:]-t_0
tau = tau[0:len(tau)//2]

def func(t, a1, a2) :
    return a1*np.exp(-a2*t)
popt, pcov = curve_fit(func, tau, acf, p0=(4,0.05))

print("Friction force de-correlation time = "+str(5.0/popt[1])+" ps")

mean_cos_2 = mean_cos**2
plt.semilogx(tau, acf, 'b-', linewidth=2.00, label='ACF')
plt.semilogx(tau, np.zeros(tau.shape), 'r--', linewidth=2.00, label=r'cos($\theta_0$)='+str(mean_cos))
plt.semilogx(tau, popt[0]*np.exp(-popt[1]*tau), 'k-.', linewidth=2.00, label='exp. fit')
plt.title('Autocorrelation function of $f=cos_0-cos$', fontsize=20.0)
plt.xlabel('time [ps]', fontsize=20.0)
plt.ylabel('ACF [-1]', fontsize=20.0)
plt.legend()
plt.xlim([tau[0], tau[-1]])
plt.show()

cl_pos = array_from_file('/home/michele/densmap/ShearDropModes/'+folder_label+'/position_upper.txt')[idx_0:] \
        + 0.5*array_from_file('/home/michele/densmap/ShearDropModes/'+folder_label+'/radius_upper.txt')[idx_0:]
cl_pos = detrend(cl_pos)
mean_cl = cl_pos.mean()
# disp = cl_pos-mean_cl
disp = cl_pos
acf = np.zeros(disp.shape)
N = len(disp)
for k in range(N) :
    acf[k] = np.sum(disp*np.roll(disp, k))/N
acf = acf[:len(acf)//2]

plt.plot(tau, acf, 'b-', linewidth=2.00, label='ACF')
plt.plot(tau, np.ones(tau.shape)*mean_cl**2, 'r--', linewidth=2.00, label='x_0='+str(mean_cl))
plt.title('Position autocorrelation function $x-x_0$', fontsize=20.0)
plt.xlabel('time [ps]', fontsize=20.0)
plt.ylabel('ACF [nm^2]', fontsize=20.0)
plt.legend()
plt.xlim([tau[0], tau[-1]])
plt.show()

# To be continued ...
