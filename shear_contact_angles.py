import densmap as dm

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

from scipy.optimize import curve_fit

def array_from_file( filename ):
    my_list = []
    with open(filename, 'r') as f:
        for line in f:
            my_list.append(float(line.split()[0]))
    return np.array(my_list)

time = array_from_file('/home/michele/densmap/ShearDropModes/Q2/time.txt')

mean_angle = np.zeros(5)
std_angle = np.zeros(5)

dt = 12.5
t_0 = 3000
idx_0 = np.abs( time-t_0 ).argmin()

"""

theta = []
for q in range(5) :
    tl = array_from_file('/home/michele/densmap/ShearDropModes/Q'+str(q+1)+'/angle_bl.txt')
    tr = array_from_file('/home/michele/densmap/ShearDropModes/Q'+str(q+1)+'/angle_br.txt')
    bl = array_from_file('/home/michele/densmap/ShearDropModes/Q'+str(q+1)+'/angle_tl.txt')
    br = array_from_file('/home/michele/densmap/ShearDropModes/Q'+str(q+1)+'/angle_tr.txt')
    theta.append( 0.25 * ( tl + tr + bl + br ) )
    mean_angle[q] = theta[-1][idx_0:].mean()
    std_angle[q] = theta[-1][idx_0:].std()

var_angle = std_angle*std_angle

print("Charge vs contact angle:")
for q in range(5) :
    print("q:"+str(q+1)+"\tmean="+str(mean_angle[q])+"\tstd="+str(std_angle[q]))

"""

# Computing autocorrelation function for q2
tl = array_from_file('/home/michele/densmap/ShearDropModes/Q2/angle_bl.txt')[idx_0:]
tr = array_from_file('/home/michele/densmap/ShearDropModes/Q2/angle_br.txt')[idx_0:]
bl = array_from_file('/home/michele/densmap/ShearDropModes/Q2/angle_tl.txt')[idx_0:]
br = array_from_file('/home/michele/densmap/ShearDropModes/Q2/angle_tr.txt')[idx_0:]
theta_q2 = 0.25 * ( tl + tr + bl + br )
mean_angle = theta_q2.mean()
std_angle = theta_q2.std()
theta_q2 = theta_q2
acf = np.zeros(theta_q2.shape)
N = len(theta_q2)
for k in range(N) :
    # Non-cyclic
    # acf[k] = np.sum(theta_q2[:N-k]*theta_q2[k:])/(N-k)
    # Cyclic
    acf[k] = np.sum(theta_q2*np.roll(theta_q2, k))/N
    # acf[k] = ( np.sum(tl*np.roll(tl, k)) + np.sum(tr*np.roll(tr, k)) + \
    #     np.sum(bl*np.roll(bl, k)) + np.sum(br*np.roll(br, k)) ) / (4*N)
acf = acf[:len(acf)//2]

# acf = np.correlate(theta1, theta1, 'full')
# acf = acf[acf.size//2:]

# plt.plot(time[idx_0:], theta[1][idx_0:])
tau = time[idx_0:]-t_0
tau = tau[0:len(tau)//2]

# I am lazy
def func(t, a1, a2) :
    return a1*np.exp(-a2*t)+mean_angle**2
popt, pcov = curve_fit(func, tau, acf, p0=(4,0.05))

plt.plot(tau, acf, 'b-')
plt.plot(tau, np.ones(tau.shape)*mean_angle**2, 'r--')
plt.plot(tau, mean_angle**2 + popt[0]*np.exp(-popt[1]*tau), 'k-.')
plt.show()

# Lennard-Jones
""""
time = array_from_file('/home/michele/densmap/ShearChar/LJ/time.txt')
tl = array_from_file('/home/michele/densmap/ShearChar/LJ/angle_bl.txt')
tr = array_from_file('/home/michele/densmap/ShearChar/LJ/angle_br.txt')
bl = array_from_file('/home/michele/densmap/ShearChar/LJ/angle_tl.txt')
br = array_from_file('/home/michele/densmap/ShearChar/LJ/angle_tr.txt')
theta_lj = 0.25 * ( tl + tr + bl + br )
print("LJ\tmean="+str(theta_lj.mean())+"\tstd="+str(theta_lj.std()))
"""
