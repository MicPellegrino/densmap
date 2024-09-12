import densmap as dm

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

from scipy.optimize import curve_fit

import pandas as pd
import seaborn as sn

def array_from_file( filename ):
    my_list = []
    with open(filename, 'r') as f:
        for line in f:
            my_list.append(float(line.split()[0]))
    return np.array(my_list)

# folder_label="EquilWallOffset"
# folder_label="EquilNewRestraints"
# folder_label="EquilNoWallInt"
# folder_label="Q2"
# folder_label ="QT"
# folder_label = "Petter"

folder_label = "NeoQ2"

time = array_from_file('/home/michele/densmap/ShearDropModes/'+folder_label+'/time.txt')

mean_angle = np.zeros(5)
std_angle = np.zeros(5)

dt = 12.5
# t_0 = 1000    # Q1
t_0 = 2000    # Q2
# t_0 = 4000    # Q3
# t_0 = 5000    # Q4
# t_0 = 7000    # Q5
idx_0 = np.abs( time-t_0 ).argmin()

tl = array_from_file('/home/michele/densmap/ShearDropModes/'+folder_label+'/angle_bl.txt')[idx_0:]
tr = array_from_file('/home/michele/densmap/ShearDropModes/'+folder_label+'/angle_br.txt')[idx_0:]
bl = array_from_file('/home/michele/densmap/ShearDropModes/'+folder_label+'/angle_tl.txt')[idx_0:]
br = array_from_file('/home/michele/densmap/ShearDropModes/'+folder_label+'/angle_tr.txt')[idx_0:]

# Computing autocorrelation function for the contact angle
theta_q2 = 0.25 * ( tl + tr + bl + br )
# theta_q2 = br
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

print("Time constrant: tau="+str(1.0/popt[1])+"ps")

mean_angle_2 = mean_angle**2
"""
plt.plot(tau, acf/mean_angle_2, 'b-', linewidth=2.00, label='ACF')
plt.plot(tau, np.ones(tau.shape), 'r--', linewidth=2.00, label='mean^2='+'{:.2f}'.format(mean_angle_2)+"deg^2")
plt.plot(tau, 1.0 + popt[0]*np.exp(-popt[1]*tau)/mean_angle_2, 'k-.', linewidth=2.00, label='exp. fit')
plt.title('Autocorrelation function', fontsize=30.0)
plt.xlabel('time [ps]', fontsize=20.0)
plt.ylabel(r'ACF/$<\theta_0>^2$ [-1]', fontsize=30.0)
plt.legend(fontsize=20.0)
plt.xlim([tau[0], tau[-1]])
plt.xticks(fontsize=15.0)
plt.yticks(fontsize=15.0)
plt.show()
"""

# Averaging
array = dict()
mean = dict()
std = dict()
bins = dict()
dist = dict()
labels = ['tl', 'tr', 'bl', 'br']
cols = dict()
cols['tl'] = 'b-'
cols['tr'] = 'r-'
cols['bl'] = 'c-'
cols['br'] = 'm-'
plt.figure()
for l in labels :
    array[l] = array_from_file('/home/michele/densmap/ShearDropModes/'+folder_label+'/angle_'+l+'.txt')
    mean[l], std[l], bins[l], dist[l] = dm.position_distribution(array[l][idx_0:], int(np.sqrt(len(array[l][idx_0:]))))
    plt.step(bins[l], dist[l], cols[l], label=l.upper()+r', $<\theta>$='+"{:.3f}".format(mean[l])+"deg", linewidth=7.5)
plt.title('Equilibrium c. a. distribution', fontsize=40.0)
plt.xlabel(r'$\theta-<\theta>$ [deg]', fontsize=37.5)
plt.ylabel('pdf', fontsize=37.5)
plt.xticks(fontsize=35.0)
plt.yticks(fontsize=35.0)
plt.legend(fontsize=37.5)
plt.show()

theta_eq = 0.25 * ( tl + tr + bl + br )
mean_angle = theta_eq.mean()
std_angle = theta_eq.std()
print(folder_label)
# print("Eq. c.a. = "+str(mean_angle)+" +/- "+str(std_angle/np.sqrt(len(theta_eq))))
print("<theta>="+str(mean_angle)+"; std(theta)="+str(std_angle)+"; err(theta)="+str(std_angle/np.sqrt(len(theta_eq))))

for l in labels :
    plt.plot(time, array[l], cols[l], label=l+', mean='+"{:.3f}".format(mean[l])+"deg")
plt.title('Equilibrium c. a. signal', fontsize=40.0)
plt.xlabel('time [ps]', fontsize=37.5)
plt.ylabel('angle [deg]', fontsize=37.5)
plt.xticks(fontsize=35.0)
plt.yticks(fontsize=35.0)
plt.legend(fontsize=37.5)
plt.xlim([time[0], time[-1]])
plt.show()

# Correlation matrix
"""
data = {'TL': array['tl'][idx_0:],
        'BL': array['bl'][idx_0:],
        'TR': array['tr'][idx_0:],
        'BR': array['br'][idx_0:] }
df = pd.DataFrame(data,columns=['TL','BL','TR','BR'])
covMatrix = pd.DataFrame.cov(df)
sn.set(font_scale=2.5)
sn.heatmap(covMatrix, annot=True, fmt='.3g', cmap="YlGnBu")
plt.title('Covariance matrix for equilibrium c. a. [deg^2]', fontsize=40.0)
plt.axis('equal')
plt.show()
"""