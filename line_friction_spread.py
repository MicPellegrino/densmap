"""
    Rational approximation
"""

import numpy as np

import matplotlib as mpl
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

def cos( theta ) :
    return np.cos(np.deg2rad(theta))

cos_vec = np.vectorize(cos)

# Reference units
mu = 0.877                  # mPa*s
gamma = 57.8                # mPa*m
U_ref = (gamma/mu)*1e-3     # nm/ps
theta_0 = 95.0              # deg

# Obtaining the signal from saved .txt files
folder_name = 'SpreadingData/FlatQ2'
time = array_from_file(folder_name+'/time.txt')
foot_l = array_from_file(folder_name+'/foot_l.txt')
foot_r = array_from_file(folder_name+'/foot_r.txt')
angle_l = array_from_file(folder_name+'/angle_l.txt')
angle_r = array_from_file(folder_name+'/angle_r.txt')
radius = array_from_file(folder_name+'/radius_fit.txt')

# Cutoff inertial phase
dt = time[1]-time[0]
T_ini = 100.0                  # [ps]
T_fin = 10100.0
time_window = T_fin-T_ini
print("Time window = "+str(time_window))
print("Time step   = "+str(dt))
N_ini = int( T_ini / dt )
N_fin = int( T_fin / dt )
time = time[N_ini:N_fin]
foot_l = foot_l[N_ini:N_fin]
foot_r = foot_r[N_ini:N_fin]
angle_l = angle_l[N_ini:N_fin]
angle_r = angle_r[N_ini:N_fin]

# Averaging between laft and right c.l.
contact_line_pos = 0.5*(foot_r-foot_l)
contact_angle = 0.5*(angle_r+angle_l)

# Rational polynomial fit
def rational(x, p, q):
    return np.polyval(p, x) / np.polyval(q + [1.0], x)

def rational_3_3(x, p0, p1, p2, q1, q2):
    return rational(x, [p0, p1, p2], [q1, q2])

popt, pcov = opt.curve_fit(rational_3_3, time, contact_line_pos)

# Plot spreading radius
plt.title('Spreading branches (raw value and filtered)', fontsize=30.0)
plt.plot(time, foot_l, 'b-', linewidth=1.75, label='left (raw)')
plt.plot(time, foot_r, 'r-', linewidth=1.75, label='right (raw)')
plt.plot(time, contact_line_pos, 'k-', linewidth=1.75, label='mean (right-left)')

plt.plot(time, rational_3_3(time, *popt), 'k--', linewidth=2.5, label='fit')
plt.xlabel('t [ps]', fontsize=30.0)
plt.ylabel('x [nm]', fontsize=30.0)
plt.ylim([0.0, 175.0])
plt.xlim([time[0], time[-1]])
plt.legend(fontsize=20.0)
plt.xticks(fontsize=20.0)
plt.yticks(fontsize=20.0)
plt.show()

# Velocity from raw data (very noisy)
velocity_l = np.zeros(len(foot_l))
velocity_r = np.zeros(len(foot_r))
velocity_l[1:-1] = -0.5 * np.subtract(foot_l[2:],foot_l[0:-2]) / dt
velocity_r[1:-1] = 0.5 * np.subtract(foot_r[2:],foot_r[0:-2]) / dt
velocity_l[0] = -( foot_l[1]-foot_l[0] ) / dt
velocity_r[0] = ( foot_r[1]-foot_r[0] ) / dt
velocity_l[-1] = -( foot_l[-1]-foot_l[-2] ) / dt
velocity_r[-1] = ( foot_r[-1]-foot_r[-2] ) / dt

# Velocity from rational approximation
p_0 = np.array(popt[0:3])
p_1 = np.polyder(p_0, m=1)
q_0 = np.concatenate((popt[3:5],[1.0]))
q_1 = np.polyder(q_0, m=1)
def velocity_fit(t) :
    num = ( np.polyval(p_1,t)*np.polyval(q_0,t) - np.polyval(p_0,t)*np.polyval(q_1,t) )
    den = ( np.polyval(q_0,t) )**2
    return num/den
velocity_fit = velocity_fit(time)

# Plot velocity
plt.title('Apparent wetting speed (raw value and filtered)', fontsize=30.0)
plt.plot(time, velocity_l, 'b-', linewidth=1.0, label='left (raw)')
plt.plot(time, velocity_r, 'r-', linewidth=1.0, label='right (raw)')
plt.plot(time, velocity_fit, 'k--', linewidth=2.5, label='left (fit)')
plt.xlabel('t [ps]', fontsize=30.0)
plt.ylabel('dx/dt [nm/ps]', fontsize=30.0)
plt.xlim([time[0], time[-1]])
plt.legend(fontsize=20.0)
plt.xticks(fontsize=20.0)
plt.yticks(fontsize=20.0)
plt.show()

# Rolling average
delta_t_avg = 500           # [ps]
N_avg = int(delta_t_avg/dt)
contact_angle = np.convolve(contact_angle, np.ones(N_avg)/N_avg, mode='same')

# Plot contact angles
plt.title('Apparent wetting contact angles', fontsize=30.0)
plt.plot(time, angle_l, 'b-', linewidth=1.0, label='left (raw)')
plt.plot(time, angle_r, 'r-', linewidth=1.0, label='left (raw)')
plt.plot(time[N_avg:-N_avg], contact_angle[N_avg:-N_avg], 'k--', linewidth=2.5, label='left (fit)')
plt.xlabel('t [ps]', fontsize=30.0)
plt.ylabel(r'$\theta$ [deg]', fontsize=30.0)
plt.xlim([time[0], time[-1]])
plt.legend(fontsize=20.0)
plt.xticks(fontsize=20.0)
plt.yticks(fontsize=20.0)
plt.show()


# Rescale speed
velocity_fit_red = velocity_fit[N_avg:-N_avg]/U_ref

# Transform into cos and plot againts velocity
plt.title(r'Linear regression: $U\sim-a\cdot\cos(\theta)+b$', fontsize=30.0)
cos_ca = cos_vec(contact_angle[N_avg:-N_avg])
coef_ca = np.polyfit(cos_ca, velocity_fit_red, 1)
cos_range = np.linspace(max(cos_ca), min(cos_ca), 10)
plt.plot(cos_ca, velocity_fit_red, 'k.')
plt.plot(cos_range, np.polyval(coef_ca, cos_range), 'k-', linewidth=2.0, label='linfit')
plt.xlabel(r'$cos(\theta(t))$ [-1]', fontsize=30.0)
plt.ylabel(r'$U/U_{ref}$ [-1]', fontsize=30.0)
plt.legend(fontsize=20.0)
plt.xticks(fontsize=20.0)
plt.yticks(fontsize=20.0)
plt.show()

# Contact line friction estimate
mu_st = np.zeros(2)
mu_st[0] = -1.0/coef_ca[0]
mu_st[1] = cos(theta_0)/coef_ca[1]
print("mu* (slope)    = "+str( mu_st[0] ))
print("mu* (interc.)  = "+str( mu_st[1] ))
print("mu* (best fit) = "+str( np.mean(mu_st) ))
