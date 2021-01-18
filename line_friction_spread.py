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

# Obtaining the signal from saved .txt files
folder_name = 'SpreadingData/FlatQ3'
time = array_from_file(folder_name+'/time.txt')
foot_l = array_from_file(folder_name+'/foot_l.txt')
foot_r = array_from_file(folder_name+'/foot_r.txt')
angle_l = array_from_file(folder_name+'/angle_l.txt')
angle_r = array_from_file(folder_name+'/angle_r.txt')
radius = array_from_file(folder_name+'/radius_fit.txt')

# Cutoff inertial phase
dt = time[1]-time[0]
T_cut = 200.0      # [ps]
N = int( T_cut / dt )
time = time[N:]
foot_l = foot_l[N:]
foot_r = foot_r[N:]
angle_l = angle_l[N:]
angle_r = angle_r[N:]

# Rational polynomial fit
def rational(x, p, q):
    return np.polyval(p, x) / np.polyval(q + [1.0], x)

def rational_3_3(x, p0, p1, p2, q1, q2):
    return rational(x, [p0, p1, p2], [q1, q2])

popt_l, pcov_l = opt.curve_fit(rational_3_3, time, foot_l)
popt_r, pcov_r = opt.curve_fit(rational_3_3, time, foot_r)

# Plot spreading radius
plt.title('Spreading branches (raw value and filtered)', fontsize=20.0)
plt.plot(time, foot_l, 'b-', linewidth=1.0, label='left (raw)')
plt.plot(time, foot_r, 'r-', linewidth=1.0, label='right (raw)')
plt.plot(time, rational_3_3(time, *popt_l), 'b--', linewidth=2.0, label='left (fit)')
plt.plot(time, rational_3_3(time, *popt_r), 'r--', linewidth=2.0, label='right (fit)')
plt.xlabel('t [ps]', fontsize=20.0)
plt.ylabel('x [nm]', fontsize=20.0)
plt.ylim([0.0, 175.0])
plt.xlim([time[0], time[-1]])
plt.legend(fontsize=15.0)
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
pl_0 = np.array(popt_l[0:3])
pr_0 = np.array(popt_r[0:3])
pl_1 = np.polyder(pl_0, m=1)
pr_1 = np.polyder(pr_0, m=1)
ql_0 = np.concatenate((popt_l[3:5],[1.0]))
qr_0 = np.concatenate((popt_r[3:5],[1.0]))
ql_1 = np.polyder(ql_0, m=1)
qr_1 = np.polyder(qr_0, m=1)
def velocity_l_fit(t) :
    num = ( np.polyval(pl_1,t)*np.polyval(ql_0,t) - np.polyval(pl_0,t)*np.polyval(ql_1,t) )
    den = ( np.polyval(ql_0,t) )**2
    return -num/den
def velocity_r_fit(t) :
    num = ( np.polyval(pr_1,t)*np.polyval(qr_0,t) - np.polyval(pr_0,t)*np.polyval(qr_1,t) )
    den = ( np.polyval(qr_0,t) )**2
    return num/den
velocity_l_fit = velocity_l_fit(time)
velocity_r_fit = velocity_r_fit(time)

# Plot velocity
plt.title('Apparent wetting speed (raw value and filtered)', fontsize=20.0)
plt.plot(time, velocity_l, 'b-', linewidth=1.0, label='left (raw)')
plt.plot(time, velocity_r, 'r-', linewidth=1.0, label='right (raw)')
plt.plot(time, velocity_l_fit, 'b--', linewidth=2.0, label='left (fit)')
plt.plot(time, velocity_r_fit, 'r--', linewidth=2.0, label='right (fit)')
plt.xlabel('t [ps]', fontsize=20.0)
plt.ylabel('dx/dt [nm/ps]', fontsize=20.0)
plt.xlim([time[0], time[-1]])
plt.legend(fontsize=15.0)
plt.show()

# Plot contact angles
plt.title('Apparent wetting contact angles', fontsize=20.0)
plt.plot(time, angle_l, 'b-', linewidth=1.25)
plt.plot(time, angle_r, 'r-', linewidth=1.25)
plt.xlabel('t [ps]', fontsize=20.0)
plt.ylabel(r'$\theta$ [deg]', fontsize=20.0)
plt.show()

# Transform into cos and plot againts velocity
plt.title(r'Linear regression: $U\sim-a\cdot\cos(\theta)+b$', fontsize=20.0)
cos_l = np.cos(np.deg2rad(angle_l))
cos_r = np.cos(np.deg2rad(angle_r))
coef_l = np.polyfit(-cos_l, velocity_l_fit, 1)
coef_r = np.polyfit(-cos_r, velocity_r_fit, 1)
cos_range = np.linspace(-0.75, 0.50, 10)
plt.plot(cos_l, velocity_l_fit, 'b.')
plt.plot(cos_r, velocity_r_fit, 'r.')
plt.plot(cos_range, np.polyval(coef_l, -cos_range), 'b-', linewidth=1.75)
plt.plot(cos_range, np.polyval(coef_r, -cos_range), 'r-', linewidth=1.75)
plt.xlabel(r'$cos(\theta(t))$ [-1]', fontsize=20.0)
plt.ylabel(r'$U$ [nm/ps]', fontsize=20.0)
plt.show()

########
# MISC #
########

"""
plt.title('Correlation between c.a. and c.l. speed')
plt.plot(cos(angle_l_filtered), velocity_l_filtered, 'bx')
plt.plot(cos(angle_r_filtered), velocity_r_filtered, 'ro')
plt.xlabel(r'$cos(\theta)$ [-1]', fontsize=20.0)
plt.ylabel('dx/dt [nm/ps]', fontsize=20.0)
plt.show()
"""

"""
plt.title('Microscopic wetting speed (raw value and filtered)')
plt.plot(time, velocity_l_micro, 'b-', linewidth=0.75, label='left (raw)')
plt.plot(time, velocity_r_micro, 'r-', linewidth=0.75, label='right (raw)')
plt.plot(time, velocity_l_micro_filtered, 'b--', linewidth=2.0, label='left (fil.)')
plt.plot(time, velocity_r_micro_filtered, 'r--', linewidth=2.0, label='right (fil.)')
plt.xlabel('t [ps]', fontsize=20.0)
plt.ylabel('ds/dt [nm/ps]', fontsize=20.0)
plt.xlim([time[0], time[-1]])
plt.legend(fontsize=20.0)
plt.show()
"""

"""
plt.title('Microscopic wetting contact angles')
plt.plot(time, angle_l_micro, 'b-', linewidth=0.75, label='left (raw)')
plt.plot(time, angle_r_micro, 'r-', linewidth=0.75, label='right (raw)')
plt.plot(time, angle_l_micro_filtered, 'b--', linewidth=2.0, label='left (fil.)')
plt.plot(time, angle_r_micro_filtered, 'r--', linewidth=2.0, label='right (fil.)')
plt.xlabel('t [ps]', fontsize=20.0)
plt.ylabel(r'$\theta$ [deg]', fontsize=20.0)
plt.xlim([time[0], time[-1]])
plt.legend(fontsize=20.0)
plt.show()
"""

"""
plt.title('Correlation between c.a. and c.l. speed')
plt.plot(cos(angle_l_micro_filtered), velocity_l_micro_filtered, 'bx', label='left')
plt.plot(cos(angle_r_micro_filtered), velocity_r_micro_filtered, 'ro', label='right')
plt.xlabel(r'$cos(\theta)$ [-1]', fontsize=20.0)
plt.ylabel('dx/dt [nm/ps]', fontsize=20.0)
plt.legend(fontsize=20.0)
plt.show()
"""

# Nondim.
"""
X_mid = 85.7    # [mn]
D0 = 50.0       # [nm]
T0 = 2000.0     # [ps]
pos_l_scaled = (X_mid-foot_l_filtered)/D0
pos_r_scaled = (foot_r_filtered-X_mid)/D0
vel_l_scaled = T0*velocity_l_micro_filtered/D0
vel_r_scaled = T0*velocity_r_micro_filtered/D0
"""

"""
plt.title('Position vs velocity')
plt.plot(pos_l_scaled, vel_l_scaled, 'bx', label='left')
plt.plot(pos_r_scaled, vel_r_scaled, 'ro', label='right')
plt.xlabel('$x/D_0$ [-1]', fontsize=20.0)
plt.ylabel('$(dx/dt)/(D_0/T_0)$ [-1]', fontsize=20.0)
plt.legend(fontsize=20.0)
plt.show()
"""

"""
plt.title('Contact angle vs position')
plt.plot(pos_l_scaled, cos(angle_l_micro_filtered), 'bx', label='left')
plt.plot(pos_r_scaled, cos(angle_r_micro_filtered), 'ro', label='right')
plt.xlabel('$x/D_0$ [-1]', fontsize=20.0)
plt.ylabel(r'$\cos(\theta)$ [-1]', fontsize=20.0)
plt.legend(fontsize=20.0)
plt.show()
"""

"""
plt.title('Xia-Steen-like plot')
plt.plot(pos_l_scaled*cos(angle_l_micro_filtered), pos_l_scaled*vel_l_scaled, 'bx', label='left')
plt.plot(pos_r_scaled*cos(angle_r_micro_filtered), pos_r_scaled*vel_r_scaled, 'ro', label='right')
plt.xlabel(r'$(dx/dt)x/(D_0^2/T_0)$ [-1]', fontsize=20.0)
plt.ylabel(r'$\cos(\theta)x/D_0$ [-1]', fontsize=20.0)
plt.legend(fontsize=20.0)
plt.show()
"""

#########################
# STUDY NOISE STRUCTURE #
#########################

"""
f_l, P_l = sgn.periodogram(foot_l, detrend='constant')
f_r, P_r = sgn.periodogram(foot_r, detrend='constant')
f_l = f_l[1:-1]
f_r = f_r[1:-1]
P_l = P_l[1:-1]
P_r = P_r[1:-1]
plt.semilogy(f_l, P_l, 'b-')
plt.semilogy(f_r, P_r, 'r-')
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.title('Radius (P.S.)')
plt.show()
"""
