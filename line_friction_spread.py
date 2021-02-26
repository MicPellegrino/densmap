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
theta_0 = 38.8              # deg

# Obtaining the signal from saved .txt files
folder_name = 'SpreadingData/FlatQ5'
time = array_from_file(folder_name+'/time.txt')
foot_l = array_from_file(folder_name+'/foot_l.txt')
foot_r = array_from_file(folder_name+'/foot_r.txt')
angle_l = array_from_file(folder_name+'/angle_l.txt')
angle_r = array_from_file(folder_name+'/angle_r.txt')
radius = array_from_file(folder_name+'/radius_fit.txt')

# Cutoff inertial phase
dt = time[1]-time[0]
T_ini = 500.0                  # [ps]
T_fin = 5000.0
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

# Rational polynomial fit
def rational(x, p, q):
    return np.polyval(p, x) / np.polyval(q + [1.0], x)

def rational_3_3(x, p0, p1, p2, q1, q2):
    return rational(x, [p0, p1, p2], [q1, q2])

popt_l, pcov_l = opt.curve_fit(rational_3_3, time, foot_l)
popt_r, pcov_r = opt.curve_fit(rational_3_3, time, foot_r)

R0 = 25.0   # [nm]
diam = foot_r-foot_l
# t_90deg = T_cut + dt*np.argmin(np.abs(diam-2.0*np.sqrt(2.0)*R0))
# print("Time 90 deg = "+str(t_90deg))

# Plot spreading radius
plt.title('Spreading branches (raw value and filtered)', fontsize=30.0)
plt.plot(time, foot_l, 'b-', linewidth=1.0, label='left (raw)')
plt.plot(time, foot_r, 'r-', linewidth=1.0, label='right (raw)')
plt.plot(time, foot_r-foot_l, 'k-', linewidth=1.75, label='wetted area')
plt.plot(time, 2.0*np.sqrt(2.0)*R0*np.ones(time.shape), 'g--', linewidth=1.75, label='wetted area')
plt.plot(time, rational_3_3(time, *popt_l), 'b--', linewidth=2.5, label='left (fit)')
plt.plot(time, rational_3_3(time, *popt_r), 'r--', linewidth=2.5, label='right (fit)')
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
plt.title('Apparent wetting speed (raw value and filtered)', fontsize=30.0)
plt.plot(time, velocity_l, 'b-', linewidth=1.0, label='left (raw)')
plt.plot(time, velocity_r, 'r-', linewidth=1.0, label='right (raw)')
plt.plot(time, velocity_l_fit, 'b--', linewidth=2.5, label='left (fit)')
plt.plot(time, velocity_r_fit, 'r--', linewidth=2.5, label='right (fit)')
plt.xlabel('t [ps]', fontsize=30.0)
plt.ylabel('dx/dt [nm/ps]', fontsize=30.0)
plt.xlim([time[0], time[-1]])
plt.legend(fontsize=20.0)
plt.xticks(fontsize=20.0)
plt.yticks(fontsize=20.0)
plt.show()

# Rational approximation of c.a.
popt_l, pcov_l = opt.curve_fit(rational_3_3, time, angle_l)
popt_r, pcov_r = opt.curve_fit(rational_3_3, time, angle_r)
angle_l_fit = rational_3_3(time, *popt_l)
angle_r_fit = rational_3_3(time, *popt_r)

# Plot contact angles
plt.title('Apparent wetting contact angles', fontsize=30.0)
plt.plot(time, angle_l, 'b-', linewidth=1.0, label='left (raw)')
plt.plot(time, angle_r, 'r-', linewidth=1.0, label='left (raw)')
plt.plot(time, angle_l_fit, 'b--', linewidth=2.5, label='left (fit)')
plt.plot(time, angle_r_fit, 'r--', linewidth=2.5, label='right (fit)')
plt.plot(time, 90.0*np.ones(time.shape), 'g--', linewidth=3.75, )
plt.xlabel('t [ps]', fontsize=30.0)
plt.ylabel(r'$\theta$ [deg]', fontsize=30.0)
plt.xlim([time[0], time[-1]])
plt.legend(fontsize=20.0)
plt.xticks(fontsize=20.0)
plt.yticks(fontsize=20.0)
plt.show()


# Rescale speed
velocity_l_fit_red = velocity_l_fit/U_ref
velocity_r_fit_red = velocity_r_fit/U_ref

# Transform into cos and plot againts velocity
plt.title(r'Linear regression: $U\sim-a\cdot\cos(\theta)+b$', fontsize=30.0)
cos_l = cos_vec(angle_l)
cos_r = cos_vec(angle_r)
coef_l = np.polyfit(cos_l, velocity_l_fit_red, 1)
coef_r = np.polyfit(cos_r, velocity_r_fit_red, 1)
cos_range = np.linspace(max(max(cos_l),max(cos_r)), min(min(cos_l),min(cos_r)), 10)
plt.plot(cos_l, velocity_l_fit_red, 'b.')
plt.plot(cos_r, velocity_r_fit_red, 'r.')
plt.plot(cos_range, np.polyval(coef_l, cos_range), 'b-', linewidth=2.0, label='linfit (left)')
plt.plot(cos_range, np.polyval(coef_r, cos_range), 'r-', linewidth=2.0, label='linfit (right)')
plt.xlabel(r'$cos(\theta(t))$ [-1]', fontsize=30.0)
plt.ylabel(r'$U/U_{ref}$ [-1]', fontsize=30.0)
plt.legend(fontsize=20.0)
plt.xticks(fontsize=20.0)
plt.yticks(fontsize=20.0)
plt.show()

# Contact line friction estimate
mu_st = np.zeros(4)
mu_st[0] = -1.0/coef_l[0]
mu_st[1] = -1.0/coef_r[0]
mu_st[2] = cos(theta_0)/coef_l[1]
mu_st[3] = cos(theta_0)/coef_r[1]
print("mu* (left, slope)  = "+str( mu_st[0] ))
print("mu* (right, slope) = "+str( mu_st[1] ))
print("mu* (left, inter)  = "+str( mu_st[2] ))
print("mu* (right, inter) = "+str( mu_st[3] ))
print("mu* (best fit)     = "+str( np.mean(mu_st) ))

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
