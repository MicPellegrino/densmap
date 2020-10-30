"""
    OLD (and probably stupid)
"""

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy.ndimage as smg
import scipy.signal as sgn

def array_from_file( filename ):
    my_list = []
    with open(filename, 'r') as f:
        for line in f:
            my_list.append(float(line.split()[0]))
    return np.array(my_list)

def cos( theta ) :
    return np.cos(np.deg2rad(theta))

"""
    Following the approach by Xia and Steen, one may also plot:
    1) Velocity (scaled) vs position (scaled)
    2) C.A. vs position
    3) Position*C.A. vs position*velocity
"""

# Obtaining the signal from saved .txt files
folder_name = 'SpreadingData/H2Q4'
time = array_from_file(folder_name+'/time.txt')
foot_l = array_from_file(folder_name+'/foot_l.txt')
foot_r = array_from_file(folder_name+'/foot_r.txt')
angle_l = array_from_file(folder_name+'/angle_l.txt')
angle_r = array_from_file(folder_name+'/angle_r.txt')
radius = array_from_file(folder_name+'/radius_fit.txt')

# Definition of phase and projective area
h = 5.570423    # [nm]
k = 0.179520    # [nm^-1]
a = h*k
Ly = 4.67654    # [nm]
phi_0 = k*Ly
area_proj = lambda x : np.sqrt( 1.0 + (a**2)*(np.cos(k*x+Ly)**2) )
alpha_sub = lambda x : -np.arctan(a*np.cos(k*x+Ly))

# Cutoff inertial phase
dt = time[1]-time[0]
T_cut = 1000.0      # [ps]
N = int( T_cut / dt )
time = time[N:]
foot_l = foot_l[N:]
foot_r = foot_r[N:]
angle_l = angle_l[N:]
angle_r = angle_r[N:]

# Velocity from raw data (very noisy)
velocity_l = np.zeros(len(foot_l))
velocity_r = np.zeros(len(foot_r))
velocity_l[1:-1] = -0.5 * np.subtract(foot_l[2:],foot_l[0:-2]) / dt
velocity_r[1:-1] = 0.5 * np.subtract(foot_r[2:],foot_r[0:-2]) / dt
velocity_l[0] = -( foot_l[1]-foot_l[0] ) / dt
velocity_r[0] = ( foot_r[1]-foot_r[0] ) / dt
velocity_l[-1] = -( foot_l[-1]-foot_l[-2] ) / dt
velocity_r[-1] = ( foot_r[-1]-foot_r[-2] ) / dt

# Manually-tuned filter
sigma_pos = 5.0
foot_l_filtered = smg.gaussian_filter1d(foot_l, sigma=sigma_pos)
foot_r_filtered = smg.gaussian_filter1d(foot_r, sigma=sigma_pos)
sigma_ang = 5.0
angle_l_filtered = smg.gaussian_filter1d(angle_l, sigma=sigma_ang)
angle_r_filtered = smg.gaussian_filter1d(angle_r, sigma=sigma_ang)

# Velocity from filtered data (less noisy, hopefully)
velocity_l_filtered = np.zeros(len(foot_l_filtered))
velocity_r_filtered = np.zeros(len(foot_r_filtered))
velocity_l_filtered[1:-1] = -0.5 * np.subtract(foot_l_filtered[2:],foot_l_filtered[0:-2]) / dt
velocity_r_filtered[1:-1] = 0.5 * np.subtract(foot_r_filtered[2:],foot_r_filtered[0:-2]) / dt
velocity_l_filtered[0] = -( foot_l_filtered[1]-foot_l_filtered[0] ) / dt
velocity_r_filtered[0] = ( foot_r_filtered[1]-foot_r_filtered[0] ) / dt
velocity_l_filtered[-1] = -( foot_l_filtered[-1]-foot_l_filtered[-2] ) / dt
velocity_r_filtered[-1] = ( foot_r_filtered[-1]-foot_r_filtered[-2] ) / dt

# Apply are projector to velocity dx/dt -> ds/dt
velocity_l_micro = velocity_l * area_proj(foot_l)
velocity_r_micro = velocity_r * area_proj(foot_r)
velocity_l_micro_filtered = smg.gaussian_filter1d(velocity_l_micro, sigma=sigma_pos)
velocity_r_micro_filtered = smg.gaussian_filter1d(velocity_r_micro, sigma=sigma_pos)

# Obtain the microscopic contact angle
angle_l_micro = angle_l - alpha_sub(foot_l)
angle_r_micro = angle_r - alpha_sub(foot_r)
angle_l_micro_filtered = smg.gaussian_filter1d(angle_l_micro, sigma=sigma_ang)
angle_r_micro_filtered = smg.gaussian_filter1d(angle_r_micro, sigma=sigma_ang)

#########
# PLOTS #
#########

"""
plt.title('Spreading branches (raw value and filtered)')
plt.plot(time, foot_l, 'b-', linewidth=0.75, label='left (raw)')
plt.plot(time, foot_r, 'r-', linewidth=0.75, label='right (raw)')
plt.plot(time, foot_l_filtered, 'b--', linewidth=2.0, label='left (fil.)')
plt.plot(time, foot_r_filtered, 'r--', linewidth=2.0, label='right (fil.)')
plt.xlabel('t [ps]', fontsize=20.0)
plt.ylabel('x [nm]', fontsize=20.0)
plt.ylim([0.0, 175.0])
plt.xlim([time[0], time[-1]])
plt.legend(fontsize=20.0)
plt.show()
"""

"""
plt.title('Apparent wetting speed (raw value and filtered)')
plt.plot(time, velocity_l, 'b-', linewidth=0.75)
plt.plot(time, velocity_r, 'r-', linewidth=0.75)
plt.plot(time, velocity_l_filtered, 'b--', linewidth=2.0)
plt.plot(time, velocity_r_filtered, 'r--', linewidth=2.0)
plt.xlabel('t [ps]', fontsize=20.0)
plt.ylabel('dx/dt [nm/ps]', fontsize=20.0)
plt.xlim([time[0], time[-1]])
plt.show()
"""

"""
plt.title('Apparent wetting contact angles')
plt.plot(time, angle_l, 'b-', linewidth=0.75)
plt.plot(time, angle_r, 'r-', linewidth=0.75)
plt.plot(time, angle_l_filtered, 'b--', linewidth=2.0)
plt.plot(time, angle_r_filtered, 'r--', linewidth=2.0)
plt.xlabel('t [ps]', fontsize=20.0)
plt.ylabel(r'$\theta$ [deg]', fontsize=20.0)
plt.show()
"""

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
X_mid = 85.7    # [mn]
D0 = 50.0       # [nm]
T0 = 2000.0     # [ps]
pos_l_scaled = (X_mid-foot_l_filtered)/D0
pos_r_scaled = (foot_r_filtered-X_mid)/D0
vel_l_scaled = T0*velocity_l_micro_filtered/D0
vel_r_scaled = T0*velocity_r_micro_filtered/D0

plt.title('Position vs velocity')
plt.plot(pos_l_scaled, vel_l_scaled, 'bx', label='left')
plt.plot(pos_r_scaled, vel_r_scaled, 'ro', label='right')
plt.xlabel('$x/D_0$ [-1]', fontsize=20.0)
plt.ylabel('$(dx/dt)/(D_0/T_0)$ [-1]', fontsize=20.0)
plt.legend(fontsize=20.0)
plt.show()

plt.title('Contact angle vs position')
plt.plot(pos_l_scaled, cos(angle_l_micro_filtered), 'bx', label='left')
plt.plot(pos_r_scaled, cos(angle_r_micro_filtered), 'ro', label='right')
plt.xlabel('$x/D_0$ [-1]', fontsize=20.0)
plt.ylabel(r'$\cos(\theta)$ [-1]', fontsize=20.0)
plt.legend(fontsize=20.0)
plt.show()

plt.title('Xia-Steen-like plot')
plt.plot(pos_l_scaled*cos(angle_l_micro_filtered), pos_l_scaled*vel_l_scaled, 'bx', label='left')
plt.plot(pos_r_scaled*cos(angle_r_micro_filtered), pos_r_scaled*vel_r_scaled, 'ro', label='right')
plt.xlabel(r'$(dx/dt)x/(D_0^2/T_0)$ [-1]', fontsize=20.0)
plt.ylabel(r'$\cos(\theta)x/D_0$ [-1]', fontsize=20.0)
plt.legend(fontsize=20.0)
plt.show()

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
