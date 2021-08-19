"""
    Rational approximation
"""

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy.ndimage as smg
import scipy.signal as sgn
import scipy.optimize as opt

SAV_GOL = True

def array_from_file( filename ):
    my_list = []
    with open(filename, 'r') as f:
        for line in f:
            my_list.append(float(line.split()[0]))
    return np.array(my_list)

def cos( theta ) :
    return np.cos(np.deg2rad(theta))
cos_vec = np.vectorize(cos)

def sin( theta ) :
    return np.sin(np.deg2rad(theta))
sin_vec = np.vectorize(sin)

# Plotting params
plot_sampling = 20
plot_tcksize = 25

# Reference units
mu = 0.877                  # mPa*s
gamma = 57.8                # mPa*m
U_ref = (gamma/mu)*1e-3     # nm/ps
theta_0 = 29.8              # deg

# Obtaining the signal from saved .txt files
folder_name = 'SpreadingData/A07R05Q4'
time = array_from_file(folder_name+'/time.txt')
foot_l = array_from_file(folder_name+'/foot_l.txt')
foot_r = array_from_file(folder_name+'/foot_r.txt')
angle_l = array_from_file(folder_name+'/angle_l.txt')
angle_r = array_from_file(folder_name+'/angle_r.txt')
sub_angle_l = array_from_file(folder_name+'/sub_angle_l.txt')
sub_angle_r = array_from_file(folder_name+'/sub_angle_r.txt')
radius = array_from_file(folder_name+'/radius_fit.txt')
angle_circle = array_from_file(folder_name+'/angle_fit.txt')

# Cutoff inertial phase
dt = time[1]-time[0]
T_ini = 20.0                  # [ps]
T_fin = 20560
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
sub_angle_l = sub_angle_l[N_ini:N_fin]
sub_angle_r = sub_angle_r[N_ini:N_fin]
radius = radius[N_ini:N_fin]
angle_circle = angle_circle[N_ini:N_fin]
init_center = 0.5*(foot_l[0]+foot_r[0])

# Savitzky-Golay filter
sav_gol_win = int(1510/dt)
sav_gol_win = sav_gol_win + (1-sav_gol_win%2)
sav_gol_deg = 3
foot_l_sg = sgn.savgol_filter(foot_l, sav_gol_win, sav_gol_deg)
foot_r_sg = sgn.savgol_filter(foot_r, sav_gol_win, sav_gol_deg)

velocity_l = np.zeros(len(foot_l))
velocity_r = np.zeros(len(foot_r))

if SAV_GOL:
    # Velocity from Savitzky-Golay filtered data (less noisy)
    velocity_l[1:-1] = -0.5 * np.subtract(foot_l_sg[2:],foot_l_sg[0:-2]) / dt
    velocity_r[1:-1] = 0.5 * np.subtract(foot_r_sg[2:],foot_r_sg[0:-2]) / dt
    velocity_l[0] = -( foot_l_sg[1]-foot_l_sg[0] ) / dt
    velocity_r[0] = ( foot_r_sg[1]-foot_r_sg[0] ) / dt
    velocity_l[-1] = -( foot_l_sg[-1]-foot_l_sg[-2] ) / dt
    velocity_r[-1] = ( foot_r_sg[-1]-foot_r_sg[-2] ) / dt
else :
    # Velocity from raw data (very noisy)
    velocity_l[1:-1] = -0.5 * np.subtract(foot_l[2:],foot_l[0:-2]) / dt
    velocity_r[1:-1] = 0.5 * np.subtract(foot_r[2:],foot_r[0:-2]) / dt
    velocity_l[0] = -( foot_l[1]-foot_l[0] ) / dt
    velocity_r[0] = ( foot_r[1]-foot_r[0] ) / dt
    velocity_l[-1] = -( foot_l[-1]-foot_l[-2] ) / dt
    velocity_r[-1] = ( foot_r[-1]-foot_r[-2] ) / dt

time_ns = time*1e-3

fig1, (ax1, ax2) = plt.subplots(1, 2)

ax1.set_title('Spreading branches', fontsize=25.0)
ax1.plot(time_ns, init_center-foot_l, 'b-', linewidth=2.0, label='left (raw)')
ax1.plot(time_ns, foot_r-init_center, 'r-', linewidth=2.0, label='right (raw)')
ax1.plot(time_ns, init_center-foot_l_sg, 'b--', linewidth=3.0, label='left (Sav-Gol)')
ax1.plot(time_ns, foot_r_sg-init_center, 'r--', linewidth=3.0, label='right (Sav-Gol)')
ax1.set_xlabel('t [ns]', fontsize=30.0)
ax1.set_ylabel('x [nm]', fontsize=30.0)
ax1.set_ylim([5.0, 70.0])
ax1.set_xlim([time_ns[0], time_ns[-1]])
ax1.legend(fontsize=20.0)
ax1.tick_params(axis='x', labelsize=plot_tcksize)
ax1.tick_params(axis='y', labelsize=plot_tcksize)

# Spreading curves
ax2.loglog(time_ns, init_center-foot_l, 'b-', linewidth=3.0, label='position')
ax2.loglog(time_ns, foot_r-init_center, 'r-', linewidth=3.0, label='position')
ax2.loglog(time_ns[200:], 50*time_ns[200:]**(1/10), 'm--', linewidth=3.0, label=r'Tanner$\sim t^{1/10}$')
ax2.loglog(time_ns[0:75], 37*time_ns[0:75]**(1/2), 'g-.', linewidth=3.0, label=r'inertial$\sim t^{1/2}$')
ax2.set_xlabel('t [ns]', fontsize=30.0)
ax2.set_ylabel('x [nm]', fontsize=30.0)
ax2.legend(fontsize=20.0)

plt.show()

# Rolling average
delta_t_avg = 500           # [ps]
N_avg = int(delta_t_avg/dt)
contact_angle_l = np.convolve(angle_l, np.ones(N_avg)/N_avg, mode='same')
contact_angle_r = np.convolve(angle_r, np.ones(N_avg)/N_avg, mode='same')

if SAV_GOL :
    # Already filtered
    velocity_l_filter = velocity_l
    velocity_r_filter = velocity_r
else :
    # Rolling average
    velocity_l_filter = np.convolve(velocity_l, np.ones(N_avg)/N_avg, mode='same')
    velocity_r_filter = np.convolve(velocity_r, np.ones(N_avg)/N_avg, mode='same')

fig3, (ax5, ax6) = plt.subplots(1, 2)

# Plot contact angles and wetting speed

ax5.set_title('Contact line speed', fontsize=25.0)
ax5.plot(time_ns[0::plot_sampling], 1e3*velocity_l[0::plot_sampling], 'bs', markersize=8,  markerfacecolor="None", label='left (raw)')
ax5.plot(time_ns[0::plot_sampling], 1e3*velocity_r[0::plot_sampling], 'rd', markersize=10, markerfacecolor="None", label='right (raw)')
ax5.plot(time_ns[N_avg:-N_avg], 1e3*velocity_l_filter[N_avg:-N_avg], 'b-', linewidth=3.0, label='left (filter)')
ax5.plot(time_ns[N_avg:-N_avg], 1e3*velocity_r_filter[N_avg:-N_avg], 'r-', linewidth=3.0, label='right (filter)')
ax5.set_xlabel('t [ns]', fontsize=30.0)
ax5.set_ylabel('dx/dt [nm/ns]', fontsize=30.0)
ax5.set_xlim([time_ns[0], time_ns[-1]])
ax5.legend(fontsize=20.0)
ax5.tick_params(axis='x', labelsize=plot_tcksize)
ax5.tick_params(axis='y', labelsize=plot_tcksize)

ax6.set_title('Dynamic contact angle', fontsize=25.0)
ax6.plot(time_ns[0::plot_sampling], angle_l[0::plot_sampling], 'bs', markersize=8, markeredgewidth=1.0, markerfacecolor="None", label='left (raw)')
ax6.plot(time_ns[0::plot_sampling], angle_r[0::plot_sampling], 'rd', markersize=10, markeredgewidth=1.0, markerfacecolor="None", label='right (raw)')
ax6.plot(time_ns, angle_circle, 'g-', linewidth=3.5, label='circle')
ax6.plot(time_ns[N_avg:-N_avg], contact_angle_l[N_avg:-N_avg], 'b-', linewidth=3.0, label='left (filter)')
ax6.plot(time_ns[N_avg:-N_avg], contact_angle_r[N_avg:-N_avg], 'r-', linewidth=3.0, label='right (filter)')
ax6.set_xlabel('t [ps]', fontsize=30.0)
ax6.set_ylabel(r'$\theta$ [deg]', fontsize=30.0)
ax6.set_xlim([time_ns[0], time_ns[-1]])
ax6.legend(fontsize=20.0)
ax6.tick_params(axis='x', labelsize=plot_tcksize)
ax6.tick_params(axis='y', labelsize=plot_tcksize)

plt.show()

# Assessing Cox law
# diff3 = np.deg2rad(angle_circle[N_avg:-N_avg])**3 - np.deg2rad(contact_angle[N_avg:-N_avg])**3
diff3 = np.concatenate( (np.deg2rad(angle_circle[N_avg:-N_avg]+sub_angle_l[N_avg:-N_avg])**3 - np.deg2rad(angle_l[N_avg:-N_avg]+sub_angle_l[N_avg:-N_avg])**3,np.deg2rad(angle_circle[N_avg:-N_avg]-sub_angle_r[N_avg:-N_avg])**3 - np.deg2rad(angle_r[N_avg:-N_avg]-sub_angle_r[N_avg:-N_avg])**3), axis=None )
# vec_cox = velocity_fit_red
vec_cox = np.concatenate( (velocity_l_filter[N_avg:-N_avg],velocity_r_filter[N_avg:-N_avg]), axis=None)/U_ref
cox = lambda t, p : p*t
p_cox, _ = opt.curve_fit(cox, vec_cox, diff3)
length_ratio = np.exp(p_cox[0]/9.0)
print("L/L_m (Cox) = "+str(length_ratio))

plt.title('Viscous bending effect', fontsize=30.0)
plt.semilogx(vec_cox[0::plot_sampling], diff3[0::plot_sampling], 'k.', markerfacecolor="None", markersize=22.5, markeredgewidth=2.0, label='MD')
Ca = np.linspace(0.004, max(vec_cox+0.05), 250)
plt.semilogx(Ca, p_cox[0]*Ca, 'm--', linewidth=3.0, label=r'fit: $(\theta_c-\alpha)^3-\theta_m^3\sim p_0\cdot U$')
plt.legend(fontsize=20.0)
plt.ylabel(r'$(\theta_c-\alpha)^3-\theta_m^3$ [-1]', fontsize=30.0)
plt.xlabel(r'$U/U_{ref}$ [-1]', fontsize=30.0)
plt.yticks(fontsize=plot_tcksize)
plt.xticks(fontsize=plot_tcksize)
plt.xlim([Ca[0], Ca[-1]])
plt.show()

plt.title('Microscopic contact angle', fontsize=30.0)
plt.plot(time_ns, angle_l+sub_angle_l, 'b-', linewidth=1.5, label='left')
plt.plot(time_ns, angle_r-sub_angle_r, 'r-', linewidth=1.5, label='right')
plt.plot(time_ns, 37.8*np.ones(time_ns.shape), 'k--', linewidth=3.0, label='equilibrium (flat)')
plt.legend(fontsize=20.0)
plt.yticks(fontsize=plot_tcksize)
plt.xticks(fontsize=plot_tcksize)
plt.xlim([time_ns[0], time_ns[-1]])
plt.show()

# Rescale speed
velocity_l_red = velocity_l_filter[N_avg:-N_avg]/U_ref
velocity_r_red = velocity_r_filter[N_avg:-N_avg]/U_ref
velocity_fit_red = np.concatenate((velocity_l_red, velocity_r_red), axis=None)

# Transform into cos and plot againts velocity
# cos_ca_l = cos(theta_0)-cos_vec(contact_angle_l[N_avg:-N_avg])
# cos_ca_r = cos(theta_0)-cos_vec(contact_angle_r[N_avg:-N_avg])
cos_ca_l = cos(theta_0)-cos_vec(angle_circle[N_avg:-N_avg])
cos_ca_r = cos(theta_0)-cos_vec(angle_circle[N_avg:-N_avg])
cos_ca = np.concatenate((cos_ca_l, cos_ca_r), axis=None)
cos_range = np.linspace(0, max(cos_ca), 25)

t_disc = 0;
N_disc = np.argmin(np.abs(time-t_disc))

mkt_exp = lambda x, p0, p1 : p0*np.exp(-p1*x)
popt, _ = opt.curve_fit(mkt_exp, cos_ca[N_disc:], np.abs(velocity_fit_red[N_disc:]))

xi_range = np.linspace(-2.0, 2.0, 500)
t_range  = np.linspace(0.0, 180.0, 500)

fig2, (ax3) = plt.subplots(1, 1)

ax3.set_title('MKT expression fit', fontsize=30.0)
ax3.plot(cos_ca[N_disc::int(0.75*plot_sampling)], np.abs(velocity_fit_red[N_disc::int(0.75*plot_sampling)]), 'k.', markerfacecolor="None", markersize=22.5, markeredgewidth=2.0, label='MD')
# ax3.plot(cos_range, mkt_exp(cos_range, *popt), 'm-.', linewidth=3.0, label=r'fit: $U\sim p_0 \cdot \exp(p_1 x)$')
ax3.set_xlabel(r'$\cos\theta_0-\cos\theta$ [-1]', fontsize=30.0)
ax3.set_ylabel(r'$|U/U_{ref}|$ [-1]', fontsize=30.0)
ax3.legend(fontsize=20.0)
ax3.tick_params(axis='x', labelsize=plot_tcksize)
ax3.tick_params(axis='y', labelsize=plot_tcksize)

plt.show()
