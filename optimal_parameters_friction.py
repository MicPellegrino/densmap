import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy.ndimage as smg
import scipy.signal as sgn
import scipy.optimize as opt

from matplotlib import cm

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

# Reference units
mu = 0.877                  # [mPa*s]
gamma = 57.8                # [mPa*m]
U_ref = (gamma/mu)*1e-3     # [nm/ps]

# Time window
T_ini = 100.0               # [ps]
T_fin = 24310.0             # [ps]

theta_0 = 72.0196332498935

folder_name = 'SpreadingData/FlatQ3ADV'
time = array_from_file(folder_name+'/time.txt')
foot_l = array_from_file(folder_name+'/foot_l.txt')
foot_r = array_from_file(folder_name+'/foot_r.txt')
angle_l = array_from_file(folder_name+'/angle_l.txt')
angle_r = array_from_file(folder_name+'/angle_r.txt')
radius = array_from_file(folder_name+'/radius_fit.txt')
angle_circle = array_from_file(folder_name+'/angle_fit.txt')
dt = time[1]-time[0]
time_window = T_fin-T_ini
N_ini = int( T_ini / dt )
N_fin = int( T_fin / dt )
time = time[N_ini:N_fin]
foot_l = foot_l[N_ini:N_fin]
foot_r = foot_r[N_ini:N_fin]
angle_l = angle_l[N_ini:N_fin]
angle_r = angle_r[N_ini:N_fin]
radius = radius[N_ini:N_fin]
angle_circle = angle_circle[N_ini:N_fin]
# Averaging between laft and right c.l.
contact_line_pos = 0.5*(foot_r-foot_l)
# contact_line_pos = 0.5*radius
contact_angle = 0.5*(angle_r+angle_l)
# Rational polynomial fit
def rational(x, p, q):
    return np.polyval(p, x) / np.polyval(q + [1.0], x)
def rational_3_3(x, p0, p1, p2, q1, q2):
    return rational(x, [p0, p1, p2], [q1, q2])
def rational_4_2(x, p0, p1, p2, p4, q1):
    return rational(x, [p0, p1, p2, p4], [q1,])
# popt, pcov = opt.curve_fit(rational_3_3, time, contact_line_pos)
popt, pcov = opt.curve_fit(rational_4_2, time, contact_line_pos)
# Velocity from raw data (very noisy)
velocity_l = np.zeros(len(foot_l))
velocity_r = np.zeros(len(foot_r))
velocity_l[1:-1] = -0.5 * np.subtract(foot_l[2:],foot_l[0:-2]) / dt
velocity_r[1:-1] = 0.5 * np.subtract(foot_r[2:],foot_r[0:-2]) / dt
velocity_l[0] = -( foot_l[1]-foot_l[0] ) / dt
velocity_r[0] = ( foot_r[1]-foot_r[0] ) / dt
velocity_l[-1] = -( foot_l[-1]-foot_l[-2] ) / dt
velocity_r[-1] = ( foot_r[-1]-foot_r[-2] ) / dt
p_0 = np.array(popt[0:4])
p_1 = np.polyder(p_0, m=1)
q_0 = np.concatenate((popt[4:5],[1.0]))
q_1 = np.polyder(q_0, m=1)
def velocity_fit(t) :
    num = ( np.polyval(p_1,t)*np.polyval(q_0,t) - np.polyval(p_0,t)*np.polyval(q_1,t) )
    den = ( np.polyval(q_0,t) )**2
    return num/den
velocity_fit = velocity_fit(time)

# Minimum CL advancement velocity threshold (rough estimate, a bit arbitrary)
# 1.875e-3
vmin_values = np.linspace(1e-5, 1e-4, 10)    # [nm/ps]
# Rolling average
delta_t_avg_values = np.linspace(25, 1000, 50)   # [ps]

err_mkt = np.zeros((10,50))
err_th = np.zeros((10,50))

mkt_a1 = np.zeros((10,50))
mkt_a3 = np.zeros((10,50))
themo_b0 = np.zeros((10,50))
themo_b1 = np.zeros((10,50))

i = 0

mu_st_max = -1
mu_st_min = 100
mu_th_max = -1
mu_th_min = 100

mkt_a3_max = 0
themo_b1_max = 0
mkt_a3_min = 0
themo_b1_min = 0

err_min = 1000

for vmin in vmin_values :

    j = 0

    for delta_t_avg in delta_t_avg_values :
        
        print("(i,j) = ("+str(i)+","+str(j)+")")

        print("vmin = "+str(1e3*vmin)+" [nm/ns]")
        print("delta_t_avg = "+str(delta_t_avg)+" [ps]")
        
        N_avg = int(delta_t_avg/dt)
        contact_angle_smooth = np.convolve(contact_angle, np.ones(N_avg)/N_avg, mode='same')
        idx_steady = np.argmin(np.abs(velocity_fit-vmin))
        t_steady = dt*idx_steady
        velocity_fit_red = velocity_fit[N_avg:idx_steady]/U_ref
        cos_ca = cos(theta_0)-cos_vec(contact_angle_smooth[N_avg:idx_steady])
        mkt_3 = lambda x, a1, a3 : a1*x + a3*(x**3)
        therm_fun = lambda t, b0, b1 : b0 * np.exp(-b1*(0.5*sin(t)+cos(t))**2) * (cos(theta_0)-cos(t))
        popt, _ = opt.curve_fit(mkt_3, cos_ca, velocity_fit_red)
        popt_therm, _ = opt.curve_fit(therm_fun, contact_angle_smooth[N_avg:idx_steady], velocity_fit_red)
        
        mu_st = 1.0/popt[0]
        mu_th = 1.0/popt_therm[0]
        if mu_st > mu_st_max :
            mu_st_max = mu_st
            mkt_a3_max = popt[1]
        if mu_st < mu_st_min :
            mu_st_min = mu_st
            mkt_a3_min = popt[1]
        if mu_th > mu_th_max :
            mu_th_max = mu_th
            mkt_b1_max = popt_therm[1]
        if mu_th < mu_th_min :
            mu_th_min = mu_th
            mkt_b1_min = popt_therm[1]
        
        beta = popt[1] * mu_st
        mu_st_fun = lambda xi : mu_st / (1.0 + beta*xi**2 )
        mu_th_fun = lambda t : mu_th * np.exp(popt_therm[1]*(0.5*sin(t)+cos(t))**2)

        err_mkt[i,j] = np.sqrt( np.sum((mu_st_fun(cos_ca)-velocity_fit_red)*(mu_st_fun(cos_ca)-velocity_fit_red)) \
            / len(velocity_fit_red) )
        err_th[i,j] = np.sqrt( np.sum((mu_th_fun(contact_angle_smooth[N_avg:idx_steady])-velocity_fit_red) \
            * (mu_th_fun(contact_angle_smooth[N_avg:idx_steady])-velocity_fit_red)) / len(velocity_fit_red) )

        if err_mkt[i,j]+err_th[i,j] < err_min :
            err_min = err_mkt[i,j]+err_th[i,j]
            i_min = i
            j_min = j

        mkt_a1[i,j]   = popt[0]
        mkt_a3[i,j]   = popt[1]
        themo_b0[i,j] = popt_therm[0]
        themo_b1[i,j] = popt_therm[1]

        print("err_mkt = "+str(err_mkt[i,j]))
        print("err_th = "+str(err_th[i,j]))

        j += 1

    i += 1

print("mu_st = ["+str(mu_st_min)+','+str(mu_st_max)+']')
print("mu_th = ["+str(mu_th_min)+','+str(mu_th_max)+']')
print("vmin =        "+str(1e3*vmin_values[i_min])+" [nm/ns]")
print("delta_t_avg = "+str(delta_t_avg_values[j_min])+" [ps]")

mu_st_fun_min = lambda xi : mu_st_min / (1.0 + (mkt_a3_min*mu_st_min)*xi**2 )
mu_st_fun_max = lambda xi : mu_st_max / (1.0 + (mkt_a3_max*mu_st_max)*xi**2 )
mu_th_fun_min = lambda t : mu_th_min * np.exp(mkt_b1_max*(0.5*sin(t)+cos(t))**2)
mu_th_fun_max = lambda t : mu_th_max * np.exp(mkt_b1_min*(0.5*sin(t)+cos(t))**2)

xi_range = np.linspace(-2.0, 2.0, 500)
t_range = np.linspace(0.0, 180.0, 500)
fig, (ax11, ax22) = plt.subplots(1, 2)
ax11.set_title('Nonlinear MKT', fontsize=30.0)
ax11.fill_between(xi_range, mu_st_fun_min(xi_range), mu_st_fun_max(xi_range) )
ax22.set_title('Johansson & Hess 2018', fontsize=30.0)
ax22.fill_between(t_range, mu_th_fun_min(t_range), mu_th_fun_max(t_range) )
plt.show()

X, Y = np.meshgrid(vmin_values, delta_t_avg_values, sparse=False, indexing='ij')

# fig1, (ax1, ax2) = plt.subplots(1, 2)
# ax1.pcolormesh(X, Y, err_mkt, cmap=cm.bwr, vmin=0, vmax=1)
# ax2.pcolormesh(X, Y, err_th, cmap=cm.bwr, vmin=0, vmax=1)
plt.pcolormesh(X, Y, err_th+err_mkt, cmap=cm.plasma)
plt.xlabel('v_min')
plt.ylabel('delta_t_avg')
plt.colorbar()
plt.show()

fig1, ( (ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
pcm = ax1.pcolormesh(X, Y, mkt_a1, cmap=cm.plasma)
ax1.set_xlabel('v_min')
ax1.set_ylabel('delta_t_avg')
fig1.colorbar(pcm, ax=ax1)
ax1.set_title('mkt_a1')
pcm = ax2.pcolormesh(X, Y, mkt_a3, cmap=cm.plasma)
ax2.set_xlabel('v_min')
ax2.set_ylabel('delta_t_avg')
fig1.colorbar(pcm, ax=ax2)
ax2.set_title('mkt_a3')
pcm = ax3.pcolormesh(X, Y, themo_b0, cmap=cm.plasma)
ax3.set_xlabel('v_min')
ax3.set_ylabel('delta_t_avg')
fig1.colorbar(pcm, ax=ax3)
ax3.set_title('thermo_b0')
pcm = ax4.pcolormesh(X, Y, themo_b1, cmap=cm.plasma)
ax4.set_xlabel('v_min')
ax4.set_ylabel('delta_t_avg')
fig1.colorbar(pcm, ax=ax4)
ax4.set_title('thermo_b1')
plt.show()
