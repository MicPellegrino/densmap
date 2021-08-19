import numpy as np
import numpy.random as rng

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


print("##############################################################################")
print("# First test: rational polynomial fit for spreading curves over flat surface #")
print("##############################################################################")
folder_name = 'SpreadingData/FlatQ3ADV'
time = array_from_file(folder_name+'/time.txt')
foot_l = array_from_file(folder_name+'/foot_l.txt')
foot_r = array_from_file(folder_name+'/foot_r.txt')
dt = time[1]-time[0]
T_ini = 50.0                  # [ps]
T_fin = 19940.0
time_window = T_fin-T_ini
N_ini = int( T_ini / dt )
N_fin = int( T_fin / dt )
time = time[N_ini:N_fin]
foot_l = foot_l[N_ini:N_fin]
foot_r = foot_r[N_ini:N_fin]
contact_line_pos = 0.5*(foot_r-foot_l)

# Obtaining test and validation dataset
N_tot = len(time)
idx = rng.permutation(N_tot)
time_per = time[idx]
cl_per = contact_line_pos[idx]
N_train = int(0.75*N_tot)
time_train = time_per[0:N_train]
time_test = time_per[N_train:]
cl_train = cl_per[0:N_train]
cl_test = cl_per[N_train:]

def rational(x, n, m, p):
    return np.polyval(p[0:n], x) / np.polyval(np.concatenate((p[n:n+m-1],[1.0]), axis=None), x)

def rational_n_m(x, n, m, *p):
    pr = []
    for t in enumerate(p) :
        pr.append(t[1])
    pr = np.array(pr)
    return rational(x, n, m, pr)

max_deg_n = 6
max_deg_m = 6

test_error_matrix = np.zeros((max_deg_n,max_deg_m))
total_error_matrix = np.zeros((max_deg_n,max_deg_m))

err_min = 10000000
min_deg_n = max_deg_n
min_deg_m = max_deg_m

for deg_n in range(1,max_deg_n+1) :
    for deg_m in range(1,max_deg_m+1) :
        print("Testing n="+str(deg_n)+",m="+str(deg_m))
        rational_fun = lambda x, *p : rational_n_m(x, deg_n, deg_m, *p)
        # Training set to train the model
        popt, pcov = opt.curve_fit(rational_fun, time_train, cl_train, p0=np.ones(5))
        # Test set to compute the error
        cl_eval = rational_fun(time_test, *popt)
        err_test = np.sqrt(np.sum( (cl_test-cl_eval)**2 ))/len(cl_eval)
        # cl_eval = rational_fun(time, *popt)
        # err_tot = np.sqrt(np.sum( (contact_line_pos-cl_eval)**2 ))/len(cl_eval)
        test_error_matrix[deg_n-1, deg_m-1] = err_test
        # total_error_matrix[deg_n-1, deg_m-1] = err_tot
        if err_test < err_min:
            err_min = err_test
            min_deg_n = deg_n
            min_deg_m = deg_m

rational_fun = lambda x, *p : rational_n_m(x, min_deg_n, min_deg_m, *p)
popt, pcov = opt.curve_fit(rational_fun, time, contact_line_pos, p0=np.ones(5))

plt.title("n="+str(min_deg_n)+", m="+str(min_deg_m))
plt.plot(time, rational_fun(time, *popt))
plt.plot(time, contact_line_pos)
plt.show()

print("-------------------------------------------")
print("Optimal numerator order   = "+str(min_deg_n))
print("Optimal demoninator order = "+str(min_deg_m))
print("-------------------------------------------")

print("##############################################################################")
print("# Second test: Savitzky–Golay filter for spreading curves over rough surface #")
print("##############################################################################")
folder_name = 'SpreadingData/A07R15Q4'
time = array_from_file(folder_name+'/time.txt')
foot_l = array_from_file(folder_name+'/foot_l.txt')
foot_r = array_from_file(folder_name+'/foot_r.txt')
dt = time[1]-time[0]
T_ini = 20.0                   # [ps]
T_fin = 33760.0
time_window = T_fin-T_ini
N_ini = int( T_ini / dt )
N_fin = int( T_fin / dt )
time = time[N_ini:N_fin]
foot_l = foot_l[N_ini:N_fin]
foot_r = foot_r[N_ini:N_fin]
T_spike_0 = 8500                # [ps]
T_spike_1 = 14500           
N_spike_0 = int( T_spike_0 / dt )
N_spike_1 = int( T_spike_1 / dt )
T_pin_0 = 14500                # [ps]
T_pin_1 = 23000           
N_pin_0 = int( T_pin_0 / dt )
N_pin_1 = int( T_pin_1 / dt )

min_deg_sg = 3
max_deg_sg = 3
max_win_sg = int(5000/dt)
delta_win  = int(100/dt)
vector_win = np.linspace(delta_win, max_win_sg, int(max_win_sg/delta_win))

max_sn_ratio = 0
opt_deg_sg = max_deg_sg
opt_win_sg = max_win_sg
# Norm penalization term
alpha = 0.0

signal_noise_matrix = np.zeros((max_deg_sg-min_deg_sg+1,len(vector_win)))
velocity_l = np.zeros(len(foot_l))
velocity_r = np.zeros(len(foot_r))

for deg_sg in range(min_deg_sg,max_deg_sg+1) :
    for vw in range(len(vector_win)) :
        sav_gol_win = int(vector_win[vw])
        sav_gol_win = sav_gol_win + (1-sav_gol_win%2)
        print("Testing n="+str(deg_sg)+",win="+str(sav_gol_win))
        foot_l_sg = sgn.savgol_filter(foot_l, sav_gol_win, deg_sg, deriv=1)
        foot_r_sg = sgn.savgol_filter(foot_r, sav_gol_win, deg_sg, deriv=1)
        velocity_l[1:-1] = -0.5 * np.subtract(foot_l_sg[2:],foot_l_sg[0:-2]) / dt
        velocity_r[1:-1] = 0.5 * np.subtract(foot_r_sg[2:],foot_r_sg[0:-2]) / dt
        velocity_l[0] = -( foot_l_sg[1]-foot_l_sg[0] ) / dt
        velocity_r[0] = ( foot_r_sg[1]-foot_r_sg[0] ) / dt
        velocity_l[-1] = -( foot_l_sg[-1]-foot_l_sg[-2] ) / dt
        velocity_r[-1] = ( foot_r_sg[-1]-foot_r_sg[-2] ) / dt
        signal = min( max( velocity_l[N_spike_0:N_spike_1] ), max( velocity_r[N_spike_0:N_spike_1] ) )
        noise  = max( np.std(velocity_l[N_pin_0:N_pin_1]), np.std(velocity_r[N_pin_0:N_pin_1]) )
        # 2-norm difference #
        # n2_dist = np.sum((foot_l_sg[N_spike_0:N_spike_1]-foot_l[N_spike_0:N_spike_1])**2)
        # n2_dist += np.sum((foot_r_sg[N_spike_0:N_spike_1]-foot_r[N_spike_0:N_spike_1])**2)
        #####################
        sn_ratio = signal/noise
        if sn_ratio > max_sn_ratio :
            max_sn_ratio = sn_ratio
            opt_deg_sg = deg_sg
            opt_win_sg = sav_gol_win
        signal_noise_matrix[deg_sg-min_deg_sg, vw] = sn_ratio

print(max_sn_ratio)

print("-------------------------------------")
print("Optimal S-G order  = "+str(opt_deg_sg))
print("Optimal S-G window = "+str(opt_win_sg))
print("-------------------------------------")

foot_l_sg = sgn.savgol_filter(foot_l, opt_win_sg, opt_deg_sg)
foot_r_sg = sgn.savgol_filter(foot_r, opt_win_sg, opt_deg_sg)
velocity_l[1:-1] = -0.5 * np.subtract(foot_l_sg[2:],foot_l_sg[0:-2]) / dt
velocity_r[1:-1] = 0.5 * np.subtract(foot_r_sg[2:],foot_r_sg[0:-2]) / dt
velocity_l[0] = -( foot_l_sg[1]-foot_l_sg[0] ) / dt
velocity_r[0] = ( foot_r_sg[1]-foot_r_sg[0] ) / dt
velocity_l[-1] = -( foot_l_sg[-1]-foot_l_sg[-2] ) / dt
velocity_r[-1] = ( foot_r_sg[-1]-foot_r_sg[-2] ) / dt

time_ns = (1e-3)*time
init_center = 0.5*(foot_l[0]+foot_r[0])
plot_sampling = 20
plot_tcksize = 25
delta_t_avg = 500           # [ps]
N_avg = int(delta_t_avg/dt)
velocity_l_filter = velocity_l
velocity_r_filter = velocity_r

fig1, (ax1, ax5) = plt.subplots(1, 2)
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
ax5.set_title('Contact line speed', fontsize=25.0)
ax5.plot(time_ns[N_avg:-N_avg], 1e3*velocity_l_filter[N_avg:-N_avg], 'b-', linewidth=2.5, label='left (filter)')
ax5.plot(time_ns[N_avg:-N_avg], 1e3*velocity_r_filter[N_avg:-N_avg], 'r-', linewidth=2.5, label='right (filter)')
ax5.plot(time_ns[N_avg:-N_avg], np.zeros(time_ns[N_avg:-N_avg].shape), 'k--', linewidth=1.5)
ax5.plot([T_spike_0*1e-3, T_spike_0*1e-3], [0.0, max(1e3*velocity_l_filter[N_avg:-N_avg])], 'k--')
ax5.plot([T_spike_1*1e-3, T_spike_1*1e-3], [0.0, max(1e3*velocity_l_filter[N_avg:-N_avg])], 'k--')
ax5.plot([T_pin_1*1e-3, T_pin_1*1e-3],     [0.0, max(1e3*velocity_l_filter[N_avg:-N_avg])], 'k--')
ax5.set_xlabel('t [ns]', fontsize=30.0)
ax5.set_ylabel('dx/dt [nm/ns]', fontsize=30.0)
ax5.set_xlim([time_ns[0], time_ns[-1]])
ax5.legend(fontsize=20.0)
ax5.tick_params(axis='x', labelsize=plot_tcksize)
ax5.tick_params(axis='y', labelsize=plot_tcksize)
plt.show()

plt.plot(time_ns, init_center-foot_l, 'b-', linewidth=3.0)
plt.plot(time_ns, foot_r-init_center, 'r-', linewidth=3.0)
plt.plot(time_ns, init_center-foot_l_sg, 'b--', linewidth=3.5)
plt.plot(time_ns, foot_r_sg-init_center, 'r--', linewidth=3.5)
plt.ylim([42.0, 50.0])
plt.xlim([9.5, 11.5])
plt.show()

plt.plot(time_ns[N_avg:-N_avg], 1e3*velocity_l_filter[N_avg:-N_avg], 'b-', linewidth=2.5, label='left (filter)')
plt.plot(time_ns[N_avg:-N_avg], 1e3*velocity_r_filter[N_avg:-N_avg], 'r-', linewidth=2.5, label='right (filter)')
plt.plot(time_ns[N_avg:-N_avg], np.zeros(time_ns[N_avg:-N_avg].shape), 'k--', linewidth=1.5)
plt.xlabel('t [ns]', fontsize=30.0)
plt.ylabel('dx/dt [nm/ns]', fontsize=30.0)
plt.xlim([time_ns[N_avg], time_ns[-N_avg]])
plt.legend(fontsize=20.0)
plt.xticks(fontsize=plot_tcksize)
plt.yticks(fontsize=plot_tcksize)
plt.show()

print("############################################################################")
print("# Third test: rational polynomial fit for contact angles over flat surface #")
print("############################################################################")
folder_name = 'SpreadingData/FlatQ3ADV'
time = array_from_file(folder_name+'/time.txt')
angle_l = array_from_file(folder_name+'/angle_l.txt')
angle_r = array_from_file(folder_name+'/angle_r.txt')
dt = time[1]-time[0]
T_ini = 50.0                  # [ps]
T_fin = 19940.0
time_window = T_fin-T_ini
N_ini = int( T_ini / dt )
N_fin = int( T_fin / dt )
time = time[N_ini:N_fin]
angle_l = angle_l[N_ini:N_fin]
angle_r = angle_r[N_ini:N_fin]
angle_mean= 0.5*(angle_l+angle_r)

# Obtaining test and validation dataset
N_tot = len(time)
idx = rng.permutation(N_tot)
time_per = time[idx]
cl_per = angle_mean[idx]
N_train = int(0.75*N_tot)
time_train = time_per[0:N_train]
time_test = time_per[N_train:]
cl_train = cl_per[0:N_train]
cl_test = cl_per[N_train:]

def rational(x, n, m, p):
    return np.polyval(p[0:n], x) / np.polyval(np.concatenate((p[n:n+m-1],[1.0]), axis=None), x)

def rational_n_m(x, n, m, *p):
    pr = []
    for t in enumerate(p) :
        pr.append(t[1])
    pr = np.array(pr)
    return rational(x, n, m, pr)

max_deg_n = 10
max_deg_m = 10

test_error_matrix = np.zeros((max_deg_n,max_deg_m))
total_error_matrix = np.zeros((max_deg_n,max_deg_m))

err_min = 10000000
min_deg_n = max_deg_n
min_deg_m = max_deg_m

for deg_n in range(1,max_deg_n+1) :
    for deg_m in range(1,max_deg_m+1) :
        print("Testing n="+str(deg_n)+",m="+str(deg_m))
        rational_fun = lambda x, *p : rational_n_m(x, deg_n, deg_m, *p)
        # Training set to train the model
        popt, pcov = opt.curve_fit(rational_fun, time_train, cl_train, p0=np.ones(5))
        # Test set to compute the error
        cl_eval = rational_fun(time_test, *popt)
        err_test = np.sqrt(np.sum( (cl_test-cl_eval)**2 ))/len(cl_eval)
        # cl_eval = rational_fun(time, *popt)
        # err_tot = np.sqrt(np.sum( (contact_line_pos-cl_eval)**2 ))/len(cl_eval)
        test_error_matrix[deg_n-1, deg_m-1] = err_test
        # total_error_matrix[deg_n-1, deg_m-1] = err_tot
        if err_test < err_min:
            err_min = err_test
            min_deg_n = deg_n
            min_deg_m = deg_m

rational_fun = lambda x, *p : rational_n_m(x, min_deg_n, min_deg_m, *p)
popt, pcov = opt.curve_fit(rational_fun, time, angle_mean, p0=np.ones(5))

plt.title("n="+str(min_deg_n)+", m="+str(min_deg_m))
plt.plot(time, rational_fun(time, *popt))
plt.plot(time, angle_mean)
plt.show()

print("##########################################################################")
print("# Forth test: Savitzky–Golay filter for contact angle over rough surface #")
print("##########################################################################")
folder_name = 'SpreadingData/A07R15Q4'
time = array_from_file(folder_name+'/time.txt')
angle_l = array_from_file(folder_name+'/angle_l.txt')
angle_r = array_from_file(folder_name+'/angle_r.txt')
dt = time[1]-time[0]
T_ini = 20.0                   # [ps]
T_fin = 33760.0
time_window = T_fin-T_ini
N_ini = int( T_ini / dt )
N_fin = int( T_fin / dt )
time = time[N_ini:N_fin]
angle_l = angle_l[N_ini:N_fin]
angle_r = angle_r[N_ini:N_fin]
T_spike_0 = 8500                # [ps]
T_spike_1 = 14500           
N_spike_0 = int( T_spike_0 / dt )
N_spike_1 = int( T_spike_1 / dt )
T_pin_0 = 14500                # [ps]
T_pin_1 = 23000           
N_pin_0 = int( T_pin_0 / dt )
N_pin_1 = int( T_pin_1 / dt )

min_deg_sg = 1
max_deg_sg = 6
max_win_sg = int(5000/dt)
delta_win  = int(100/dt)
vector_win = np.linspace(delta_win, max_win_sg, int(max_win_sg/delta_win))

max_sn_ratio = 0
opt_deg_sg = max_deg_sg
opt_win_sg = max_win_sg
# Norm penalization term
alpha = 0.0

signal_noise_matrix = np.zeros((max_deg_sg-min_deg_sg+1,len(vector_win)))
velocity_l = np.zeros(len(foot_l))
velocity_r = np.zeros(len(foot_r))

for deg_sg in range(min_deg_sg,max_deg_sg+1) :
    for vw in range(len(vector_win)) :
        sav_gol_win = int(vector_win[vw])
        sav_gol_win = sav_gol_win + (1-sav_gol_win%2)
        print("Testing n="+str(deg_sg)+",win="+str(sav_gol_win))
        angle_l_sg = sgn.savgol_filter(angle_l, sav_gol_win, deg_sg, deriv=0)
        angle_r_sg = sgn.savgol_filter(angle_r, sav_gol_win, deg_sg, deriv=0)
        signal = min( max(angle_l_sg[N_spike_0:N_spike_1])-min(angle_l_sg[N_spike_0:N_spike_1]) , \
                max(angle_r_sg[N_spike_0:N_spike_1])-min(angle_r_sg[N_spike_0:N_spike_1]) )
        noise  = max( np.std(angle_l_sg[N_pin_0:N_pin_1]),      np.std(angle_r_sg[N_pin_0:N_pin_1]) )
        # 2-norm difference #
        # n2_dist = np.sum((foot_l_sg[N_spike_0:N_spike_1]-foot_l[N_spike_0:N_spike_1])**2)
        # n2_dist += np.sum((foot_r_sg[N_spike_0:N_spike_1]-foot_r[N_spike_0:N_spike_1])**2)
        #####################
        sn_ratio = signal/noise
        if sn_ratio > max_sn_ratio :
            max_sn_ratio = sn_ratio
            opt_deg_sg = deg_sg
            opt_win_sg = sav_gol_win
        signal_noise_matrix[deg_sg-min_deg_sg, vw] = sn_ratio

print(max_sn_ratio)

print("-------------------------------------")
print("Optimal S-G order  = "+str(opt_deg_sg))
print("Optimal S-G window = "+str(opt_win_sg))
print("-------------------------------------")

angle_l_sg = sgn.savgol_filter(angle_l, opt_win_sg, opt_deg_sg)
angle_r_sg = sgn.savgol_filter(angle_r, opt_win_sg, opt_deg_sg)


plt.plot(time_ns[N_avg:-N_avg], angle_l[N_avg:-N_avg], 'b.', linewidth=2.5, label='left (raw)')
plt.plot(time_ns[N_avg:-N_avg], angle_r[N_avg:-N_avg], 'r.', linewidth=2.5, label='right (raw)')
plt.plot(time_ns[N_avg:-N_avg], angle_l_sg[N_avg:-N_avg], 'b-', linewidth=2.5, label='left (filter)')
plt.plot(time_ns[N_avg:-N_avg], angle_r_sg[N_avg:-N_avg], 'r-', linewidth=2.5, label='right (filter)')
plt.plot([T_spike_0*1e-3, T_spike_0*1e-3], [0.0, max(angle_l_sg[N_avg:-N_avg])], 'k--')
plt.plot([T_spike_1*1e-3, T_spike_1*1e-3], [0.0, max(angle_l_sg[N_avg:-N_avg])], 'k--')
plt.plot([T_pin_1*1e-3, T_pin_1*1e-3],     [0.0, max(angle_l_sg[N_avg:-N_avg])], 'k--')
plt.xlabel('t [ns]', fontsize=30.0)
plt.ylabel('theta [deg]', fontsize=30.0)
plt.xlim([time_ns[N_avg], time_ns[-N_avg]])
plt.legend(fontsize=20.0)
plt.xticks(fontsize=plot_tcksize)
plt.yticks(fontsize=plot_tcksize)
plt.show()
