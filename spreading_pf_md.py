import numpy as np
import matplotlib.pyplot as plt

pf_output = 'q3_stats_MD.txt'

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

time_pf = []
pos_left_pf = []
pos_right_pf = []
angle_left_pf = []
angle_right_pf = []

N_off = 100

f = open(pf_output, "r")
n = 0
for line in f :
    cols = line.split()
    n += 1
    if n > N_off :
        time_pf.append(float(cols[0]))
        pos_left_pf.append(float(cols[1]))
        pos_right_pf.append(float(cols[3]))
        angle_left_pf.append(float(cols[5]))
        angle_right_pf.append(float(cols[6]))
f.close()

time_pf = np.array(time_pf)
pos_left_pf = np.array(pos_left_pf)
pos_right_pf = np.array(pos_right_pf)
angle_left_pf = np.array(angle_left_pf)
angle_right_pf = np.array(angle_right_pf)

# MD data
folder_name = 'SpreadingData/A07R15Q3'
time = array_from_file(folder_name+'/time.txt')
foot_l = array_from_file(folder_name+'/foot_l.txt')
foot_r = array_from_file(folder_name+'/foot_r.txt')
angle_l = array_from_file(folder_name+'/angle_l.txt')
angle_r = array_from_file(folder_name+'/angle_r.txt')
sub_angle_l = array_from_file(folder_name+'/sub_angle_l.txt')
sub_angle_r = array_from_file(folder_name+'/sub_angle_r.txt')
radius = array_from_file(folder_name+'/radius_fit.txt')
angle_circle = array_from_file(folder_name+'/angle_fit.txt')

angle_l += sub_angle_l
angle_r -= sub_angle_r
center = 0.5*(foot_r[0]+foot_l[0])

N_avg = np.argmin(np.abs(time-28000))
avg_angle_l_md = np.mean(angle_l[N_avg:])
avg_angle_r_md = np.mean(angle_r[N_avg:])
print('theta_avg (left)  = '+str(avg_angle_l_md))
print('theta_avg (right) = '+str(avg_angle_r_md))

plt.title('Contact line position', fontsize=25.0)
plt.plot(1e3*time_pf, pos_right_pf+center, 'r--', linewidth=3.0, label='PF')
plt.plot(1e3*time_pf, pos_left_pf+center, 'r--', linewidth=3.0)
plt.plot(time, foot_r, 'g-', linewidth=2.0, label='MD')
plt.plot(time, foot_l, 'g-', linewidth=2.0)
plt.legend(fontsize=20.0)
plt.xticks(fontsize=17.5)
plt.yticks(fontsize=17.5)
plt.xlabel(r'$t$ [ps]', fontsize=20.0)
plt.ylabel(r'$x_{cl}$ [nm]', fontsize=20.0)
plt.show()

fig3, (ax1, ax2) = plt.subplots(2, 1)

ax1.set_title('Microscopic contact angle (right)', fontsize=25.0)
ax1.plot(time, angle_r, 'g.', label='MD')
ax1.plot([time[N_avg], time[-1]], [avg_angle_r_md, avg_angle_r_md], 'k--', linewidth=4.0)
ax2.set_title('                          (left)' , fontsize=25.0)
ax2.plot(time, angle_l, 'g.')
ax2.plot([time[N_avg], time[-1]], [avg_angle_l_md, avg_angle_l_md], 'k--', linewidth=4.0)
ax1.plot(1e3*time_pf, angle_right_pf, 'r--', linewidth=3.0, label='PF')
plt.setp(ax1.get_xticklabels(), visible=False)
ax2.plot(1e3*time_pf, angle_left_pf, 'r--', linewidth=3.0)
ax1.legend(fontsize=20.0)
ax1.tick_params(axis='y', labelsize=17.5)
ax2.tick_params(axis='y', labelsize=17.5)
ax2.tick_params(axis='x', labelsize=17.5)
ax2.set_xlabel(r'$t$ [ps]', fontsize=20.0)
ax2.set_ylabel(r'$\theta$ [deg]', fontsize=20.0)
ax1.set_ylabel(r'$\theta$ [deg]', fontsize=20.0)
plt.show()
