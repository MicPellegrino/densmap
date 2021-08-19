import numpy as np
import matplotlib.pyplot as plt

def array_from_file( filename ):
    my_list = []
    with open(filename, 'r') as f:
        for line in f:
            my_list.append(float(line.split()[0]))
    return np.array(my_list)

folders = [ 'ShearDynamic/Q2_Ca020', 
            'ShearDynamic/Q3_Ca010',
            'ShearDynamic/Q4_Ca002' ]

t_win = 20000   # ps
dt = 12.5       # ps
N0 = int(t_win/dt)

# Contact lines
contact_lines = dict()
for f in folders :
    contact_lines[f] = dict()
    tl = array_from_file(f+'/position_upper.txt') - 0.5*array_from_file(f+'/radius_upper.txt')
    tr = array_from_file(f+'/position_upper.txt') + 0.5*array_from_file(f+'/radius_upper.txt')
    bl = array_from_file(f+'/position_lower.txt') - 0.5*array_from_file(f+'/radius_lower.txt')
    br = array_from_file(f+'/position_lower.txt') + 0.5*array_from_file(f+'/radius_lower.txt')
    contact_lines[f]['tl'] = tl[-N0:]
    contact_lines[f]['tr'] = tr[-N0:]
    contact_lines[f]['bl'] = bl[-N0:]
    contact_lines[f]['br'] = br[-N0:]

# Displacement comparison
displacements = dict()
displacements_rec = dict()
disp_avg = dict()
disp_osc = dict()
for f in folders :
    displacements[f] = 0.5*(contact_lines[f]['tl']-contact_lines[f]['bl'])+0.5*(contact_lines[f]['tr']-contact_lines[f]['br'])
    displacements_rec[f] = dict()
    displacements_rec[f]['l'] = contact_lines[f]['tl']-contact_lines[f]['bl']
    displacements_rec[f]['r'] = contact_lines[f]['tr']-contact_lines[f]['br']
    disp_avg[f] = np.mean(displacements[f]) 
    disp_osc[f] = displacements[f] - disp_avg[f]

time = np.linspace(0.0, t_win, N0)
for f in folders :
    plt.plot(time, disp_osc[f], label=f, linewidth=2.0)
plt.title('Dispacement fluctuation', fontsize=25.0)
plt.ylabel(r'$\delta(\Delta x)-<\Delta x>$ [nm]', fontsize=20.0)
plt.xlabel(r'$t$ [ps]', fontsize=20.0)
plt.yticks(fontsize=15.0)
plt.xticks(fontsize=15.0)
plt.legend(fontsize=20.0)
plt.show()

# Contact angles
contact_angles = dict()
for f in folders :
    contact_angles[f] = dict()
    tl = array_from_file(f+'/angle_tl.txt')
    tr = array_from_file(f+'/angle_tr.txt')
    bl = array_from_file(f+'/angle_bl.txt')
    br = array_from_file(f+'/angle_br.txt')
    contact_angles[f]['tl'] = tl[-N0:]
    contact_angles[f]['tr'] = tr[-N0:]
    contact_angles[f]['bl'] = bl[-N0:]
    contact_angles[f]['br'] = br[-N0:]

contact_angles['ShearDynamic/Q2_Ca025']['t0'] = 95.1
contact_angles['ShearDynamic/Q3_Ca010']['t0'] = 68.8
contact_angles['ShearDynamic/Q4_Ca002']['t0'] = 37.8

cos = lambda t : np.cos(np.deg2rad(t))

# Angle comparison
cosine_diff = dict()
cosine_diff_int = dict()
for f in folders :
    cosine_diff[f] = dict()
    cosine_diff[f]['l'] = cos(contact_angles[f]['t0']) - cos(contact_angles[f]['bl'])
    cosine_diff[f]['r'] = cos(contact_angles[f]['t0']) - cos(contact_angles[f]['tr'])
    cosine_diff_int[f] = dict()
    cosine_diff_int[f]['l'] = dt*np.cumsum(cosine_diff[f]['l'])
    cosine_diff_int[f]['r'] = dt*np.cumsum(cosine_diff[f]['r'])

for f in folders :
    plt.plot(time, cosine_diff[f]['l'], label=f, linewidth=2.0)
plt.title('Receding contact angle fluctuation (left)', fontsize=25.0)
plt.ylabel(r'$\cos(\theta_0)-\cos(\theta)$ [1]', fontsize=20.0)
plt.xlabel(r'$t$ [ps]', fontsize=20.0)
plt.yticks(fontsize=15.0)
plt.xticks(fontsize=15.0)
plt.legend(fontsize=20.0)
plt.show()

# Only for Q4
rd_q4 = displacements_rec[f]['r']
rd_q4 = (rd_q4-np.mean(rd_q4))/np.mean(rd_q4)
cd_q4 = cosine_diff['ShearDynamic/Q4_Ca002']['r']
plt.plot(time, rd_q4, 'b-', label='relative displacement', linewidth=2.0 )
plt.plot(time, cd_q4, 'r.', label='contact angle', linewidth=2.0)
plt.ylabel('[1]', fontsize=20.0)
plt.xlabel(r'$t$ [ps]', fontsize=20.0)
plt.yticks(fontsize=15.0)
plt.xticks(fontsize=15.0)
plt.legend(fontsize=20.0)
plt.show()

print(np.corrcoef(rd_q4,cd_q4))
