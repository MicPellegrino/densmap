import numpy as np
import matplotlib.pyplot as plt

def array_from_file( filename ):
    my_list = []
    with open(filename, 'r') as f:
        for line in f:
            my_list.append(float(line.split()[0]))
    return np.array(my_list)

f = 'ShearDynamic/Q2_Ca025'

t_win = 50000   # ps
dt = 12.5       # ps
N0 = int(t_win/dt)

U = 0.008238312428734322

# Contact lines
contact_lines = dict()
tl = array_from_file(f+'/position_upper.txt') - 0.5*array_from_file(f+'/radius_upper.txt')
tr = array_from_file(f+'/position_upper.txt') + 0.5*array_from_file(f+'/radius_upper.txt')
bl = array_from_file(f+'/position_lower.txt') - 0.5*array_from_file(f+'/radius_lower.txt')
br = array_from_file(f+'/position_lower.txt') + 0.5*array_from_file(f+'/radius_lower.txt')
t_tot = dt*len(tl)
t_init = t_tot-t_win
contact_lines['tl'] = tl[-N0:]
contact_lines['tr'] = tr[-N0:]
contact_lines['bl'] = bl[-N0:]
contact_lines['br'] = br[-N0:]

# Displacement comparison
displacements_rec = dict()
displacements_rec['l'] = contact_lines['tl']-contact_lines['bl']
displacements_rec['r'] = contact_lines['tr']-contact_lines['br']

time = np.linspace(0.0, t_win, N0)

delta_t = 500

plt.plot([16600, 16600+delta_t], [21.6, 21.6+U*delta_t], 'g-', linewidth=5, label='wall')
plt.plot([16600, 16600+delta_t], [21.6, 21.6-U*delta_t], 'g-', linewidth=5)
plt.plot(t_init+time, displacements_rec['l'], 'b-', label='left', linewidth=2.5)
plt.plot(t_init+time, displacements_rec['r'], 'r-', label='right', linewidth=2.5)
plt.plot(t_init+time, 0.5*(displacements_rec['r']+displacements_rec['l']), 'k-', label='mean', linewidth=3.0)
plt.text(13500, 23.25, r'$x=U_wt$', color='green', fontsize=20.0, fontweight='bold')
plt.text(12750, 18.25, r'$x=-U_wt$', color='green', fontsize=20.0, fontweight='bold')
plt.title('Dispacement oscillation', fontsize=30.0)
# plt.ylabel(r'$\delta(\Delta x)-<\Delta x>$ [nm]', fontsize=20.0)
plt.ylabel(r'$\Delta x$ [nm]', fontsize=25.0)
plt.xlabel(r'$t$ [ps]', fontsize=25.0)
plt.yticks(fontsize=20.0)
plt.xticks(fontsize=20.0)
plt.legend(fontsize=25.0)
plt.show()

# Contact angles
contact_angles = dict()
tl = array_from_file(f+'/angle_tl.txt')
tr = array_from_file(f+'/angle_tr.txt')
bl = array_from_file(f+'/angle_bl.txt')
br = array_from_file(f+'/angle_br.txt')
contact_angles['tl'] = tl[-N0:]
contact_angles['tr'] = tr[-N0:]
contact_angles['bl'] = bl[-N0:]
contact_angles['br'] = br[-N0:]

theta_0 = 95.0

cos = lambda t : np.cos(np.deg2rad(t))

# Angle comparison
cosine_diff = dict()
cosine_diff_int = dict()
cosine_diff['l'] = cos(theta_0) - cos(contact_angles['bl'])
cosine_diff['r'] = cos(theta_0) - cos(contact_angles['tr'])
cosine_diff_int['l'] = dt*np.cumsum(cosine_diff['l'])
cosine_diff_int['r'] = dt*np.cumsum(cosine_diff['r'])

"""
plt.plot(time, cosine_diff['l'], label='left', linewidth=2.0)
plt.plot(time, cosine_diff['r'], label='right', linewidth=2.0)
plt.title('Receding contact angle fluctuation', fontsize=25.0)
plt.ylabel(r'$\cos(\theta_0)-\cos(\theta)$ [1]', fontsize=20.0)
plt.xlabel(r'$t$ [ps]', fontsize=20.0)
plt.yticks(fontsize=15.0)
plt.xticks(fontsize=15.0)
plt.legend(fontsize=20.0)
plt.show()
"""

"""
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
"""
