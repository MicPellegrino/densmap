import numpy as np
import matplotlib.pyplot as plt

def array_from_file( filename ):
    my_list = []
    with open(filename, 'r') as f:
        for line in f:
            my_list.append(float(line.split()[0]))
    return np.array(my_list)

f = 'ShearDynamic/Q2_Ca020'
f2 = 'ShearDynamic/Q2_Ca0275'

t_win = 20000   # ps
dt = 12.5       # ps
N0 = int(t_win/dt)

# U = 0.008238312428734322
U = 0.008238312428734322*20/25

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

t_win2 = 14000  # ps
N02 = int(t_win2/dt)

# Ca = 0.20
contact_lines2 = dict()
tl2 = array_from_file(f2+'/position_upper.txt') - 0.5*array_from_file(f2+'/radius_upper.txt')
tr2 = array_from_file(f2+'/position_upper.txt') + 0.5*array_from_file(f2+'/radius_upper.txt')
bl2 = array_from_file(f2+'/position_lower.txt') - 0.5*array_from_file(f2+'/radius_lower.txt')
br2 = array_from_file(f2+'/position_lower.txt') + 0.5*array_from_file(f2+'/radius_lower.txt')
contact_lines2['tl'] = tl2[-N02:]
contact_lines2['tr'] = tr2[-N02:]
contact_lines2['bl'] = bl2[-N02:]
contact_lines2['br'] = br2[-N02:]

disp_f2 = 0.5*(contact_lines2['tl']-contact_lines2['bl']+contact_lines2['tr']-contact_lines2['br'])

time = np.linspace(0.0, t_win, N0)
time2 = np.linspace(0.0, t_win2, N02)

delta_t = 510

l_th = 0.268
t_th = 1.22

plt.plot((1e-3)*time, 0.5*(displacements_rec['r']+displacements_rec['l']), 'k-', label='Ca=0.25', linewidth=3.0)
plt.plot((1e-3)*(time2+(time[-1]-time2[-1])), disp_f2, 'b-', label='Ca=0.20', linewidth=3.0)
plt.plot([20, 20+t_th], [13, 13], 'g-', linewidth= 4.5)
plt.plot([20, 20], [13, 13+l_th], 'g-', linewidth= 4.5)
plt.plot([12.5, 12.5+t_th], [21, 21], 'g-', linewidth= 4.5)
plt.plot([12.5, 12.5], [21, 21+l_th], 'g-', linewidth= 4.5)
plt.ylabel(r'$\Delta x$ [nm]', fontsize=40.0)
plt.xlabel(r'$t$ [ns]', fontsize=40.0)
plt.yticks(fontsize=30.0)
plt.xticks(fontsize=30.0)
plt.legend(fontsize=40.0)
plt.xlim([0.0, (1e-3)*t_win])
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

fig, ax1 = plt.subplots()
# These are in unitless percentages of the figure size. (0,0 is bottom left)
left, bottom, width, height = [0.50, 0.15, 0.45, 0.45]
# ax2 = fig.add_axes([left, bottom, width, height])
###
# ax1.plot(range(10), color='red')
ax1.plot(1e-3*np.array([2000, 2000+2.5*delta_t]), [1, 1+U*2.5*delta_t], 'g', \
         linestyle = (0,(1,0.5)), linewidth=5, label='wall')
# ax1.plot(1e-3*np.array([16600, 16600+delta_t]), [21.6, 21.6-U*delta_t], 'g', \
#         linestyle = (0,(1,0.5)), linewidth=5)
ax1.plot(1e-3*(t_init+time), displacements_rec['l'], 'b-', label='left', linewidth=2)
ax1.plot(1e-3*(t_init+time), displacements_rec['r'], 'r-', label='right', linewidth=2)
ax1.plot(1e-3*(t_init+time), 0.5*(displacements_rec['r']+displacements_rec['l']), 'k-', label='mean', linewidth=3.0)
ax1.text(3.0, 2.0, r'$x=U_wt$', color='green', fontsize=40.0, fontweight='bold')
# ax1.text(9.5, 16.0, r'$x=-U_wt$', color='green', fontsize=40.0, fontweight='bold')
ax1.set_ylabel(r'$\Delta x$ [nm]', fontsize=30.0)
ax1.set_xlabel(r'$t$ [ns]', fontsize=30.0)
ax1.tick_params(axis='both', labelsize=30)
ax1.legend(fontsize=30.0, loc="lower center")
ax1.set_xlim([0.0, max(1e-3*(t_init+time))])
ax1.set_ylim([0.0, 28.5])
###
"""
# ax2.plot(range(6)[::-1], color='green')
ax2.plot(1e-3*np.array([16600, 16600+delta_t]), [21.6, 21.6+U*delta_t], 'g', \
        linestyle = (0,(1,0.5)), linewidth=5, label='wall')
ax2.plot(1e-3*np.array([16600, 16600+delta_t]), [21.6, 21.6-U*delta_t], 'g', \
        linestyle = (0,(1,0.5)), linewidth=5)
ax2.plot(1e-3*(t_init+time), displacements_rec['l'], 'b-', label='left', linewidth=2)
ax2.plot(1e-3*(t_init+time), displacements_rec['r'], 'r-', label='right', linewidth=2)
ax2.plot(1e-3*(t_init+time), 0.5*(displacements_rec['r']+displacements_rec['l']), 'k-', label='mean', linewidth=3.0)
ax2.text(17.5, 25.25, r'$x=+U_wt$', color='green', fontsize=40.0, fontweight='bold')
ax2.text(17.5, 17.75, r'$x=-U_wt$', color='green', fontsize=40.0, fontweight='bold')
# ax2.set_ylabel(r'$\Delta x$ [nm]', fontsize=40.0)
# ax2.set_xlabel(r'$t$ [ns]', fontsize=40.0)
ax2.tick_params(axis='both', labelsize=20)
# ax2.legend(fontsize=40.0, loc="lower right")
ax2.axis('scaled')
ax2.set_xlim([14, 22.5])
ax2.set_ylim([17.5, 26])
"""
###
plt.show()

plt.plot(1e-3*(t_init+time), cosine_diff['l'], 'b-', label='left', linewidth=1.5)
plt.plot(1e-3*(t_init+time), cosine_diff['r'], 'r-', label='right', linewidth=1.5)
plt.title('Receding contact angle fluctuation', fontsize=25.0)
plt.ylabel(r'$\cos(\theta_0)-\cos(\theta)$ [1]', fontsize=20.0)
plt.xlabel(r'$t$ [ps]', fontsize=20.0)
plt.yticks(fontsize=15.0)
plt.xticks(fontsize=15.0)
plt.legend(fontsize=20.0)
plt.ylim([-1.0, 1.0])
plt.xlim([0.0, max(1e-3*(t_init+time))])
plt.show()

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
