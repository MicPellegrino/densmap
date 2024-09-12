import numpy as np
import densmap as dm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy.random as rng

def array_from_file( filename ):
    my_list = []
    with open(filename, 'r') as f:
        for line in f:
            my_list.append(float(line.split()[0]))
    return np.array(my_list)

def cross_validation_error(x, y, kmax=10) :
    coeff_list = []
    for k in range(kmax) :
        p = rng.permutation(len(x))
        x_train = x[p[0:int(0.8*len(x))]]
        y_train = y[p[0:int(0.8*len(y))]]
        pcoeff = np.polyfit(x_train, y_train, deg=1)
        coeff_list.append(pcoeff[0])
    coeff_list = np.array(coeff_list)
    return np.max(coeff_list)-np.min(coeff_list)

color_line = dict()
color_line['c003'] = 'm'
color_line['c005'] = 'r'
color_line['c008'] = 'g'
color_line['c010'] = 'c'

avg_window = dict()
avg_window['c003'] = 18
avg_window['c005'] = 40
avg_window['c008'] = 8
avg_window['c010'] = 5

folders = dict()
folders['c003'] = '/home/michele/densmap/ShearDynamic/Q4_Ca003/'
folders['c005'] = '/home/michele/densmap/ShearDynamic/Q4_Ca005/'
folders['c008'] = '/home/michele/densmap/ShearDynamic/Q4_Ca008/'
folders['c010'] = '/home/michele/densmap/ShearDynamic/Q4_Ca010/'

viscosity = 8.77e-4
surf_tens = 5.78e-2
wall_speed = dict()
wall_speed['c003'] = (0.015)*(surf_tens/viscosity)
wall_speed['c005'] = (0.025)*(surf_tens/viscosity)
wall_speed['c008'] = (0.040)*(surf_tens/viscosity)
wall_speed['c010'] = (0.050)*(surf_tens/viscosity)

time = dict()

top_left = dict()
bot_left = dict()
top_right = dict()
bot_right = dict()

wall_pos = dict()
spinup = 2.0    # [nm]
coeff = dict()
err_fit = dict()

for k in folders.keys() :
    
    time[k] = (1e-3)*array_from_file(folders[k]+'time.txt')
    
    top_left[k] = array_from_file(folders[k]+'position_upper.txt')-0.5*array_from_file(folders[k]+'radius_upper.txt')
    bot_left[k] = array_from_file(folders[k]+'position_lower.txt')-0.5*array_from_file(folders[k]+'radius_lower.txt')
    top_right[k] = array_from_file(folders[k]+'position_upper.txt')+0.5*array_from_file(folders[k]+'radius_upper.txt')
    bot_right[k] = array_from_file(folders[k]+'position_lower.txt')+0.5*array_from_file(folders[k]+'radius_lower.txt')
    
    wall_pos[k] = wall_speed[k]*time[k]

    nh = int(0.2*len(time[k]))
    coeff[k] = np.polyfit(time[k][nh:], bot_left[k][0]-bot_left[k][nh:], deg=1)
    err_fit[k] = cross_validation_error(time[k][nh:], bot_left[k][0]-bot_left[k][nh:])

Ca_r = dict()
Ca_r_err = dict()

# f, (ax1, ax2) = plt.subplots(1, 2)
f, (ax1) = plt.subplots()

f_sci = mticker.ScalarFormatter(useOffset=False, useMathText=True)
g_sci = lambda x,pos : "${}$".format(f_sci.set_scientific('%1.2e' % x))
fmt_sci = mticker.FuncFormatter(g_sci)

for k in folders.keys() :

    # Relative capillary number
    Ca_r[k] = (wall_speed[k]-coeff[k][0])*(viscosity/surf_tens)
    Ca_r_err[k] = (np.sqrt(err_fit[k]))*(viscosity/surf_tens)

    # label_wall=r"$U_w$="+"{:1.4f}".format(wall_speed[k])+" nm/ns"
    # label_wall = r"Ca$_w$ = "+"{:1.1e}".format(wall_speed[k]*viscosity/surf_tens)+r", Ca$_{cl}$ = "+"{:1.2e}".format(wall_speed[k]*viscosity/surf_tens-Ca_r[k])
    # label_wall = r"Ca$_w$ = "+"{}".format(fmt_sci(wall_speed[k]*viscosity/surf_tens))+r", Ca$_{cl}$ = "+"{}".format(fmt_sci(wall_speed[k]*viscosity/surf_tens-Ca_r[k]))
    # label_wall = r"Ca$_w$ = "+r"{}".format(fmt_sci(wall_speed[k]*viscosity/surf_tens))
    # label_text = r"Ca$_{cl}$ = "+r"{}".format(fmt_sci(wall_speed[k]*viscosity/surf_tens-Ca_r[k]))
    label_wall = r"Ca$_w$ = "+"{:1.3f}".format(wall_speed[k]*viscosity/surf_tens)
    label_text = r"Ca$_{cl}$ = "+"{:1.4f}".format(wall_speed[k]*viscosity/surf_tens-Ca_r[k])

    dt = time[k][1]-time[k][0]
    i_cut = int(spinup/dt)
    i_avg = int(avg_window[k]/dt)

    nh = int(0.2*len(time[k]))

    
    ax1.plot( time[k], -bot_left[k]+bot_left[k][0], 'k.')
    if k=='c005' :
        ax1.plot( time[k][nh:], np.polyval(coeff[k],time[k][nh:]), color_line[k]+'--',
            linewidth=5.0, label=label_wall )
    
    speed_ratio = coeff[k][0]/wall_speed[k]
    print("U_w = "+str(wall_speed[k])+"; U_cl/U_w = "+str(speed_ratio))
    print("Relative c.l. Ca = "+str(Ca_r[k])+" +/ "+str(Ca_r_err[k]) )

    angle = np.rad2deg(np.arctan(coeff[k]))
    if k=='c003' :
        # ax1.text(30, 10, s=r'$\alpha$')
        # ax1.text(28, 10, s=label_text, fontsize=25, rotation=angle, rotation_mode='anchor', transform_rotates_text=True)
        # ax1.text(28, 10, s=label_text, fontsize=25)
        print("")
    elif k=='c005' :
        # ax1.text(35, 30, s=label_text, fontsize=25, rotation=angle, rotation_mode='anchor', transform_rotates_text=True)
        ax1.text(35, 30, s=label_text, fontsize=30)
        ax1.text(10, 11.4, s='(a)', fontsize=30)
        ax1.text(20, 21.8, s='(b)', fontsize=30)
        ax1.text(30, 26.3, s='(c)', fontsize=30)
        ax1.text(40, 38.0, s='(d)', fontsize=30)
    elif k=='c008' :
        # ax1.text(20, 38, s=label_text, fontsize=25, rotation=angle, rotation_mode='anchor', transform_rotates_text=True)
        # ax1.text(20, 38, s=label_text, fontsize=25)
        print("")
    else :
        # ax1.text(1, 33, s=label_text, fontsize=25, rotation=angle, rotation_mode='anchor', transform_rotates_text=True)
        # ax1.text(1, 33, s=label_text, fontsize=25)
        print("")

ax1.legend(fontsize=37.5, loc='best')
ax1.plot([0,50],[0,0], 'k:', linewidth=3.5)
ax1.tick_params(axis="both", labelsize=30)
ax1.set_xlabel(r'$t$ [ns]', fontsize=37.5)
ax1.set_ylabel(r'$x_{cl}$ [nm]', fontsize=37.5)
ax1.set_xlim([0,50])
ax1.set_ylim([-3,56])
ax1.set_aspect(0.6441075720228246*50/(56+3))
plt.show()

f, (ax2) = plt.subplots()

for k in folders.keys() :

    # ax2.plot(wall_speed[k]*viscosity/surf_tens, Ca_r[k], 'ko', markersize=12.5, \
    #     markerfacecolor='none', markeredgewidth=4)
    ax2.errorbar(wall_speed[k]*viscosity/surf_tens, Ca_r[k], yerr=Ca_r_err[k], \
        fmt='ko', elinewidth=5.5, capsize=10, capthick=5, markersize=20, markerfacecolor=None)

ax2.plot([0.01, 0.055], [0.0125, 0.0125], 'r-', linewidth=5.0, label=r"Ca$_{cr}$")
ax2.plot([0.01, 0.055], [0.0125+0.00245, 0.0125+0.00245], 'r--')
ax2.plot([0.01, 0.055], [0.0125-0.00245, 0.0125-0.00245], 'r--')
ax2.set_ylim([0.0, 0.0175])
ax2.set_xlim([0.01, 0.055])
ax2.tick_params(axis="both", labelsize=25)
ax2.set_xlabel(r'Ca$_w$ [1]', fontsize=37.5)
ax2.set_ylabel(r'Ca$_w$-Ca$_{cl}$ [1]', fontsize=37.5)
ax2.legend(fontsize=37.5)
ax2.xaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
ax2.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
ax2.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
ax2.xaxis.get_offset_text().set_fontsize(27.5)
ax2.yaxis.get_offset_text().set_fontsize(27.5)
ax2.set_aspect(0.6876176417990977*(0.055-0.01)/0.0175)
plt.show()