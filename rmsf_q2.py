import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt
import matplotlib.pylab as plab

def array_from_file( filename ):
    my_list = []
    with open(filename, 'r') as f:
        for line in f:
            my_list.append(float(line.split()[0]))
    return np.array(my_list)

def cross_validation_rmsf(x, kmax=25) :
    std_vec = []
    for k in range(kmax) :
        p = rng.permutation(len(x))
        x_train = x[p[0:int(0.8*len(x))]]
        std_vec.append(np.std(x_train))
    std_vec = np.array(std_vec)
    return np.max(std_vec)-np.min(std_vec)

folder_pattern = '/home/michele/densmap/ShearDynamic/Q2_Ca'

k_vec       = ['005', '010', '015', '020', '025', '0275', '030', '040']

capillary   = [0.05, 0.1, 0.15, 0.20, 0.25, 0.275, 0.30, 0.40]

receding_left   = dict()
receding_right  = dict()
time            = dict()
com             = dict()

tini = dict()
tfin = dict()

tini['005']  = 2500
tini['010']  = 2500
tini['015']  = 2500
tini['020']  = 2500
tini['025']  = 6500
tini['0275'] = 6500
tini['030']  = 2500
tini['040']  = 2500

tfin['005']  = None
tfin['010']  = None
tfin['015']  = None
tfin['020']  = None
tfin['025']  = None
tfin['0275'] = 14000
tfin['030']  = 9000
tfin['040']  = 3400

labels=dict()
labels['005']  = r'Ca$_{w}$ = 0.05'
labels['010']  = r'Ca$_{w}$ = 0.10'
labels['015']  = r'Ca$_{w}$ = 0.15'
labels['020']  = r'Ca$_{w}$ = 0.20'
labels['025']  = r'Ca$_{w}$ = 0.25'
labels['0275'] = r'Ca$_{w}$ = 0.275'
labels['030']  = r'Ca$_{w}$ = 0.30'
labels['040']  = r'Ca$_{w}$ = 0.40'

fluctuationl    = dict()
fluctuationr    = dict()
rmsfl           = []
rmsf_errl       = []
rmsfr           = []
rmsf_errr       = []

n = 8
colors = plab.cm.gnuplot(np.linspace(0,1,n))

fig, (ax1) = plt.subplots()

i = 0
for k in k_vec :

    receding_left[k]    = array_from_file(folder_pattern+k+'/position_lower.txt')-0.5*array_from_file(folder_pattern+k+'/radius_lower.txt')
    receding_right[k]   = array_from_file(folder_pattern+k+'/position_upper.txt')+0.5*array_from_file(folder_pattern+k+'/radius_upper.txt')
    time[k]             = array_from_file(folder_pattern+k+'/time.txt')
    com[k]              = array_from_file(folder_pattern+k+'/xcom.txt')

    iini = np.argmin(np.abs(time[k]-tini[k]))

    if tfin[k] == None :
        ifin = len(time[k])-1
    else :
        ifin = np.argmin(np.abs(time[k]-tfin[k]))

    t = time[k][iini:ifin]-time[k][iini]
    xl = -receding_left[k][iini:ifin]+com[k][iini:ifin]
    xr = receding_right[k][iini:ifin]-com[k][iini:ifin]
    xl = xl-xl[0]
    xr = xr-xr[0]

    pcoeffl = np.polyfit(t, xl, deg=1)
    trendl = np.polyval(pcoeffl, t)
    fluctuationl[k] = xl-trendl

    pcoeffr = np.polyfit(t, xr, deg=1)
    trendr = np.polyval(pcoeffr, t)
    fluctuationr[k] = xr-trendr

    # plt.plot(t, x, '.', label=k)
    # plt.plot(t, trend, 'k-')
    ax1.plot(t[-1]*(1e-3), fluctuationl[k][-1], 's', label=labels[k], color=colors[i], markersize=20)
    ax1.plot(t*(1e-3), fluctuationl[k], '.', color=colors[i])

    rmsfl.append( np.std(fluctuationl[k]) )
    rmsf_errl.append( cross_validation_rmsf(fluctuationl[k]) )

    rmsfr.append( np.std(fluctuationr[k]) )
    rmsf_errr.append( cross_validation_rmsf(fluctuationr[k]) )

    print(k + ", RMS = " + str(rmsfl[-1]) + " +/- " + str(rmsf_errl[-1]) + " nm")
    print(k + ", RMS = " + str(rmsfr[-1]) + " +/- " + str(rmsf_errr[-1]) + " nm")

    i+=1

ax1.legend(fontsize=20)
ax1.set_xlabel(r'$t$ [ns]', fontsize=25)
ax1.set_ylabel(r'$x_{cl}$ detrend [nm]', fontsize=25)
ax1.tick_params('both', labelsize=20)
plt.show()

fig, (ax2) = plt.subplots()

l_th = 0.26763089802534887
vdw_spce = 0.3166
d_sio2 = 0.45

"""
ax2.errorbar(capillary, rmsfl, yerr=rmsf_errl, fmt='rD', elinewidth=2, capsize=5, capthick=2, markersize=10, markerfacecolor=None, label='left int.')
ax2.errorbar(capillary, rmsfr, yerr=rmsf_errr, fmt='bs', elinewidth=2, capsize=5, capthick=2, markersize=10, markerfacecolor=None, label='right int.')
"""

rmsfl = np.array(rmsfl)
rmsfr = np.array(rmsfr)
rmsf_errl = np.array(rmsf_errl)
rmsf_errr = np.array(rmsf_errr)
ax2.errorbar(capillary, 0.5*(rmsfl+rmsfr), yerr=rmsf_errl+rmsf_errr, fmt='ko', elinewidth=5.5, capsize=10, capthick=5, markersize=20, markerfacecolor=None)
ax2.plot([0.2625, 0.2625], [0.0, 1.2], 'r--', linewidth=5.0, label=r'Ca$_{cr}$')
ax2.plot([0.03, 0.42], [d_sio2, d_sio2], 'b--', linewidth=5.0, label=r'SiO$_2$ lattice spacing')
ax2.legend(fontsize=37.5)
ax2.set_ylim([0.0, 1.2])
ax2.set_xlim([0.03, 0.42])
ax2.set_xlabel(r'Ca$_{cl}$ [1]', fontsize=37.5)
ax2.set_ylabel('RMSF [nm]', fontsize=37.5)
ax2.tick_params('both', labelsize=30)
# ax2.set_aspect((0.42-0.03)/1.2)
ax2.set_aspect(0.5919805845826159*(0.42-0.03)/1.2)
plt.show()