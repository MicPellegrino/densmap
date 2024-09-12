import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import numpy.random as rng

def array_from_file( filename ):
    my_list = []
    with open(filename, 'r') as f:
        for line in f:
            my_list.append(float(line.split()[0]))
    return np.array(my_list)

def fit_bootstrap(f, x, y, a=0.8, k=10, err=None) :

    popt, _ = opt.curve_fit(f, x, y, sigma=err)

    n = len(x)
    m = len(popt)
    popt_vec = np.zeros((k,m))
    for i in range(k) :
        perm = np.arange(n)
        rng.shuffle(perm)
        perm = perm[:int(n*a)]
        temp, _ = opt.curve_fit(f, x[perm], y[perm], sigma=err[perm])
        popt_vec[i,:] = temp

    perr = np.std(popt_vec, axis=0)

    return popt, 5*perr

file_names_jump = ["p_jump_38eq.txt","p_jump_38adv.txt","p_jump_38rec.txt"]
file_names_roll = ["p_roll_38eq.txt","p_roll_38adv.txt","p_roll_38rec.txt"]
file_names_jump_err = ["p_jump_38eq_err.txt","p_jump_38adv_err.txt","p_jump_38rec_err.txt"]
file_names_roll_err = ["p_roll_38eq_err.txt","p_roll_38adv_err.txt","p_roll_38rec_err.txt"]
file_names_time = ["tlag_vec_38eq.txt","tlag_vec_38adv.txt","tlag_vec_38rec.txt"]
titles = [r'(Ca$_{cl}$=0.0)', r'(Ca$_{cl}$=0.02)', r'(Ca$_{cl}$=-0.02)']

assert len(file_names_jump) == len(file_names_roll) \
        and len(file_names_roll) == len(file_names_time) \
        and len(file_names_time) == len(titles), \
        "Input file arrays have different length!"

lags = array_from_file(file_names_time[0])
t_sat_roll = int(1*60)
t_sat_jump = int(1*85)
t_sat_plot = 100
n_sat_roll = int(t_sat_roll/(lags[1]-lags[0]))
n_sat_jump = int(t_sat_jump/(lags[1]-lags[0]))
n_sat_plot = int(t_sat_plot/(lags[1]-lags[0]))
exp_rate = lambda t, a, tau : a*(1-np.exp(-t/tau))

n_plots = len(file_names_jump)

"""

fig, ax = plt.subplots(n_plots, 1)

for n in range(n_plots) :

    ax[n].set_title(titles[n])

    ax[n].loglog(array_from_file(file_names_time[n]), 
        1-array_from_file(file_names_jump[n]), 'r-', label='jump')

    ax[n].loglog(array_from_file(file_names_time[0]), 
        1-array_from_file(file_names_roll[n]), 'b-', label='roll')

    ax[n].set_xlabel(r'$t_{lag}$ [ps]')

    if n==0 :
        ax[n].set_ylabel(r'$p_{ij}$')
        ax[n].legend()

plt.show()

"""

fig, ax = plt.subplots(n_plots, 1)

for n in range(n_plots) :

    """
    popt_jump, pcov_jump = opt.curve_fit(exp_rate, lags[1:n_sat_jump], 
        array_from_file(file_names_jump[n])[1:n_sat_jump],
        sigma=array_from_file(file_names_jump_err[n])[1:n_sat_jump])
    popt_roll, pcov_roll = opt.curve_fit(exp_rate, lags[1:n_sat_roll], 
        array_from_file(file_names_roll[n])[1:n_sat_roll],
        sigma=array_from_file(file_names_roll_err[n])[1:n_sat_roll])
    """
    popt_jump, perr_jump = fit_bootstrap(exp_rate, lags[1:n_sat_jump], 
        array_from_file(file_names_jump[n])[1:n_sat_jump],
        err=array_from_file(file_names_jump_err[n])[1:n_sat_jump])
    popt_roll, perr_roll = fit_bootstrap(exp_rate, lags[1:n_sat_roll], 
        array_from_file(file_names_roll[n])[1:n_sat_roll],
        err=array_from_file(file_names_roll_err[n])[1:n_sat_roll])

    # ax[n].set_title(titles[n])

    A = popt_jump[0]
    # A_err = pcov_jump[0][0]
    A_err = perr_jump[0]
    tau = popt_jump[1]
    # tau_err = pcov_jump[1][1]
    tau_err = perr_jump[1]

    print("# # #")
    print(file_names_jump[n])
    print("tau = "+str(tau)+" +/- "+str(tau_err))
    print("A = "+str(A)+" +/- "+str(A_err))

    ax[n].plot(array_from_file(file_names_time[n]), 
        array_from_file(file_names_jump[n]), 'r-', label='jump', linewidth=1.5)
    ax[n].plot(array_from_file(file_names_time[n])[:n_sat_plot], 
        exp_rate(lags, *popt_jump)[:n_sat_plot], 'k:', linewidth=3.0)

    ax[n].fill_between(array_from_file(file_names_time[n]),
        array_from_file(file_names_jump[n])+array_from_file(file_names_jump_err[n]),
        array_from_file(file_names_jump[n])-array_from_file(file_names_jump_err[n]),
        color='red', alpha=0.25)

    A = popt_roll[0]
    # A_err = pcov_roll[0][0]
    A_err = perr_roll[0]
    tau = popt_roll[1]
    # tau_err = pcov_roll[1][1]
    tau_err = perr_roll[1]

    print("# # #")
    print(file_names_roll[n])
    print("tau = "+str(tau)+" +/- "+str(tau_err))
    print("A = "+str(A)+" +/- "+str(A_err))

    ax[n].plot(array_from_file(file_names_time[0]), 
        array_from_file(file_names_roll[n]), 'b-', label='roll', linewidth=1.5)
    ax[n].plot(array_from_file(file_names_time[n])[:n_sat_plot], 
        exp_rate(lags, *popt_roll)[:n_sat_plot], 'k:', linewidth=3.0)

    ax[n].fill_between(array_from_file(file_names_time[n]),
        array_from_file(file_names_roll[n])+array_from_file(file_names_roll_err[n]),
        array_from_file(file_names_roll[n])-array_from_file(file_names_roll_err[n]),
        color='blue', alpha=0.25)

    ax[n].set_ylabel(titles[n]+"\n"+r'$p_{ij}$', fontsize=27.5)

    if n==0 :
        ax[n].legend(fontsize=25, loc="upper right")
        ax[n].set_title(r"$\theta_0=38^\circ$", fontsize=30)
    if n==2 :
        ax[n].set_xlabel(r'$t_{lag}$ [ps]', fontsize=30)
    ax[n].set_ylim([0.0, 0.15])
    ax[n].set_xlim([0.0, array_from_file(file_names_time[n])[-1]])

    ax[n].tick_params(labelsize=20)

plt.show()

print("# # #")

"""

# Exponential tilting from eq. and jumping/rolling
exc_roll_adv = array_from_file(file_names_roll[1])/array_from_file(file_names_roll[0])
exc_roll_rec = array_from_file(file_names_roll[2])/array_from_file(file_names_roll[0])
exc_jump_adv = array_from_file(file_names_jump[1])/array_from_file(file_names_jump[0])
exc_jump_rec = array_from_file(file_names_jump[2])/array_from_file(file_names_jump[0])

fig, ax = plt.subplots(1, 2)

ax[0].step(array_from_file(file_names_time[0]), exc_roll_adv, 'b-')
ax[0].step(array_from_file(file_names_time[0]), exc_jump_adv, 'r-')
#ax[0].fill_betweenx([0.60, 1.4], [40,40], [85,85], color='lightgrey', alpha=0.5)
ax[0].set_ylim([0.60, 1.4])

ax[1].step(array_from_file(file_names_time[0]), exc_roll_rec, 'b-')
ax[1].step(array_from_file(file_names_time[0]), exc_jump_rec, 'r-')
# ax[1].fill_betweenx([0.60, 1.4], [40,40], [85,85], color='lightgrey', alpha=0.5)
ax[1].set_ylim([0.60, 1.4])

plt.show()

"""