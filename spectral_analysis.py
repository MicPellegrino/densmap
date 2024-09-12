import numpy as np
import scipy as sc
import scipy.fftpack
import scipy.signal
import matplotlib.pyplot as plt

def array_from_file( filename ):
    my_list = []
    with open(filename, 'r') as f:
        for line in f:
            my_list.append(float(line.split()[0]))
    return np.array(my_list)

# Read contact lines from file
folder_roots = ['EquilDpQ2', 'Q2Ca005', 'Q2Ca010', 'Q2Ca015', 'Q2Ca020', 'Q2Ca025']
int_labels = ['tl', 'tr', 'bl', 'br', 'ml', 'mr']

x_int = dict()
X_fft = dict()
freqs = dict()
X_norm = dict()

# In units of picoseconds, the samplig frequency is just:
dt = 12.5   # [ps]
f_s = 1/dt

for i in folder_roots :
    x_int[i] = dict()
    X_fft[i] = dict()
    freqs[i] = dict()
    X_norm[i] = dict()
    for j in int_labels :
        x_int[i][j] = array_from_file('ContactLinesSignals/'+i+'/'+i+'_'+j+'.txt')
        std = np.std(x_int[i][j])
        x_int[i][j] = sc.signal.detrend(x_int[i][j]/std)
        X_fft[i][j] = sc.fftpack.fft(x_int[i][j])
        freqs[i][j] = sc.fftpack.fftfreq(len(x_int[i][j])) * f_s
        X_fft[i][j] = X_fft[i][j][ 1 : int(0.5*len(X_fft[i][j])) ]
        freqs[i][j] = freqs[i][j][ 1 : int(0.5*len(freqs[i][j])) ]
        dw = freqs[i][j][1]-freqs[i][j][0]
        X_norm[i][j] = np.abs(X_fft[i][j])
        X_norm[i][j] /= np.sum(dw*X_norm[i][j])

ifps=['tr','tl']

nax = 0
fig, ax = plt.subplots(2, 2)

for interface_point in ifps :

    ratio_n = len(freqs['Q2Ca025'][interface_point])/len(freqs['EquilDpQ2'][interface_point])
    cast_n = int(np.floor(ratio_n))
    rest_n = len(freqs['Q2Ca025'][interface_point])%cast_n
    print(cast_n)
    print(rest_n)
    freqs['Q2Ca025'][interface_point] = freqs['Q2Ca025'][interface_point][0:-rest_n]
    freqs['Q2Ca025'][interface_point] = freqs['Q2Ca025'][interface_point].reshape(-1, cast_n).mean(axis=1)
    X_norm['Q2Ca025'][interface_point] = X_norm['Q2Ca025'][interface_point][0:-rest_n]
    X_norm['Q2Ca025'][interface_point] = X_norm['Q2Ca025'][interface_point].reshape(-1, cast_n).mean(axis=1)
    dw_new = freqs['Q2Ca025'][interface_point][1]-freqs['Q2Ca025'][interface_point][0]
    X_norm['Q2Ca025'][interface_point] = X_norm['Q2Ca025'][interface_point]/np.sum(dw_new*X_norm['Q2Ca025'][interface_point])

    dw_equil = freqs['EquilDpQ2'][interface_point][1]-freqs['EquilDpQ2'][interface_point][0]
    print(np.sum(dw_new*X_norm['Q2Ca025'][interface_point]))
    print(np.sum(dw_equil*X_norm['EquilDpQ2'][interface_point]))

    min_len = min( len(x_int['Q2Ca025'][interface_point]), len(x_int['EquilDpQ2'][interface_point]) )
    ax[nax][0].plot(dt*np.linspace(0,min_len,min_len), x_int['EquilDpQ2'][interface_point][0:min_len], 'r-')
    ax[nax][0].plot(dt*np.linspace(0,min_len,min_len), x_int['Q2Ca025'][interface_point][0:min_len], 'b-')
    if nax == 1 :
        ax[nax][0].set_xlabel(r'$t$ [ps]', fontsize=30.0)
    ax[nax][0].set_ylabel(r'detr. $\tilde{x}$ [1]', fontsize=30.0)
    ax[nax][0].tick_params(axis='both', labelsize=20)
    if nax == 0 :
        ax[nax][0].set_title('(a)', fontsize=30.0, x=-0.015)
    if nax == 1 :
        ax[nax][0].set_title('(c)', fontsize=30.0, x=-0.015)

    # ax[nax][1].fill_between(freqs['Q2Ca025']['bl'], X_norm['Q2Ca025']['bl'], color='blue', step="pre", alpha=0.4, label='Ca=0.25')
    # ax[nax][1].fill_between(freqs['EquilDpQ2']['bl'], X_norm['EquilDpQ2']['bl'], color='red', step="pre", alpha=0.4, label='Equil.')
    # ax[nax][1].step(freqs['EquilDpQ2'][interface_point], X_norm['EquilDpQ2'][interface_point], 'r-', label='Equil.')
    # ax[nax][1].step(freqs['Q2Ca025'][interface_point], X_norm['Q2Ca025'][interface_point], 'b-', label='Ca=0.25')
    # ax[nax][1].set_yscale('log')
    # ax[nax][1].set_xscale('log')
    ax[nax][1].loglog(freqs['EquilDpQ2'][interface_point], X_norm['EquilDpQ2'][interface_point], 'r-', label='Equil.')
    ax[nax][1].loglog(freqs['Q2Ca025'][interface_point], X_norm['Q2Ca025'][interface_point], 'b-', label='Ca=0.25')

    if nax == 1 :
        ax[nax][1].set_xlabel(r'$\omega$ [1/ps]', fontsize=30.0)
    ax[nax][1].set_ylabel(r'$|FFT(\tilde{x})|$ [1]', fontsize=30.0)
    ax[nax][1].tick_params(axis='both', labelsize=25)
    if nax == 0 :
        ax[nax][1].legend(fontsize=30.0)
    if nax == 0 :
        ax[nax][1].set_title('(b)', fontsize=30.0, x=-0.015)
    if nax == 1 :
        ax[nax][1].set_title('(d)', fontsize=30.0, x=-0.015)

    nax+=1

"""
ax[0][1].loglog([7.3e-5, 0.0003], [475.88, 475.88], 'r--', linewidth=3.0)
ax[0][1].loglog([7.3e-5, 0.0003], [1250.1, 1250.1], 'b--', linewidth=3.0)
ax[0][1].text(1e-4, 275.06, '475.88',  fontsize=17.5, fontweight='bold')
ax[0][1].text(1e-4, 1450.06, '1250.1', fontsize=17.5, fontweight='bold')

ax[1][1].loglog([7.3e-5, 0.0003], [204.06, 204.06], 'r--', linewidth=3.0)
ax[1][1].loglog([7.3e-5, 0.0003], [912.06, 912.06], 'b--', linewidth=3.0)
ax[1][1].text(1e-4, 119.06, '204.82',  fontsize=17.5, fontweight='bold')
ax[1][1].text(1e-4, 1050.06, '912.06', fontsize=17.5, fontweight='bold')
"""

plt.show()