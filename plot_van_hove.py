import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

lags = np.logspace(0,3,10,dtype=int)

folder_name = 'VanHove'

vhx = dict()
vhz = dict()

fig, (ax1,ax2) = plt.subplots(1,2)

n = 0

for k in lags :

    n += 1

    x, vhx[k] = np.loadtxt(folder_name+'/vanhove_x_'+str(k).zfill(4)+'.txt', unpack=True)
    z, vhz[k] = np.loadtxt(folder_name+'/vanhove_z_'+str(k).zfill(4)+'.txt', unpack=True)
    ax1.plot(x, vhx[k], c=cm.hot((len(lags)-n)/len(lags)))
    ax2.plot(z, vhz[k], c=cm.hot((len(lags)-n)/len(lags)))

plt.show()
