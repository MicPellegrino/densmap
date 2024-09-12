import densmap as dm 
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

folder = 'VelocityProfileHexane'
filenames = ['HexCa006q60','HexCa008q60','HexCa010q60','HexCa012q60']

tracking_flies = [  'ShearWatHex/C006Q60/position_lower.txt',
                    'ShearWatHex/C008Q60/position_lower.txt',
                    'ShearWatHex/C010Q60/position_lower.txt',
                    'ShearWatHex/C012Q60/position_lower.txt']

delta_lambda = np.array([4.469565217391304e-06,5.959420289855072e-06,7.4492753623188395e-6,8.939130434782607e-06])
dt = 0.002
dx_res = 1.0
v_wall = dx_res*delta_lambda/dt

vx_scaled_list = []
vz_scaled_list = []

shift = 0
dshift = 0
# cl_displacement = []

for i in range(len(filenames)) :

    npzfile = np.load(folder+'/'+filenames[i]+'_vx.npz')
    x = npzfile['arr_0']

    # posx = np.loadtxt(tracking_flies[i])
    # posx_avg = np.mean(posx[2*len(posx)//3:])
    # j_ref = np.argmin(np.abs(x-posx_avg))-len(x)//2

    j_ref = int(shift/(x[1]-x[0]))
    shift += dshift

    vx_scaled = npzfile['arr_1']/v_wall[i]
    vx_scaled = np.roll(vx_scaled, j_ref)
    vx_scaled_list.append(vx_scaled)

    npzfile = np.load(folder+'/'+filenames[i]+'_vz.npz')
    x = npzfile['arr_0']
    vz_scaled = npzfile['arr_1']/v_wall[i]
    vz_scaled = np.roll(vz_scaled, j_ref)
    vz_scaled_list.append(vz_scaled)

    if i == 0 :
        plt.plot(x, vx_scaled, 'ro', label=r'$u_x$/$u_w$')
        plt.plot(x, vz_scaled, 'bo', label=r'$u_z$/$u_w$')
    else :
        plt.plot(x, vx_scaled, 'ro')
        plt.plot(x, vz_scaled, 'bo')

plt.plot(x, -np.ones(x.shape), 'k--', linewidth=4)
plt.xlim([x[0],x[-1]])

plt.legend(fontsize=30)
plt.ylabel(r'$u$/$u_w$', fontsize=35)
plt.xlabel(r'$x$ [nm]', fontsize=35)
plt.tick_params(axis='both',labelsize=25)

plt.show()