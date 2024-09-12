import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

import numpy as np
import densmap as dm
from matplotlib import cm
import scipy.optimize as opt

# Function for detecting the interface
def detect_interface(darray,z0,nwin=10):
    i0 = np.abs(z-z0).argmin()
    imax = Nz-i0+1
    branch = np.zeros((2,imax-i0),dtype=float)
    for j in range(i0, imax) :
        branch[1,j-i0] = z_fold[j]
        dtar = 0.5*max(np.mean(darray[:nwin,j]),np.mean(darray[-nwin:,j]))
        for i in range(1,Nx//2) :
            if darray[i,j] > dtar and darray[i-1,j] < dtar or darray[i,j] < dtar and darray[i-1,j] > dtar :
                branch[0,j-i0] = ((darray[i,j]-dtar)*x[i-1]+(dtar-darray[i-1,j])*x[i])/(darray[i,j]-darray[i-1,j])
                break
    return branch

def arccot(x) :
    return 0.5*np.pi-np.arctan(x)

FP = dm.fitting_parameters( par_file='parameters_shear_hex.txt' )
liq1 = 'SOL'
liq2 = 'HEX'
file_root = 'flow_'

""" Reading input """
folder_name = FP.folder_name
Lx = FP.lenght_x
Lz = FP.lenght_z
n_init = FP.first_stamp
n_fin = FP.last_stamp
dt = FP.time_step

""" Creating meshgrid """
density_array = dm.read_density_file(folder_name+'/'+file_root+liq1+'_{:05d}'.format(n_init)+'.dat', bin='y')
Nx = density_array.shape[0]
Nz = density_array.shape[1]
hx = Lx/Nx
hz = Lz/Nz
x = hx*np.arange(0.0,Nx,1.0, dtype=float)+0.5*hx
z = hz*np.arange(0.0,Nz,1.0, dtype=float)+0.5*hz
X, Z = np.meshgrid(x, z, sparse=False, indexing='ij')
ihalf = (Nx//2)+(Nx%2)
density_array_sol_avg = np.zeros_like(density_array)
density_array_but_avg = np.zeros_like(density_array)

x_fold = hx*np.arange(0.0,Nx//2,1.0, dtype=float)+0.5*hx
z_fold = z
X_fold, Z_fold = np.meshgrid(x_fold, z_fold, sparse=False, indexing='ij')

n_dump = 10

for idx in range(n_init, n_fin+1 ):
    
    if idx%n_dump==0 :
        print("Obtainig frame "+str(idx))
        t_label = str(dt*idx)+' ps'

    density_array = dm.read_density_file(folder_name+'/'+file_root+liq1+'_{:05d}'.format(idx)+'.dat', bin='y')
    
    density_x = np.mean(density_array,axis=1)
    xcom = np.sum(density_x*x)/np.sum(density_x)
    icom = int(np.round(xcom/hx))
    ishift = ihalf-icom
    
    density_array = np.roll(density_array, ishift, axis=0)
    density_array_sol_avg += density_array

    density_array = dm.read_density_file(folder_name+'/'+file_root+liq2+'_{:05d}'.format(idx)+'.dat', bin='y')
    density_x = np.mean(density_array,axis=1)
    
    density_array = np.roll(density_array, ishift, axis=0)
    density_array_but_avg += density_array
    
density_array_sol_avg /= (n_fin-n_init+1)
density_array_but_avg /= (n_fin-n_init+1)
density_array_tot_avg = density_array_sol_avg+density_array_but_avg

xwin_wat = np.array([30,50])
zwin_wat = np.array([5,15])
xwin_but1 = np.array([0,10])
xwin_but2 = np.array([70,80])
zwin_but = np.array([5,15])
nxwin_wat = np.array(xwin_wat/hx,dtype=int)
nzwin_wat = np.array(zwin_wat/hz,dtype=int)
nxwin_but1 = np.array(xwin_but1/hx,dtype=int)
nxwin_but2 = np.array(xwin_but2/hx,dtype=int)
nzwin_but = np.array(zwin_but/hz,dtype=int)
rho_wat = 1.66054*np.mean(density_array_sol_avg[nxwin_wat[0]:nxwin_wat[1],nzwin_wat[0]:nzwin_wat[1]])
rho_but1 = 1.66054*np.mean(density_array_but_avg[nxwin_but1[0]:nxwin_but1[1],nzwin_but[0]:nzwin_but[1]])
rho_but2 = 1.66054*np.mean(density_array_but_avg[nxwin_but2[0]:nxwin_but2[1],nzwin_but[0]:nzwin_but[1]])
rho_but = 0.5*(rho_but1+rho_but2)

z0_ref = 1.5
i_local_ca = 3
density_array_sol_avglr = 0.5*(density_array_sol_avg[:Nx//2,:]+
                               np.flipud(np.fliplr(density_array_sol_avg[-Nx//2+1:,:])))
branch_sol = detect_interface(density_array_sol_avglr,z0=z0_ref)
density_array_but_avglr = 0.5*(density_array_but_avg[:Nx//2,:]+
                               np.flipud(np.fliplr(density_array_but_avg[-Nx//2+1:,:])))
branch_but = detect_interface(density_array_but_avglr,z0=z0_ref)
density_array_tot_avglr = 0.5*(density_array_tot_avg[:Nx//2,:]+
                               np.flipud(np.fliplr(density_array_tot_avg[-Nx//2+1:,:])))

### TANGENT-BASED ###
npoint = len(branch_sol[0,:])
m_sol = []
m_but = []
m_sol.append(np.polyfit(branch_sol[0,:3],branch_sol[1,:3],deg=1)[0])
m_but.append(np.polyfit(branch_but[0,:3],branch_but[1,:3],deg=1)[0])
for i in range(1,npoint-1) :
    m_sol.append(np.polyfit(branch_sol[0,i-1:i+2],branch_sol[1,i-1:i+2],deg=1)[0])
    m_but.append(np.polyfit(branch_but[0,i-1:i+2],branch_but[1,i-1:i+2],deg=1)[0])
m_sol.append(np.polyfit(branch_sol[0,-3:],branch_sol[1,-3:],deg=1)[0])
m_but.append(np.polyfit(branch_but[0,-3:],branch_but[1,-3:],deg=1)[0])
m_sol = np.array(m_sol)
m_but = np.array(m_but)
theta_sol = np.abs(np.rad2deg(np.arctan(m_sol)))
theta_but = np.abs(np.rad2deg(np.arctan(m_but)))

peq = np.polyfit(branch_sol[1,:],theta_sol,deg=1,full=False,cov=True)

x_ca_sol_bot = branch_sol[0,:i_local_ca]
z_ca_sol_bot = branch_sol[1,:i_local_ca]
x_ca_sol_top = branch_sol[0,-i_local_ca:]
z_ca_sol_top = branch_sol[1,-i_local_ca:]
x_ca_but_bot = branch_but[0,:i_local_ca]
z_ca_but_bot = branch_but[1,:i_local_ca]
x_ca_but_top = branch_but[0,-i_local_ca:]
z_ca_but_top = branch_but[1,-i_local_ca:]

### TANGENT-BASED ###
coeff_sol_bot, cov_sol_bot = np.polyfit(x_ca_sol_bot,z_ca_sol_bot,deg=1,full=False,cov=True)
coeff_sol_top, cov_sol_top = np.polyfit(x_ca_sol_top,z_ca_sol_top,deg=1,full=False,cov=True)
coeff_but_bot, cov_but_bot = np.polyfit(x_ca_but_bot,z_ca_but_bot,deg=1,full=False,cov=True)
coeff_but_top, cov_but_top = np.polyfit(x_ca_but_top,z_ca_but_top,deg=1,full=False,cov=True)
slope_sol_bot = coeff_sol_bot[0]
slope_sol_top = coeff_sol_top[0]
slope_but_bot = coeff_but_bot[0]
slope_but_top = coeff_but_top[0]
dslope_sol_bot = np.sqrt(cov_sol_bot[0][0])
dslope_sol_top = np.sqrt(cov_sol_top[0][0])
dslope_but_bot = np.sqrt(cov_but_bot[0][0])
dslope_but_top = np.sqrt(cov_but_top[0][0])

tickfs = 30
labelfs = 35
cmaplfs = 32.5
cmaptfs = 27.5

# dmap1 = plt.pcolormesh(X_fold, Z_fold, 1.66054*density_array_sol_avglr, cmap=cm.Blues)
dmap1 = plt.pcolormesh(X_fold, Z_fold, 1.66054*density_array_tot_avglr, cmap=cm.bone, vmin=100)
plt.plot(branch_sol[0,:],branch_sol[1,:],'k-',linewidth=5)
plt.plot(branch_sol[0,:i_local_ca],branch_sol[1,:i_local_ca],'kx',markersize=15,markeredgewidth=4.0)
plt.plot(branch_sol[0,-i_local_ca:],branch_sol[1,-i_local_ca:],'kx',markersize=15,markeredgewidth=4.0)
plt.tick_params(axis='both',labelsize=tickfs)
plt.xlabel(r'$x$ [nm]',fontsize=labelfs)
plt.ylabel(r'$z$ [nm]',fontsize=labelfs)
cb1 = plt.colorbar(dmap1)
# cb1.set_label(r'$\rho_w$ [kg/m$^3$]',labelpad=20,fontsize=30)
cb1.set_label(r'$\rho$ [kg/m$^3$]',labelpad=0,fontsize=cmaplfs)
cb1.ax.tick_params(labelsize=cmaptfs)
plt.xlim([14,18])
plt.ylim([0,7.5])
plt.show()