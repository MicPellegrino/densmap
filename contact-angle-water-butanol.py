import numpy as np
import matplotlib.pyplot as plt
import densmap as dm
from matplotlib import cm
import scipy.optimize as opt

# FP = dm.fitting_parameters( par_file='parameters_density.txt' )
FP = dm.fitting_parameters( par_file='parameters_shear.txt' )

liq1 = 'SOL'
liq2 = 'HEX'

folder_name = FP.folder_name
file_root = 'flow_'

Lx = FP.lenght_x
Lz = FP.lenght_z

n_init = FP.first_stamp
n_fin = FP.last_stamp
dt = FP.time_step

print("Creating meshgrid")
density_array = dm.read_density_file(folder_name+'/'+file_root+'SOL_{:05d}'.format(n_init)+'.dat', bin='y')
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
z_fold = hz*np.arange(0.0,Nz//2,1.0, dtype=float)+0.5*hz
X_fold, Z_fold = np.meshgrid(x_fold, z_fold, sparse=False, indexing='ij')

# Function for detecting the interface
def detect_interface(darray,z0,nwin=10):
    i0 = np.abs(z-z0).argmin()
    imax = Nz//2
    branch = np.zeros((2,imax-i0),dtype=float)
    for j in range(i0, imax) :
        branch[1,j-i0] = z_fold[j]
        dtar = 0.5*max(np.mean(darray[:nwin,j]),np.mean(darray[-nwin:,j]))
        for i in range(1,Nx//2) :
            if darray[i,j] > dtar and darray[i-1,j] < dtar or darray[i,j] < dtar and darray[i-1,j] > dtar :
                branch[0,j-i0] = ((darray[i,j]-dtar)*x[i-1]+(dtar-darray[i-1,j])*x[i])/(darray[i,j]-darray[i-1,j])
                break
    return branch

# Circle fit
def circle_fit(xi,zi,xm) :
    def calc_R(xc):
        """ calculate the distance of each 2D points from the center (xc, 0.5*Lz) """
        return np.sqrt((xi-xc)**2+(zi-0.5*Lz)**2)
    def f_2(xc):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, 0.5*Lz) """
        Ri = calc_R(xc)
        return Ri - Ri.mean()
    center_2, ier = opt.leastsq(f_2, xm)
    xc_2 = center_2
    Ri_2 = calc_R(center_2)
    R_2 = Ri_2.mean()
    return xc_2, R_2

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
    xcom = np.sum(density_x*x)/np.sum(density_x)
    icom = int(np.round(xcom/hx))
    ishift = ihalf-icom
    density_array = np.roll(density_array, -ishift, axis=0)
    density_array_but_avg += density_array

z0_ref = 1.7
theta = np.linspace(0,2*np.pi,360)
xm_ref = Lx/2
# xm_ref = 0
zc = Lz/2

density_array_sol_avg /= (n_fin-n_init+1)
density_array_sol_avg = 0.5*(density_array_sol_avg[:Nx//2,:]+np.flipud(density_array_sol_avg[-Nx//2+1:,:]))
density_array_sol_avg = 0.5*(density_array_sol_avg[:,:Nz//2]+np.fliplr(density_array_sol_avg[:,-Nz//2+1:]))

branch_sol = detect_interface(density_array_sol_avg,z0=z0_ref)
xcs, rs = circle_fit(branch_sol[0,:],branch_sol[1,:],xm_ref)
xs = rs*np.cos(theta)+xcs
zs = rs*np.sin(theta)+Lz/2

cot_circle = (z0_ref-zc)/np.sqrt(rs*rs-(z0_ref-zc)**2)
theta_circle = np.rad2deg( -np.arctan( cot_circle )+0.5*np.pi )
theta_circle = theta_circle + 180*(theta_circle<=0)
if xcs < Lx/4 :
    theta_circle = 180-theta_circle
print('theta_0 = '+str(theta_circle))

density_array_but_avg /= (n_fin-n_init+1)
density_array_but_avg = 0.5*(density_array_but_avg[:Nx//2,:]+np.flipud(density_array_but_avg[-Nx//2+1:,:]))
density_array_but_avg = 0.5*(density_array_but_avg[:,:Nz//2]+np.fliplr(density_array_but_avg[:,-Nz//2+1:]))

branch_but = detect_interface(density_array_but_avg,z0=z0_ref)
xcb, rb = circle_fit(branch_but[0,:],branch_but[1,:],xm_ref)
xb = rb*np.cos(theta)+xcb
zb = rb*np.sin(theta)+Lz/2

cot_circle = (z0_ref-zc)/np.sqrt(rb*rb-(z0_ref-zc)**2)
theta_circle = np.rad2deg( -np.arctan( cot_circle )+0.5*np.pi )
theta_circle = theta_circle + 180*(theta_circle<=0)
if xcb < Lx/4 :
    theta_circle = 180-theta_circle
print('theta_0 = '+str(theta_circle))

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

dmap1 = ax1.pcolormesh(X_fold, Z_fold, 1.66054*density_array_sol_avg, cmap=cm.bwr)
ax1.plot(branch_sol[0,:],branch_sol[1,:],'k-',linewidth=3)
ax1.plot(xs,zs,'g:',linewidth=3)
ax1.tick_params(axis='both', labelsize=25)
ax1.set_ylabel(r'$z$ [nm]', fontsize=30)
ax1.set_xlim([0,Lx/2])
ax1.set_ylim([0,Lz/2])
cb1 = plt.colorbar(dmap1,ax=ax1)
cb1.set_label(r'$\rho_w$ [kg/m$^3$]',fontsize=25, labelpad=20)
cb1.ax.tick_params(labelsize=20)

dmap2 = ax2.pcolormesh(X_fold, Z_fold, 1.66054*density_array_but_avg, cmap=cm.bwr)
ax2.plot(branch_but[0,:],branch_but[1,:],'k-',linewidth=3)
ax2.plot(xb,zb,'g:',linewidth=3)
ax2.tick_params(axis='both', labelsize=25)
ax2.set_ylabel(r'$z$ [nm]', fontsize=30)
ax2.set_xlabel(r'$x$ [nm]', fontsize=30)
ax2.set_xlim([0,Lx/2])
ax2.set_ylim([0,Lz/2])
cb2 = plt.colorbar(dmap2,ax=ax2)
cb2.set_label(r'$\rho_b$ [kg/m$^3$]',fontsize=25, labelpad=20)
cb2.ax.tick_params(labelsize=20)

plt.show()