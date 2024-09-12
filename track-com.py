import numpy as np
import densmap as dm
import matplotlib.pyplot as plt

# FP = dm.fitting_parameters( par_file='parameters_shear_large.txt' )
FP = dm.fitting_parameters( par_file='parameters_density.txt' )

folder_name = FP.folder_name
file_root = 'flow_SOL_'

Lx = FP.lenght_x
Lz = FP.lenght_z

n_init = FP.first_stamp
n_fin = FP.last_stamp
dt = FP.time_step

print("Creating meshgrid")
rho_ref = dm.read_density_file(folder_name+'/'+file_root+'{:05d}'.format(n_init)+'.dat', bin='y')
Nx = rho_ref.shape[0]
Nz = rho_ref.shape[1]
hx = Lx/Nx
hz = Lz/Nz
x = hx*np.arange(0.0,Nx,1.0, dtype=float)+0.5*hx
z = hz*np.arange(0.0,Nz,1.0, dtype=float)+0.5*hz

xcom_vec = []
t_vec = []

n_dump=10
for idx in range(n_init, n_fin+1):
    t_vec.append(idx*dt)
    if idx%n_dump==0 :
        print("Obtainig frame "+str(idx))
        t_label = str(dt*idx)+' ps'
    rho_ref = dm.read_density_file(folder_name+'/'+file_root+'{:05d}'.format(idx)+'.dat', bin='y')
    xc = np.sum(np.sum(rho_ref,axis=1)*x)/np.sum(rho_ref)
    xcom_vec.append(xc)

t_vec_ns = (1e-3)*np.array(t_vec)

p = np.polyfit(t_vec_ns,xcom_vec,deg=1)
x_lin = np.polyval(p,t_vec_ns)

dt_md = 2e-6
print("COM speed = "+str(p[0])+"nm/ns")
print("dlambda = "+str(p[0]*dt_md))

plt.plot(t_vec_ns,xcom_vec,'k-', linewidth=2)
plt.plot(t_vec_ns,x_lin,'r--',linewidth=3)
plt.xlabel(r'$t$ [ns]',fontsize=25)
plt.ylabel(r'$x_{com}$ [nm]',fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()