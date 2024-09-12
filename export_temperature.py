import densmap as dm
import numpy as np

# FP = dm.fitting_parameters( par_file='parameters_shear.txt' )
FP = dm.fitting_parameters( par_file='parameters_small.txt' )

# Compression
# block_size = 2
block_size = 1

folder_name = FP.folder_name
file_root = 'flow_'

Lx = FP.lenght_x
Lz = FP.lenght_z

n_init = FP.first_stamp
n_fin = FP.last_stamp
dt = FP.time_step

# CREATING MESHGRID
print("Creating meshgrid")
temp_ref = dm.read_temperature_file(folder_name+'/'+file_root+'{:05d}'.format(n_init)+'.dat')
Nx = temp_ref.shape[0]
Nz = temp_ref.shape[1]
hx = Lx/Nx
hz = Lz/Nz
x = hx*np.arange(0.0,Nx,1.0, dtype=float)+0.5*hx
z = hz*np.arange(0.0,Nz,1.0, dtype=float)+0.5*hz

hxc = block_size*Lx/Nx
hzc = block_size*Lz/Nz
xc = hxc*np.arange(0.0,Nx//block_size,1.0, dtype=float)+0.5*hxc
zc = hzc*np.arange(0.0,Nz//block_size,1.0, dtype=float)+0.5*hzc

vtk_folder = "/home/michele/densmap/TestVtk"

# INITIALIZING SMOOTHING KERNEL
r_mol = FP.r_mol
smoother = dm.smooth_kernel(r_mol, hx, hz)

n_aver = 1

n_dump = 3*n_aver
rho_smooth_tot = np.zeros(temp_ref.shape)
rho_smooth_sol = np.zeros(temp_ref.shape)
rho_smooth_but = np.zeros(temp_ref.shape)

tag_a = 'SOL'
tag_b = 'BUT'

for idx in range(n_init, n_fin+1):

    if idx%n_dump==0 :
        print("Obtainig frame "+str(idx))
        t_label = str(dt*idx)+' ps'

    # Time-averaging window
    rho_smooth_tot += dm.read_temperature_file(folder_name+'/'+file_root+'{:05d}'.format(idx)+'.dat')
    rho_smooth_sol += dm.read_temperature_file(folder_name+'/'+file_root+tag_a+'_'+'{:05d}'.format(idx)+'.dat')
    rho_smooth_but += dm.read_temperature_file(folder_name+'/'+file_root+tag_b+'_'+'{:05d}'.format(idx)+'.dat')

    dm.export_scalar_vtk(xc, zc, hxc, hzc, 2.5, rho_smooth_tot, file_name=vtk_folder+"/density_tot_"+str((idx-n_init)//n_aver).zfill(5)+".vtk")
    dm.export_scalar_vtk(xc, zc, hxc, hzc, 2.5, rho_smooth_sol, file_name=vtk_folder+"/density_sol_"+str((idx-n_init)//n_aver).zfill(5)+".vtk")
    dm.export_scalar_vtk(xc, zc, hxc, hzc, 2.5, rho_smooth_but, file_name=vtk_folder+"/density_but_"+str((idx-n_init)//n_aver).zfill(5)+".vtk")

    rho_smooth_tot = np.zeros(temp_ref.shape)
    rho_smooth_sol = np.zeros(temp_ref.shape)
    rho_smooth_but = np.zeros(temp_ref.shape)

ihalf = (Nx//2)+(Nx%2)

for idx in range(n_init, n_fin+1):

    if idx%n_dump==0 :
        print("Obtainig frame "+str(idx))
        t_label = str(dt*idx)+' ps'

    # Time-averaging window
    temp_tot = dm.read_temperature_file(folder_name+'/'+file_root+'{:05d}'.format(idx)+'.dat')
    temp_sol = dm.read_temperature_file(folder_name+'/'+file_root+tag_a+'_'+'{:05d}'.format(idx)+'.dat')
    temp_but = dm.read_temperature_file(folder_name+'/'+file_root+tag_b+'_'+'{:05d}'.format(idx)+'.dat')

    temp_x = np.mean(temp_tot,axis=1)
    xcom = np.sum(temp_x*x)/np.sum(temp_x)
    icom = int(np.round(xcom/hx))
    ishift = ihalf-icom
    temp_tot = np.roll(temp_tot, ishift, axis=0)
    temp_sol = np.roll(temp_sol, ishift, axis=0)
    temp_but = np.roll(temp_but, ishift, axis=0)

    rho_smooth_tot += temp_tot
    rho_smooth_sol += temp_sol
    rho_smooth_but += temp_but

rho_smooth_tot /= (n_fin-n_init+1)
rho_smooth_sol /= (n_fin-n_init+1)
rho_smooth_but /= (n_fin-n_init+1)

dm.export_scalar_vtk(xc, zc, hxc, hzc, 2.5, rho_smooth_tot, file_name=vtk_folder+"/average_temperature_tot.vtk")
dm.export_scalar_vtk(xc, zc, hxc, hzc, 2.5, rho_smooth_sol, file_name=vtk_folder+"/average_temperature_sol.vtk")
dm.export_scalar_vtk(xc, zc, hxc, hzc, 2.5, rho_smooth_but, file_name=vtk_folder+"/average_temperature_but.vtk")