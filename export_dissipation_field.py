import densmap as dm
import numpy as np

# Viscosity
mu_w = 8.77e-4  # [Pa*s]
mu_g = 8.77e-7  # [Pa*s]
rho_ref = 986   # [Kg/m^3]
mu_fun = lambda C : max( 0.5*mu_w*(C+1.0) - 0.5*mu_g*(C-1.0), 0.0 )
mu_fun_vec = np.vectorize(mu_fun)

FP = dm.fitting_parameters( par_file='parameters_shear.txt' )

folder_name = FP.folder_name
file_root = 'flow_'

Lx = FP.lenght_x
Lz = FP.lenght_z

# CREATING MESHGRID
print("Creating meshgrid")
vel_x, vel_z = dm.read_velocity_file(folder_name+'/'+file_root+'00100.dat')
Nx = vel_x.shape[0]
Nz = vel_x.shape[1]
hx = Lx/Nx
hz = Lz/Nz
x = hx*np.arange(0.0,Nx,1.0, dtype=float)
z = hz*np.arange(0.0,Nz,1.0, dtype=float)
X, Z = np.meshgrid(x, z, sparse=False, indexing='ij')

# Manually tune, crop window [nm]
x0_crop = 87.00
x1_crop = 100.0
z0_crop = 0.000
z1_crop = 6.000
idx_x0 = np.argmin(np.abs(x-x0_crop))
idx_x1 = np.argmin(np.abs(x-x1_crop))
idx_z0 = np.argmin(np.abs(z-z0_crop))
idx_z1 = np.argmin(np.abs(z-z1_crop))
x0_crop = x[idx_x0]
x1_crop = x[idx_x1]
z0_crop = z[idx_z0]
z1_crop = z[idx_z1]
x_crop = x[idx_x0:idx_x1]
z_crop = z[idx_z0:idx_z1]
X_crop, Z_crop = np.meshgrid(x_crop, z_crop, sparse=False, indexing='ij') 

print("Zoom-in window: ["+str(x0_crop)+","+str(x1_crop)+"]x["+str(z0_crop)+","+str(z1_crop)+"], (dX x dZ)")

# Testing .vtk output function
vtk_folder = "/home/michele/densmap/TestVtk"

# INITIALIZING SMOOTHING KERNEL
p = 2.0
r_mol = p*FP.r_mol
smoother = dm.smooth_kernel(r_mol, hx, hz)

# TIME AVERAGING
n_aver = 20

n_init = FP.first_stamp
n_fin = FP.last_stamp
dt = FP.time_step

n_dump = 10
v_x_list = []
v_z_list = []
for idx in range(n_init, n_fin+1):
    if idx%n_dump==0 :
        print("Obtainig frame "+str(idx))
        t_label = str(dt*idx)+' ps'
    # Time-averaging window
    n_hist = min(n_aver, idx-n_init+1)
    w = np.exp(-np.linspace(0.0,5.0,n_hist))
    w = w / np.sum(w)
    rho = dm.read_density_file(folder_name+'/'+file_root+'{:05d}'.format(idx)+'.dat', bin='y')
    rho = rho[idx_x0:idx_x1,idx_z0:idx_z1]
    tmp_x, tmp_z = dm.read_velocity_file(folder_name+'/'+file_root+'{:05d}'.format(idx)+'.dat')
    tmp_x = tmp_x[idx_x0:idx_x1,idx_z0:idx_z1]
    tmp_z = tmp_z[idx_x0:idx_x1,idx_z0:idx_z1]
    if idx-n_init+1 > n_aver :
        v_x_list.append(tmp_x)
        v_x_list.pop(0)
        v_z_list.append(tmp_z)
        v_z_list.pop(0)
    else :
        v_x_list.append(tmp_x)
        v_z_list.append(tmp_z)
    for k in range(n_hist) :
        if k == 0 :
            smooth_v_x = w[0]*v_x_list[-1]
            smooth_v_z = w[0]*v_z_list[-1]
        else :
            smooth_v_x += w[k]*v_x_list[-k-1]
            smooth_v_z += w[k]*v_z_list[-k-1]
    smooth_v_x = dm.convolute(smooth_v_x, smoother)
    smooth_v_z = dm.convolute(smooth_v_z, smoother)
    dvx_dx = ( np.roll(smooth_v_x, 1, axis=0) + np.roll(smooth_v_x, -1, axis=0) ) / (2.0*hx)
    dvx_dz = ( np.roll(smooth_v_x, 1, axis=1) + np.roll(smooth_v_x, -1, axis=1) ) / (2.0*hz)
    dvz_dx = ( np.roll(smooth_v_z, 1, axis=0) + np.roll(smooth_v_z, -1, axis=0) ) / (2.0*hx)
    dvz_dz = ( np.roll(smooth_v_z, 1, axis=1) + np.roll(smooth_v_z, -1, axis=1) ) / (2.0*hz)
    gradient2 = np.multiply( dvx_dx, dvx_dx ) + np.multiply( dvx_dz, dvx_dz ) + np.multiply( dvz_dx, dvz_dx ) + np.multiply( dvz_dz, dvz_dz )
    viscosity = mu_fun_vec( rho/rho_ref )
    dissipation = np.multiply( viscosity, gradient2 )
    
    # Testing .vtk output function
    if n_hist==n_aver :
        # print(str(idx).zfill(5))
        dm.export_scalar_vtk(x_crop, z_crop, hx, hz, 2.5, dissipation, file_name=vtk_folder+"/dissipation_"+str(idx).zfill(5)+".vtk")
