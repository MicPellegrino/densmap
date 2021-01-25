import densmap as dm
import numpy as np

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

vtk_folder = "/home/michele/densmap/BreakageVtk"

# INITIALIZING SMOOTHING KERNEL
p = 2.0
r_mol = p*FP.r_mol
smoother = dm.smooth_kernel(r_mol, hx, hz)

# TIME AVERAGING
n_aver = 20
n_agg = 8

n_init = FP.first_stamp
n_fin = FP.last_stamp
dt = FP.time_step

n_dump = 10
na = 0
p_x_list = []
p_z_list = []
smooth_p_x_exp = np.zeros( vel_x.shape )
smooth_p_z_exp = np.zeros( vel_z.shape )
for idx in range(n_init, n_fin+1):
    na += 1
    if idx%n_dump==0 :
        print("Obtainig frame "+str(idx))
        t_label = str(dt*idx)+' ps'
    # Time-averaging window
    n_hist = min(n_aver, idx-n_init+1)
    w = np.exp(-np.linspace(0.0,5.0,n_hist))
    w = w / np.sum(w)
    rho = dm.read_density_file(folder_name+'/'+file_root+'{:05d}'.format(idx)+'.dat', bin='y')
    tmp_x, tmp_z = dm.read_velocity_file(folder_name+'/'+file_root+'{:05d}'.format(idx)+'.dat')
    tmp_x = np.multiply(rho, tmp_x)
    tmp_z = np.multiply(rho, tmp_z)
    if idx-n_init+1 > n_aver :
        p_x_list.append(tmp_x)
        p_x_list.pop(0)
        p_z_list.append(tmp_z)
        p_z_list.pop(0)
    else :
        p_x_list.append(tmp_x)
        p_z_list.append(tmp_z)
    for k in range(n_hist) :
        if k == 0 :
            smooth_p_x = w[0]*p_x_list[-1]
            smooth_p_z = w[0]*p_z_list[-1]
        else :
            smooth_p_x += w[k]*p_x_list[-k-1]
            smooth_p_z += w[k]*p_z_list[-k-1]
    smooth_p_x = dm.convolute(smooth_p_x, smoother)
    smooth_p_z = dm.convolute(smooth_p_z, smoother)
    smooth_p_x_exp += smooth_p_x
    smooth_p_z_exp += smooth_p_z
    if na == n_agg :
        na = 0
        smooth_p_x_exp /= n_agg
        smooth_p_z_exp /= n_agg
        if n_hist==n_aver :
            dm.export_vector_vtk(x, z, hx, hz, 2.5, smooth_p_x_exp, smooth_p_z_exp, \
                file_name=vtk_folder+"/momentum_"+str(idx//n_agg).zfill(5)+".vtk")
