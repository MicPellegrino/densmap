import densmap as dm
import numpy as np

CONV_KG_DALTON = 1.66053904
mode = 'com'
"""
4   3
 com
1   2
"""

def compress(A, block_size=2) :
    nx = A.shape[0]
    rx = A.shape[0]%block_size
    nz = A.shape[1]
    rz = A.shape[1]%block_size
    B = A[0:nx-rx,0:nz-rz].reshape(nx//block_size, block_size, nz//block_size, block_size).mean(axis=(1,-1))
    return B

FP = dm.fitting_parameters( par_file='parameters_shear_hex.txt' )

# Compression
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
vel_x, vel_z = dm.read_velocity_file(folder_name+'/'+file_root+str(n_init).zfill(5)+'.dat')
Nx = vel_x.shape[0]
Nz = vel_x.shape[1]
hx = Lx/Nx
hz = Lz/Nz
x = hx*np.arange(0.0,Nx,1.0, dtype=float)+0.5*hx
z = hz*np.arange(0.0,Nz,1.0, dtype=float)+0.5*hz

hxc = block_size*Lx/Nx
hzc = block_size*Lz/Nz
xc = hxc*np.arange(0.0,Nx//block_size,1.0, dtype=float)+0.5*hxc
zc = hzc*np.arange(0.0,Nz//block_size,1.0, dtype=float)+0.5*hzc

vtk_folder = "/home/michele/densmap/VtkHexCa005q60/"

cl_folder = 'ShearWatHex/C005Q60/'
cl_tl = np.loadtxt(cl_folder+'position_upper.txt')-0.5*np.loadtxt(cl_folder+'radius_upper.txt')
cl_bl = np.loadtxt(cl_folder+'position_lower.txt')-0.5*np.loadtxt(cl_folder+'radius_lower.txt')
cl_tr = np.loadtxt(cl_folder+'position_upper.txt')+0.5*np.loadtxt(cl_folder+'radius_upper.txt')
cl_br = np.loadtxt(cl_folder+'position_lower.txt')+0.5*np.loadtxt(cl_folder+'radius_lower.txt')

cl_tl = np.array( (cl_tl[n_init:n_fin]-cl_tl[n_init])/hxc, dtype=int )
cl_bl = np.array( (cl_bl[n_init:n_fin]-cl_bl[n_init])/hxc, dtype=int )
cl_tr = np.array( (cl_tr[n_init:n_fin]-cl_tr[n_init])/hxc, dtype=int )
cl_br = np.array( (cl_br[n_init:n_fin]-cl_br[n_init])/hxc, dtype=int )

# INITIALIZING SMOOTHING KERNEL
p = 2.0
r_mol = p*FP.r_mol
smoother = dm.smooth_kernel(r_mol, hx, hz)

### NB! Not doing a rolling average in time! ###

n_dump = 10
smooth_p_x_exp = np.zeros( vel_x.shape )
smooth_p_z_exp = np.zeros( vel_z.shape )
smooth_v_x_exp = np.zeros( vel_x.shape )
smooth_v_z_exp = np.zeros( vel_z.shape )
rho_tot = np.zeros( vel_z.shape )

# SHIFT FOR THE COM OF WATER ALONE! LESS NOISY
ihalf = (Nx//2)+(Nx%2)

for idx in range(n_init, n_fin+1):
    
    if idx%n_dump==0 :
        print("Obtainig frame "+str(idx))
        t_label = str(dt*idx)+' ps'
    # Time-averaging window
    rho = dm.read_density_file(folder_name+'/'+file_root+'{:05d}'.format(idx)+'.dat', bin='y')
    rho_sol = dm.read_density_file(folder_name+'/'+file_root+'SOL_'+'{:05d}'.format(idx)+'.dat', bin='y')
    
    rho_x = np.mean(rho_sol,axis=1)
    xcom = np.sum(rho_x*x)/np.sum(rho_x)
    icom = int(np.round(xcom/hx))
    ishift = ihalf-icom

    if mode == 'com' :
        rho = np.roll(rho, ishift, axis=0)
    elif mode == 'cl1' :
        rho = np.roll(rho, cl_bl, axis=0)
    elif mode == 'cl2' :
        rho = np.roll(rho, cl_br, axis=0)
    elif mode == 'cl3' :
        rho = np.roll(rho, cl_tr, axis=0)
    elif mode == 'cl4' :
        rho = np.roll(rho, cl_tl, axis=0)
    
    rho_tot += rho

    tmp_x, tmp_z = dm.read_velocity_file(folder_name+'/'+file_root+'{:05d}'.format(idx)+'.dat')
    tmp_x = np.roll(tmp_x, ishift, axis=0)
    tmp_z = np.roll(tmp_z, ishift, axis=0)
    
    smooth_v_x_exp += tmp_x
    smooth_v_x_exp += tmp_z
    tmp_x = np.multiply(rho, tmp_x)
    tmp_z = np.multiply(rho, tmp_z)
    smooth_p_x_exp += tmp_x
    smooth_p_z_exp += tmp_z

smooth_v_x_exp /= (n_fin+1-n_init)
smooth_v_z_exp /= (n_fin+1-n_init)
smooth_p_x_exp /= (n_fin+1-n_init)
smooth_p_z_exp /= (n_fin+1-n_init)

rho_tot /= (n_fin+1-n_init)
rho_tot *= 1.66054

if mode == 'com' :
    smooth_p_z_exp = 0.5*(smooth_p_z_exp-np.flipud(np.fliplr(smooth_p_z_exp)))
    smooth_p_x_exp = 0.5*(smooth_p_x_exp-np.flipud(np.fliplr(smooth_p_x_exp)))

smooth_p_x_exp = compress(smooth_p_x_exp, block_size)
smooth_p_z_exp = compress(smooth_p_z_exp, block_size)
smooth_v_x_exp = compress(smooth_v_x_exp, block_size)
smooth_v_z_exp = compress(smooth_v_z_exp, block_size)

Lsep = 18.35
xpf = xc/Lsep-0.5*Lx/Lsep
xpf_range = max(xpf)-min(xpf)
hxpf = xpf_range/len(xpf)
ypf = zc/Lsep-0.5*Lz/Lsep
ypf_range = max(ypf)-min(ypf)
hypf = ypf_range/len(ypf)

dm.export_scalarxy_vtk(xpf, ypf, hxpf, hypf, 0.1, rho_tot, 
    file_name=vtk_folder+"/average_density.vtk")
dm.export_vectorxy_vtk(xpf, ypf, hxpf, hypf, 0.1, smooth_p_x_exp, smooth_p_z_exp,
    file_name=vtk_folder+"/average_momentum.vtk")
dm.export_vectorxy_vtk(xpf, ypf, hxpf, hypf, 0.1, smooth_v_x_exp, smooth_v_z_exp,
    file_name=vtk_folder+"/average_velocity.vtk")