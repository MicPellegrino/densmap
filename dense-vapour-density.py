import densmap as dm
import numpy as np
import matplotlib.pyplot as plt

CONV_KG_DALTON = 1.66053904

def compress(A, block_size=2) :
    nx = A.shape[0]
    rx = A.shape[0]%block_size
    nz = A.shape[1]
    rz = A.shape[1]%block_size
    B = A[0:nx-rx,0:nz-rz].reshape(nx//block_size, block_size, nz//block_size, block_size).mean(axis=(1,-1))
    return B

FP = dm.fitting_parameters( par_file='parameters_shear.txt' )
block_size = 10

folder_name = FP.folder_name
file_root = 'flow_'

n_init = FP.first_stamp
n_fin = FP.last_stamp
Nz = 10

rho = dm.read_density_file(folder_name+'/'+file_root+'{:05d}'.format(n_init)+'.dat', bin='y')
rho = rho[:,Nz:-Nz]

n_dump = 50
for idx in range(n_init+1, n_fin+1):
    
    if idx%n_dump==0 :
        print("Obtainig frame "+str(idx))
    
    rho_temp = dm.read_density_file(folder_name+'/'+file_root+'{:05d}'.format(idx)+'.dat', bin='y')
    rho += rho_temp[:,Nz:-Nz]

rho /= (n_fin-n_init+1)
rho_smooth = CONV_KG_DALTON*compress(rho, block_size)

rho_vals = rho_smooth.flatten()
# n_hist_bins = int(np.ceil(np.sqrt(len(rho_vals))))
n_hist_bins = 1000
hist, bin_edges = np.histogram(rho_vals,bins=n_hist_bins)

plt.step(bin_edges[1:]-0.5*(bin_edges[1]-bin_edges[0]),hist)
plt.show()