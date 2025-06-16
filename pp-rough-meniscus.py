import glob
import os 
import densmap as dm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

a_range = [0.2,0.4,0.6,0.8,1.0]
n_range = [6,8,10,12,14,16]
ca_range = [0.050,0.075,0.100,0.125,0.150,0.175,0.200,0.225,0.250]

workdir = 'WorkdirRoughMeniscus/'
file_root = 'flow_'

# Global variables
Lx = 82.80000
Lz = 28.00000
z_min = 5
z_max = 15
z_ref = 10
dt = 12.5
delta_th = 2.0
n_dump = 50


###########################################################################################################
def postprocess_run(dat_file_list, folder_name, file_tag) :

    n_init = int(dat_file_list[0][5:10].lstrip("0"))
    n_fin = int(dat_file_list[-1][5:10].lstrip("0"))

    # Creating meshgrid
    density_array = dm.read_density_file(folder_name+'/'+file_root+'{:05d}'.format(n_init)+'.dat', bin='y')
    Nx = density_array.shape[0]
    Nz = density_array.shape[1]
    hx = Lx/Nx
    hz = Lz/Nz
    x = hx*np.arange(0.0,Nx,1.0, dtype=float)+0.5*hx
    z = hz*np.arange(0.0,Nz,1.0, dtype=float)+0.5*hz
    X, Z = np.meshgrid(x, z, sparse=False, indexing='ij')

    # Initialization
    xc = np.sum(np.multiply(density_array[:,int(z_ref//hz)+1],x))/np.sum(density_array[:,int(z_ref//hz)+1])
    dxc = 0.5*Lx-xc
    density_array = np.roll(density_array, shift=int(dxc//hx), axis=0)
    left_int, right_int = dm.detect_interface_loc(density_array, hx, hz, z_min, z_max, wall='l')
    t = []
    xcom = []
    lcl = []
    rcl = []

    # Processing frame-by-frame
    # Skippign the last frame, as it may be corrupted
    for idx in range(n_init, n_fin):
        if idx%n_dump==0 :
            print("Obtainig frame "+str(idx))
        density_array = dm.read_density_file(folder_name+'/'+file_root+'{:05d}'.format(idx)+'.dat', bin='y')
        density_array = np.roll(density_array, shift=int(dxc//hx), axis=0)
        xc = np.sum(np.multiply(density_array[:,int(z_ref//hz)+1],x))/np.sum(density_array[:,int(z_ref//hz)+1])
        dxc = 0.5*Lx-xc
        left_int, right_int = dm.detect_interface_loc(density_array, hx, hz, z_min, z_max, wall='l')
        diff_cl = right_int[0][0]-left_int[0][0]
        if diff_cl<0 or diff_cl> Lx:
            # If this is true, then most probably a film is being deposited
            break
        if idx%n_dump==0 :
            print(left_int[0][0],'|',xc,'|',right_int[0][0])
        t.append((1e-3)*idx*dt)
        xcom.append(xc)
        lcl.append(left_int[0][0])
        rcl.append(right_int[0][0])

    lcl = np.array(lcl)
    rcl = np.array(rcl)
    dcl = rcl-lcl
    t = np.array(t)

    fig, ax = plt.subplots()
    plt.title(file_tag)
    plt.plot(t,dcl,'k-')
    plt.xlabel(r"$t$ [ns]")
    plt.ylabel(r"$\Delta_{ls}x_{cl}$ [nm]")
    plt.savefig(workdir+file_tag+'.png',format='png')

    np.savez(workdir+file_tag+'.npz', t, dcl)


###########################################################################################################
for a in a_range :
    for n in n_range :
        for ca in ca_range :

            # Input and output files
            a_tag = str(a).replace('.','').ljust(2,'0')
            n_tag = str(n).rjust(2,'0')
            c_tag = str(10*ca).replace('.','').ljust(3,'0')
            file_tag = 'N'+n_tag+'A'+a_tag+'C'+c_tag
            folder_name = workdir+file_tag+'/'

            # Automagically finding the index of the first and last file
            print("#####",file_tag,"    #####")
            glob_fn = glob.glob(folder_name+'*.dat')
            dat_file_list = sorted([os.path.basename(x) for x in glob_fn])
            minsize = min([os.stat(x).st_size for x in glob_fn],default=None)
            if len(dat_file_list) == 0 :
                print("! Empty directory !")
            elif minsize == 0 :
                print("! Corrupted output !")
            else :
                print("##### POSTPROCESSING #####")
                try:
                    postprocess_run(dat_file_list, folder_name, file_tag)
                except ValueError as e:
                    print ('! Corrupted output !')