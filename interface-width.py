import densmap as dm
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

def roll_multiple(A,r) :
    nx, nz = A.shape
    for i in range(nz) :
        A[:,i] = np.roll(A[:,i],r[i])
    return A

CONV_KG_DALTON = 1.66053904

folder_name = "InterfaceWatHex"

nini = 500
nfin = 2000

file_root = 'flow_'
tag_a = 'SOL'
tag_b = 'HEX'

hx = 0.1001238679245283
hz = 0.19800879999999998
Lx = 10.61313
Lz = 4.95022
Nbinx = 106
Nbinz = 25
x = np.linspace(0.5*hx,Lx-0.5*hx,Nbinx)
ihalf = Nbinx//2

rho_smooth_tot = np.zeros((Nbinx,Nbinz))
rho_smooth_sol = np.zeros((Nbinx,Nbinz))
rho_smooth_hex = np.zeros((Nbinx,Nbinz))

nwini = 15
rho_cl_right_tot = np.zeros((nwini,Nbinz))
rho_cl_right_sol = np.zeros((nwini,Nbinz))
rho_cl_right_hex = np.zeros((nwini,Nbinz))
rho_cl_left_tot = np.zeros((nwini,Nbinz))
rho_cl_left_sol = np.zeros((nwini,Nbinz))
rho_cl_left_hex = np.zeros((nwini,Nbinz))

for idx in range(nini, nfin+1):

    temp_tot = CONV_KG_DALTON * dm.read_density_file(folder_name+'/'+file_root+'{:05d}'.format(idx)+'.dat', bin='y')
    temp_sol = CONV_KG_DALTON * dm.read_density_file(folder_name+'/'+file_root+tag_a+'_'+'{:05d}'.format(idx)+'.dat', bin='y')
    temp_hex = CONV_KG_DALTON * dm.read_density_file(folder_name+'/'+file_root+tag_b+'_'+'{:05d}'.format(idx)+'.dat', bin='y')

    temp_tot = np.transpose(temp_tot)
    temp_sol = np.transpose(temp_sol)
    temp_hex = np.transpose(temp_hex)

    """
        Shifting so that the COM is always approx. in the middle
    """
    temp_x = np.mean(temp_hex,axis=1)
    xcom = np.sum(temp_x*x)/np.sum(temp_x)
    icom = int(np.round(xcom/hx))
    ishift = ihalf-icom
    temp_tot = np.roll(temp_tot, ishift, axis=0)
    temp_sol = np.roll(temp_sol, ishift, axis=0)
    temp_hex = np.roll(temp_hex, ishift, axis=0)

    """
        Averaging the whole density field -> thermal interface width
    """
    rho_smooth_tot += temp_tot
    rho_smooth_sol += temp_sol
    rho_smooth_hex += temp_hex

    """
        Averaging slice-by-slice -> intrinsic interface width
    """
    left_branch, right_branch = dm.detect_interface_loc(temp_hex,hx,hz,0,Lz)

    ### TEST ###
    # plt.plot(left_branch[0,:],left_branch[1,:])
    # plt.plot(right_branch[0,:],right_branch[1,:])
    # plt.xlim([0,Lx])
    # plt.ylim([0,Lz])
    # plt.show()
    ############
    
    shift_right = -np.array(right_branch[0,:]/hx,dtype=int)+ihalf
    rho_cl_right_tot += roll_multiple(temp_tot, shift_right)[ihalf-nwini//2:ihalf+nwini//2+1,:]
    rho_cl_right_sol += roll_multiple(temp_sol, shift_right)[ihalf-nwini//2:ihalf+nwini//2+1,:]
    rho_cl_right_hex += roll_multiple(temp_hex, shift_right)[ihalf-nwini//2:ihalf+nwini//2+1,:]

    shift_left = -np.array(left_branch[0,:]/hx,dtype=int)+ihalf
    rho_cl_left_tot += roll_multiple(temp_tot, shift_left-shift_right)[ihalf-nwini//2:ihalf+nwini//2+1,:]
    rho_cl_left_sol += roll_multiple(temp_sol, shift_left-shift_right)[ihalf-nwini//2:ihalf+nwini//2+1,:]
    rho_cl_left_hex += roll_multiple(temp_hex, shift_left-shift_right)[ihalf-nwini//2:ihalf+nwini//2+1,:]
    
    ### TEST ###
    # plt.plot(x[:nwini], np.mean(rho_cl_right_hex,axis=1))
    # plt.show()

rho_cl_right_tot /= (nfin-nini+1)
rho_cl_right_sol /= (nfin-nini+1)
rho_cl_right_hex /= (nfin-nini+1)

rho_cl_left_tot /= (nfin-nini+1)
rho_cl_left_sol /= (nfin-nini+1)
rho_cl_left_hex /= (nfin-nini+1)

x_interface = np.linspace(-hx*nwini,hx*nwini,nwini)
x_finer = np.linspace(-hx*nwini,hx*nwini,10*nwini)
profile_interface = np.mean(rho_cl_left_hex,axis=1)
profile_interface_std = np.std(rho_cl_left_hex,axis=1)
tanh_eq = lambda x, x0, rho, eps : 0.5*rho*( 1+np.tanh((x-x0)/(eps*np.sqrt(2))) )
popt, pcov = opt.curve_fit(tanh_eq, x_interface, profile_interface, p0=(0.0,600,0.4))
print(popt)

plt.plot(x_finer, 2*tanh_eq(x_finer,*popt)/popt[1]-1, 'r--', linewidth=3)
plt.plot(x_interface, 2*profile_interface/popt[1]-1, 'ko', ms=15, markerfacecolor="None", markeredgewidth=3)
# plt.errorbar(x_interface, 2*profile_interface/popt[1]-1, yerr=profile_interface_std/popt[1], fmt='ko', ms=15, markerfacecolor="None", markeredgewidth=3)
plt.xlabel(r"$x$ [nm]",fontsize=50)
plt.ylabel(r"$C$ [1]",fontsize=50)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.subplots_adjust(left=0.20, right=0.98, top=0.95, bottom=0.14)
plt.show()

rho_smooth_tot /= (nfin-nini+1)
rho_smooth_sol /= (nfin-nini+1)
rho_smooth_hex /= (nfin-nini+1)

# plt.plot(x, np.mean(rho_smooth_hex,axis=1))
# plt.show()