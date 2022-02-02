import numpy as np
import densmap as dm
import matplotlib.pyplot as plt
from matplotlib import cm
import os

# User inputs
folder_name = 'Q1Ca030_double'
nf = 1236
ni = max(0,nf-400)
Lx = 159.75000
Lz = 30.63400
nbins = 4
ref_order = 11

# Generate meshgrid
file_root = 'flow_'
init_array = dm.read_density_file(folder_name+'/'+file_root+'00001.dat', bin='y')
Nx = init_array.shape[0]
Nz = init_array.shape[1]
hx = Lx/Nx
hz = Lz/Nz
x = hx*(np.arange(0.0,Nx,1.0, dtype=float)+0.5)
z = hz*(np.arange(0.0,Nz,1.0, dtype=float)+0.5)
X, Z = np.meshgrid(x, z, sparse=False, indexing='ij')

def detect_interface(density_array, offset, nband) :
    M = int(np.ceil(offset*Nx))
    left_branch = np.zeros( Nz, dtype=float )
    right_branch = np.zeros( Nz, dtype=float )
    mass_strip = np.zeros( Nz, dtype=float )
    xcom_strip = np.zeros( Nz, dtype=float )
    for j in range(Nz) : 
        mass_strip[j] = np.sum(density_array[:,j])
        if mass_strip[j] > 0 :
            xcom_strip[j] = np.sum(density_array[:,j]*x)/mass_strip[j]
            icom = int(xcom_strip[j]/hx)
            density_target = 0.5*np.mean(density_array[icom-nband:icom+nband,j])
            for i in range(0,M) :
                if density_array[i,j] >= density_target :
                    left_branch[j] = \
                        ((density_array[i,j]-density_target)*x[i-1]+(density_target-density_array[i-1,j])*x[i]) \
                        /(density_array[i,j]-density_array[i-1,j])
                    break
        
            for i in range(Nx-2, Nx-M, -1) :
                if density_array[i,j] >= density_target :
                    right_branch[j] = \
                        ((density_array[i,j]-density_target)*x[i+1]+(density_target-density_array[i+1,j])*x[i]) \
                        / (density_array[i,j]-density_array[i+1,j])
                    break
    xcom = np.sum(xcom_strip*mass_strip)/np.sum(mass_strip)
    return left_branch, right_branch, xcom

def average_com_detrend(n_init, n_fin, off=0.5, nb=5) :
    init_array = dm.read_density_file(folder_name+'/'+file_root+str(n_init).zfill(5)+'.dat', bin='y')
    left_branch_avg, right_branch_avg, xcom_0 = detect_interface(init_array, off, nb)
    for n in range(n_init+1, n_fin+1) :
        print("[ interface frame "+str(n)+" ]")
        n_array = dm.read_density_file(folder_name+'/'+file_root+str(n).zfill(5)+'.dat', bin='y')
        left_branch, right_branch, xcom = detect_interface(n_array, off, nb)
        left_branch_avg = left_branch_avg + left_branch - (xcom-xcom_0)
        right_branch_avg = right_branch_avg + right_branch - (xcom-xcom_0)
    return left_branch_avg/(n_fin-n_init+1), right_branch_avg/(n_fin-n_init+1)

def array_from_file( filename ):
    my_list = []
    with open(filename, 'r') as f:
        for line in f:
            my_list.append(float(line.split()[0]))
    return np.array(my_list)

def interface_polyfit(li, ri, zf, polyord) :
    pl = np.polyfit(zf, li, polyord)
    pr = np.polyfit(zf, ri, polyord)
    dpl = np.polyder(pl)
    dpr = np.polyder(pr)
    return pl, pr, dpl, dpr

def numerical_derivative(li, ri, hz) :
    dli = np.zeros(li.shape, dtype=float)
    dri = np.zeros(ri.shape, dtype=float)
    dli[1:-1] = 0.5*(li[2:]-li[:-2])/hz
    dri[1:-1] = 0.5*(ri[2:]-ri[:-2])/hz
    dli[0] = 0.5*(li[2]-4.0*li[1]+3.0*li[0])/hz
    dli[-1] = 0.5*(4.0*li[-2]-li[-3]-3.0*li[-1])/hz
    dri[0] = 0.5*(ri[2]-4.0*ri[1]+3.0*ri[0])/hz
    dri[-1] = 0.5*(4.0*ri[-2]-ri[-3]-3.0*ri[-1])/hz
    return dli, dri

save_dir = 'InterfaceData'
s1 = os.path.isfile(save_dir+'/'+folder_name+'_lb.txt')
s2 = os.path.isfile(save_dir+'/'+folder_name+'_rb.txt')
s3 = os.path.isfile(save_dir+'/'+folder_name+'_z.txt')
if s1 and s2 and s3 :
    lb = array_from_file(save_dir+'/'+folder_name+'_lb.txt')
    rb = array_from_file(save_dir+'/'+folder_name+'_rb.txt')
    zint = array_from_file(save_dir+'/'+folder_name+'_z.txt')
else:
    lb, rb = average_com_detrend(ni, nf)
    lb = lb[nbins:-nbins]
    rb = rb[nbins:-nbins]
    zint = z[nbins:-nbins]
    np.savetxt(save_dir+'/'+folder_name+'_z.txt', zint)
    np.savetxt(save_dir+'/'+folder_name+'_lb.txt', lb)
    np.savetxt(save_dir+'/'+folder_name+'_rb.txt', rb)

theta_rec_min = 180
theta_rec_max = 0
theta_adv_min = 180
theta_adv_max = 0
for d in range(-2,3):
    pl, pr, dpl, dpr = interface_polyfit(lb, rb, zint, ref_order+d)
    # dlb, drb = numerical_derivative(lb, rb, hz)
    atan_l = np.arctan(np.polyval(dpl, zint))
    atan_r = np.arctan(np.polyval(dpr, zint))
    theta_rec = 180.0-0.5*(90+np.rad2deg(atan_l[6-nbins])+90+np.rad2deg(atan_r[-6+nbins]))
    theta_adv = 0.5*(90+np.rad2deg(atan_l[-6+nbins])+90+np.rad2deg(atan_r[6-nbins]))
    if theta_rec < theta_rec_min :
        theta_rec_min = theta_rec
    if theta_adv < theta_adv_min :
        theta_adv_min = theta_adv
    if theta_rec > theta_rec_max :
        theta_rec_max = theta_rec
    if theta_adv > theta_adv_max :
        theta_adv_max = theta_adv

print("theta_adv = "+str([theta_adv_min,theta_adv_max])+" -> err_adv = "+"{:.4f}".format(0.5*(theta_adv_max-theta_adv_min)))
print("theta_rec = "+str([theta_rec_min,theta_rec_max])+" -> err_rec = "+"{:.4f}".format(0.5*(theta_rec_max-theta_rec_min)))

"""
plt.plot(lb, zint, 'r:')
plt.plot(rb, zint, 'r:')
plt.plot(np.polyval(pl, zint), zint, 'b-')
plt.plot(np.polyval(pr, zint), zint, 'b-')
plt.show()

plt.plot(90+np.rad2deg(np.arctan(dlb)), zint, 'r:')
plt.plot(90+np.rad2deg(np.arctan(drb)), zint, 'r:')
plt.plot(90+np.rad2deg(np.arctan(np.polyval(dpl, zint))), zint, 'b-')
plt.plot(90+np.rad2deg(np.arctan(np.polyval(dpr, zint))), zint, 'b-')
plt.show()
"""
