import numpy as np
import matplotlib.pyplot as plt
import densmap as dm
import os
from matplotlib import cm

mark_size = 25
mark_width = 5
line_width = 4

axis_font = 45
legend_font = 40
ticks_font = 35

CONV_KG_DALTON = 1.66053904
COLOR_MAP_H = cm.Blues

mark_size = 25
mark_width = 5
line_width = 4

axis_font = 45
legend_font = 40
ticks_font = 35

Lx = 59.85000
Lz = 25.00000

folder_root = "HexaneCA"

n_average = 200
s = np.linspace(0,2*np.pi,250)

contact_angle = []

###############
### BUTANOL ###
###############
folder_name = folder_root
print(folder_name)
arr = os.listdir(folder_name)
arr_tot = sorted([item for item in arr if len(item)==len("flow_00000.dat")])
arr_tot = arr_tot[-n_average:]
rho = CONV_KG_DALTON*dm.read_density_file(folder_name+'/'+arr_tot[0], bin='y')
dx = Lx/rho.shape[0]
x = np.linspace(0.5*dx,Lx-0.5*dx,rho.shape[0])
dz = Lz/rho.shape[1]
z = np.linspace(0.5*dz,Lz-0.5*dz,rho.shape[1])
X, Z = np.meshgrid(x, z, sparse=False, indexing='ij')
xcom = np.sum(x*np.sum(rho,axis=1))/np.sum(rho)
offset = int((xcom-0.5*Lx)/dx)
rho = np.roll(rho,-offset,axis=0)
for j in range(1,n_average) :
    file_name = folder_name+'/'+arr_tot[j]
    if (j-1) % 100 == 0 :
        print("[reading "+file_name+"]")
    rho_temp = CONV_KG_DALTON*dm.read_density_file(folder_name+'/'+arr_tot[j], bin='y')
    xcom = np.sum(x*np.sum(rho_temp,axis=1))/np.sum(rho_temp)
    offset = int((xcom-0.5*Lx)/dx)
    rho_temp = np.roll(rho_temp,-offset,axis=0)
    rho += rho_temp
print("[computing average]")
rho /= n_average
bulk_density = dm.detect_bulk_density(rho,density_th=10.0)
intf_contour = dm.detect_contour(rho,0.5*bulk_density,dx,dz)
xc, zc, R, residue = dm.circle_fit_droplet(intf_contour,z_th=2.0)
circle_x = xc + R*np.cos(s)
circle_z = zc + R*np.sin(s)
h = 1.0
cot_circle = (h-zc)/np.sqrt(R*R-(h-zc)**2)
theta_circle = np.rad2deg( -np.arctan( cot_circle )+0.5*np.pi )
theta_circle = theta_circle + 180*(theta_circle<=0)
print("[contact angle = "+str(theta_circle)+"deg]")
contact_angle.append(theta_circle)
plt.pcolor(X,Z,rho,cmap=COLOR_MAP_H)
plt.plot(intf_contour[0,:],intf_contour[1,:],'k:',linewidth=3.0)
plt.plot(circle_x,circle_z,'k-',linewidth=3.0)
cbar = plt.colorbar()
cbar.set_label(r'$\rho$ [kg/m$^3$]', rotation=270, fontsize=0.75*axis_font, labelpad=30)
cbar.ax.tick_params(labelsize=0.75*ticks_font)
plt.axis('scaled')
plt.xlim([0,Lx])
plt.ylim([0,Lz])
plt.xlabel(r'$x$', fontsize=axis_font)
plt.ylabel(r'$z$', fontsize=axis_font)
plt.tick_params(labelsize=ticks_font)
plt.show()