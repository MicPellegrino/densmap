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
COLOR_MAP_B = cm.Oranges
COLOR_MAP_W = cm.Blues

mark_size = 25
mark_width = 5
line_width = 4

axis_font = 45
legend_font = 40
ticks_font = 35

Lx = 59.85000
Lz = 25.00000

folder_root_butanol = "BUTANOL-water"
folder_root_water = "WATER-butanol"

labels = []
charges = np.array([60,61,62,63,64,65,66,67,68,69,70])
for q in charges :
    labels.append("qm0"+str(q))

n_average = 500

plot_all = False
plot_test_charge = 64
s = np.linspace(0,2*np.pi,250)

contact_angle_butanol = []
contact_angle_water = []

###############
### BUTANOL ###
###############
for k in range(len(labels)) :
    folder_name = folder_root_butanol+'-'+labels[k]
    print(folder_name)
    arr = os.listdir(folder_name)
    arr_tot = sorted([item for item in arr if len(item)==len("flow_00000.dat")])
    arr_tot = arr_tot[-n_average:]
    rho = CONV_KG_DALTON*dm.read_density_file(folder_name+'/'+arr_tot[0], bin='y')
    if k == 0 :
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
    bulk_density = dm.detect_bulk_density(rho,density_th=2.0)
    intf_contour = dm.detect_contour(rho,0.5*bulk_density,dx,dz)
    xc, zc, R, residue = dm.circle_fit_droplet(intf_contour,z_th=2.0)
    circle_x = xc + R*np.cos(s)
    circle_z = zc + R*np.sin(s)
    h = 1.0
    cot_circle = (h-zc)/np.sqrt(R*R-(h-zc)**2)
    theta_circle = np.rad2deg( -np.arctan( cot_circle )+0.5*np.pi )
    theta_circle = theta_circle + 180*(theta_circle<=0)
    print("[contact angle = "+str(theta_circle)+"deg]")
    contact_angle_butanol.append(theta_circle)
    if plot_all or k==np.where(charges==plot_test_charge)[0][0] :
        plt.pcolor(X,Z,rho,cmap=COLOR_MAP_B)
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
    else :
        print("[no plot]")

#############
### WATER ###
#############
for k in range(len(labels)) :
    folder_name = folder_root_water+'-'+labels[k]
    print(folder_name)
    arr = os.listdir(folder_name)
    arr_tot = [item for item in arr if len(item)==len("flow_00000.dat")]
    arr_tot = arr_tot[-n_average:]
    rho = CONV_KG_DALTON*dm.read_density_file(folder_name+'/'+arr_tot[0], bin='y')
    if k == 0 :
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
    bulk_density = dm.detect_bulk_density(rho,density_th=2.0)
    intf_contour = dm.detect_contour(rho,0.5*bulk_density,dx,dz)
    xc, zc, R, residue = dm.circle_fit_droplet(intf_contour,z_th=2.0)
    circle_x = xc + R*np.cos(s)
    circle_z = zc + R*np.sin(s)
    h = 1.0
    cot_circle = (h-zc)/np.sqrt(R*R-(h-zc)**2)
    theta_circle = np.rad2deg( -np.arctan( cot_circle )+0.5*np.pi )
    theta_circle = theta_circle + 180*(theta_circle<=0)
    print("[contact angle = "+str(theta_circle)+"deg]")
    contact_angle_water.append(theta_circle)
    if plot_all or k==np.where(charges==plot_test_charge)[0][0]:
        plt.pcolor(X,Z,rho,cmap=COLOR_MAP_W)
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
    else :
        print("[no plot]")


######################
### CONTACT ANGLES ###
######################
sigma_w = 511.521
sigma_b = 148.680
sigma_wb = 32.411
sigma_w_st = sigma_w/sigma_wb
sigma_b_st = sigma_b/sigma_wb
contact_angle_water = np.array(contact_angle_water)
contact_angle_butanol = np.array(contact_angle_butanol)
cos =  lambda t : np.cos(np.deg2rad(t))
cos_w = cos(contact_angle_water)
cos_b = cos(contact_angle_butanol)
cos_wb = sigma_w_st*cos_w-sigma_b_st*cos_b
cos_wb = cos_wb*(cos_wb>-1)*(cos_wb<1) + (cos_wb>1) - (cos_wb<-1)

fig, ax = plt.subplots()
ax.plot(0.01*charges, cos_w, 'bs', markersize=mark_size, label="water-vapour")
ax.plot(0.01*charges, cos_b, 'rD', markersize=mark_size, label="butanol-vapour")
ax.plot(0.01*charges, cos_wb, 'ko', markersize=mark_size, label="water-butanol (est.)")
ax.plot([0.01*charges[0],0.01*charges[-1]], [1,1], 'k--', linewidth=line_width)
ax.plot([0.01*charges[0],0.01*charges[-1]], [-1,-1], 'k--', linewidth=line_width)
ax.set_xlabel('$q$ [e]', fontsize=axis_font)
ax.set_ylabel(r'$\cos\theta_0$ [deg]', fontsize=axis_font)
ax.legend(fontsize=ticks_font, loc="lower right")
ax.set_box_aspect(0.75)
plt.tick_params(labelsize=ticks_font)
plt.show()