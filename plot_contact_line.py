import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

def array_from_file( filename ):
    my_list = []
    with open(filename, 'r') as f:
        for line in f:
            my_list.append(float(line.split()[0]))
    return np.array(my_list)

"""
folder_root = "/home/michele/densmap/ShearWatPen/"
tag_capillary = ['C010','C008','C006','C004','C002']
num_capillary = [0.1,0.08,0.06,0.04,0.02]
tag_angle = ['Q65','Q65','Q65','Q65','Q65']
num_Uw2 = np.array([1,0.8,0.6,0.4,0.2])*2*0.114
"""
"""
folder_root = "/home/michele/densmap/ShearWatBut/"
tag_capillary = ['C005','C007','C010']
num_capillary = [0.05,0.07,0.10]
tag_angle = ['Q66','Q66','Q66']
num_Uw2 = np.array([0.5,0.7,1])*0.123
"""

folder_root = "/home/michele/densmap/ShearWatHex/"
U0 = 3.72463768115942
tags_angle = ['Q60','Q65']

tag_capillary = dict()
num_capillary = dict()
cl_speed = dict()

tag_capillary['Q60'] = ['C011','C009','C007','C005']
num_capillary['Q60'] = np.array([0.11,0.09,0.07,0.05])
cl_speed['Q60'] = 10*U0*num_capillary['Q60']

tag_capillary['Q65'] = ['C008','C007','C005','C003',]
num_capillary['Q65'] = np.array([0.08,0.07,0.05,0.03])
cl_speed['Q65'] = 10*U0*num_capillary['Q65']

fig, (ax1, ax2) = plt.subplots(2, 1)
std_disp = dict()

### TEMPORARY ###
MD_steady_dispalcement_q65 = []
PFM_steady_displacement_q65 = []
U_q65 = cl_speed['Q65']

L_ref_pf = 18.35

for tag in tags_angle :

    Ndisp = 250
    Mdata = len(tag_capillary[tag])

    # Phase field simulation data
    if tag == 'Q60' :
        U_ref_pf = [4.1,3.35,2.61,1.86]
        pf_data_path = ['DataParvathyHydrophobic/statsU4d1.txt',
            'DataParvathyHydrophobic/statsU3d35.txt',
            'DataParvathyHydrophobic/statsU2d61.txt',
            'DataParvathyHydrophobic/statsU1d86.txt']

    if tag == 'Q65' :
        U_ref_pf = [2.98,2.61,1.86,1.12]
        pf_data_path = ['DataParvathyHydrophilic/statsU2d98.txt',
            'DataParvathyHydrophilic/statsU2d61.txt',
            'DataParvathyHydrophilic/statsU1d86.txt',
            'DataParvathyHydrophilic/statsU1d12.txt']

    tmax = 0
    std_disp[tag] = []

    for i in range(Mdata) :

        folder_name = folder_root+tag_capillary[tag][i]+tag

        t = (1e-3)*array_from_file(folder_name+'/time.txt')
        rl = array_from_file(folder_name+'/radius_lower.txt')
        ru = array_from_file(folder_name+'/radius_upper.txt')
        pl = array_from_file(folder_name+'/position_lower.txt')
        pu = array_from_file(folder_name+'/position_upper.txt')

        cl1 = pl+0.5*rl
        cl2 = pl-0.5*rl
        cl3 = pu+0.5*ru
        cl4 = pu-0.5*ru

        disp_centre = pu-pl

        disp_left = (cl3-cl1)
        disp_right = (cl4-cl2)
        avg_disp = 0.5*(disp_left+disp_right)
        std_disp[tag].append(np.std(avg_disp[len(avg_disp)//2:]))

        t = t-t[0]
        if t[-1] > tmax :
            tmax = t[-1]
        avg_disp = avg_disp-avg_disp[0]

        pf_data = np.loadtxt(pf_data_path[i])
        pf_data_t = pf_data[:,0]
        pf_data_x = pf_data[:,1]

        if tag == 'Q60' :
            ax1.plot(pf_data_t*L_ref_pf/U_ref_pf[i], pf_data_x*L_ref_pf, '--', linewidth=3, c=cm.winter(i/(1.5*Mdata)))
            ax1.plot(t, avg_disp, linewidth=3, label=r"$U_w=$"+str(np.round(cl_speed[tag][i],2))+"nm/ns", c=cm.winter(i/(1.5*Mdata)))
            ax1.set_xlim([0,tmax])

        if tag == 'Q65' :
            ax2.plot(pf_data_t*L_ref_pf/U_ref_pf[i], pf_data_x*L_ref_pf, '--', linewidth=3, c=cm.autumn(i/(1.5*Mdata)))
            ax2.plot(t, avg_disp, linewidth=3, label=r"$U_w=$"+str(np.round(cl_speed[tag][i],2))+"nm/ns", c=cm.autumn(i/(1.5*Mdata)))
            ax2.set_xlim([0,tmax])

        ### TEMPORARY ###
        if tag == 'Q65' :
            MD_steady_dispalcement_q65.append(np.mean(avg_disp[len(avg_disp)//2:]))
            PFM_steady_displacement_q65.append(pf_data_x[-1]*L_ref_pf)

        """
        if tag == 'Q60' :
            pf_data = np.loadtxt(pf_data_path[i])
            pf_data_t = pf_data[:,0]
            pf_data_x = pf_data[:,1]
            ax1.plot(pf_data_t*L_ref_pf/U_ref_pf, pf_data_x*L_ref_pf, '--', linewidth=3, c=cm.winter(i/(1.5*Mdata)))
            ax1.plot(t, avg_disp, linewidth=3, label=r"$U_w=$"+str(np.round(cl_speed[tag][i],2))+"nm/ns", c=cm.winter(i/(1.5*Mdata)))
            ax1.set_xlim([0,tmax])

        if tag == 'Q65' :
            ax2.plot(t, avg_disp, linewidth=3, label=r"$U_w=$"+str(np.round(cl_speed[tag][i],2))+"nm/ns", c=cm.autumn(i/(1.5*Mdata)))
            ax2.set_xlim([0,tmax])
        """
    
    # plt.legend(fontsize=25)

ax2.set_xlabel(r'$t$ [ns]',fontsize=35)
ax1.set_ylabel(r'$\Delta x_{cl}$ [nm]',fontsize=35)
ax2.set_ylabel(r'$\Delta x_{cl}$ [nm]',fontsize=35)
ax1.tick_params(axis='both',labelsize=30)
ax2.tick_params(axis='both',labelsize=30)
# plt.gca().set_xlim(left=0)
# plt.gca().set_ylim(bottom=0)
plt.xlim([0,80])

plt.show()

### STEADY DISPLACEMENT COMPARISON ###
PFM_steady_displacement_q60 = [3.9846,5.8135,7.9198,10.6491]
MD_steady_dispalcement_q60 = [3.6297,5.5872,7.4293,10.3014]
U_q60 = [1.86,2.61,3.35,4.1]

ms = 15
mew = 3

plt.plot(U_q60,PFM_steady_displacement_q60,
    'bs:',mfc='None',ms=ms,mew=mew,lw=mew,label=r"PF, $\theta_0$=97.3$^\circ$")
plt.errorbar(U_q60,MD_steady_dispalcement_q60,yerr=std_disp['Q60'],
    marker='o',mec='blue',mfc='None',ms=ms,mew=mew,lw=mew,ls='None',color='b',ecolor='b',label=r"MD, $\theta_0$=97.3$^\circ$")

### NO PHASE FIELD?
plt.plot(U_q65,PFM_steady_displacement_q65,
    'rx:',mfc='None',ms=1.2*ms,mew=mew,lw=mew,label=r"PF, $\theta_0$=80.9$^\circ$")
plt.errorbar(U_q65,MD_steady_dispalcement_q65,yerr=std_disp['Q65'],
    marker='D',mec='red',mfc='None',ms=ms,mew=mew,lw=mew,ls='None',color='r',ecolor='r',label=r"MD, $\theta_0$=80.9$^\circ$")

plt.legend(fontsize=30, loc='upper left')
plt.tick_params(axis='both',labelsize=30)
plt.xlabel(r'$u_{cl}$ [m/s]',fontsize=35)
plt.ylabel(r'$\Delta x_{cl}$ [nm]',fontsize=35)
plt.show()