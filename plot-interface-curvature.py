import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

def correct_ca (t_old) :
    t_new = []
    for t in t_old :
        if t>0 :
            t_new.append(t)
        else :
            t_new.append(180+t)
    return np.array(t_new)

fr = "InterfaceCurvatureHexane/"
# theta_0 = 97.26240221727994
theta_0 = 80.91084141031868

# ca = np.linspace(0.06,0.13,8)
# lb = ['06', '07', '08', '09', '10', '11', '12', '13']
#       2.23, 2.61, 2.98, 3.35, 3.72, 4.10, 4.47, 4.84

ca = np.linspace(0.03,0.08,6)
lb = ['03','04','05','06', '07', '08']

nex = 2
mdata = len(ca)

for i in range(mdata) :

    # npzfile = np.load(fr+'ca0'+lb[i]+'-q60.npz')
    npzfile = np.load(fr+'ca0'+lb[i]+'-q65.npz')
    z = npzfile['arr_0']
    theta = npzfile['arr_1']
    plt.plot(z[nex:-nex],theta[nex:-nex],lw=2.5,
        c=cm.hot((i+1)/(1.5*mdata)))
    plt.plot(z[nex],theta[nex],'o',ms=8,
        c=cm.hot((i+1)/(1.5*mdata)))
    plt.plot(z[-nex-1],theta[-nex-1],'o',ms=8,
        c=cm.hot((i+1)/(1.5*mdata)))

plt.plot(z[0],theta_0,'ks',ms=10)
plt.plot(z[-1],180-theta_0,'ks',ms=10)
plt.plot([z[0],z[-1]],[theta_0,180-theta_0],'k--',lw=3)

plt.xlabel(r'$z$ [nm]',fontsize=30)
plt.ylabel(r'$\theta$ [deg]',fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)

plt.show()

#######################################################

fp1 = "InterfaceCurvaturePFhydrophobic/"
fp2 = "InterfaceCurvaturePFhydrophilic/"
nex = 1

Lz = 21.15280
# fig, (ax1,ax2,ax3) = plt.subplots(1,3)
fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3)

ticksize = 25
labelsize = 30
legendsize = 27.5

theta_0 = 97.26240221727994
theta_sup = 99
theta_inf = 52

npzfile = np.load(fr+'ca007-q60.npz')
z = npzfile['arr_0']
theta = npzfile['arr_1']
pffile = np.genfromtxt(fp1+'contactangle_2d61.txt')
zpf = pffile[:,0]
tpf = pffile[:,1]
ax1.plot(z[0],theta_0,'ko',ms=10)
ax1.plot(z[-1],180-theta_0,'ko',ms=10)
ax1.plot([z[0],z[-1]],[theta_0,180-theta_0],'k--',lw=3)
ax1.plot(z,theta,'b-',lw=2.5,label='MD')
ax1.plot(zpf,tpf,'b--',lw=2.5,label='PF')
ax1.legend(fontsize=legendsize)
ax1.set_ylabel(r'$\theta_I$ [deg]',fontsize=labelsize)
ax1.tick_params(labelsize=ticksize)
ax1.set_xlim([0,Lz])
ax1.set_ylim([theta_inf,theta_sup])

npzfile = np.load(fr+'ca009-q60.npz')
z = npzfile['arr_0']
theta = npzfile['arr_1']
pffile = np.genfromtxt(fp1+'contactangle_3d35.txt')
zpf = pffile[:,0]
tpf = pffile[:,1]
ax2.plot(z[0],theta_0,'ko',ms=10)
ax2.plot(z[-1],180-theta_0,'ko',ms=10)
ax2.plot([z[0],z[-1]],[theta_0,180-theta_0],'k--',lw=3)
ax2.plot(z,theta,'b-',lw=2.5)
ax2.plot(zpf,tpf,'b--',lw=2.5)
ax2.tick_params(labelsize=ticksize)
ax2.set_xlim([0,Lz])
ax2.set_ylim([theta_inf,theta_sup])

npzfile = np.load(fr+'ca011-q60.npz')
z = npzfile['arr_0']
theta = npzfile['arr_1']
pffile = np.genfromtxt(fp1+'contactangle_4d1.txt')
zpf = pffile[:,0]
tpf = pffile[:,1]
ax3.plot(z[0],theta_0,'ko',ms=10)
ax3.plot(z[-1],180-theta_0,'ko',ms=10)
ax3.plot([z[0],z[-1]],[theta_0,180-theta_0],'k--',lw=3)
ax3.plot(z,theta,'b-',lw=2.5)
ax3.plot(zpf,tpf,'b--',lw=2.5)
ax3.tick_params(labelsize=ticksize)
ax3.set_xlim([0,Lz])
ax3.set_ylim([theta_inf,theta_sup])

theta_0 = 80.91084141031868
theta_sup = 105
theta_inf = 40

npzfile = np.load(fr+'ca003-q65.npz')
z = npzfile['arr_0']
theta = npzfile['arr_1']
pffile = np.genfromtxt(fp2+'contactangleU1d12.txt')
zpf = pffile[:,0]
print(pffile[:,1])
tpf = correct_ca(pffile[:,1])
ax4.plot(z[0],theta_0,'ko',ms=10)
ax4.plot(z[-1],180-theta_0,'ko',ms=10)
ax4.plot([z[0],z[-1]],[theta_0,180-theta_0],'k--',lw=3)
ax4.plot(z,theta,'r-',lw=2.5)
ax4.plot(zpf,tpf,'r--',lw=2.5)
ax4.set_ylabel(r'$\theta_I$ [deg]',fontsize=labelsize)
ax4.set_xlabel(r'$z$ [nm]',fontsize=labelsize)
ax4.tick_params(labelsize=ticksize)
ax4.set_xlim([0,Lz])
ax4.set_ylim([theta_inf,theta_sup])

npzfile = np.load(fr+'ca005-q65.npz')
z = npzfile['arr_0']
theta = npzfile['arr_1']
pffile = np.genfromtxt(fp2+'contactangleU1d86.txt')
zpf = pffile[:,0]
print(pffile[:,1])
tpf = correct_ca(pffile[:,1])
ax5.plot(z[0],theta_0,'ko',ms=10)
ax5.plot(z[-1],180-theta_0,'ko',ms=10)
ax5.plot([z[0],z[-1]],[theta_0,180-theta_0],'k--',lw=3)
ax5.plot(z,theta,'r-',lw=2.5)
ax5.plot(zpf,tpf,'r--',lw=2.5)
ax5.set_xlabel(r'$z$ [nm]',fontsize=labelsize)
ax5.tick_params(labelsize=ticksize)
ax5.set_xlim([0,Lz])
ax5.set_ylim([theta_inf,theta_sup])

"""
npzfile = np.load(fr+'ca007-q65.npz')
z = npzfile['arr_0']
theta = npzfile['arr_1']
pffile = np.genfromtxt(fp2+'contactangleU2d61.txt')
zpf = pffile[:,0]
print(pffile[:,1])
tpf = correct_ca(pffile[:,1])
ax6.plot(z[0],theta_0,'ko',ms=10)
ax6.plot(z[-1],180-theta_0,'ko',ms=10)
ax6.plot([z[0],z[-1]],[theta_0,180-theta_0],'k--',lw=3)
ax6.plot(z,theta,'r-',lw=2.5)
ax6.plot(zpf,tpf,'r--',lw=2.5)
ax6.set_xlabel(r'$z$ [nm]',fontsize=labelsize)
ax6.tick_params(labelsize=ticksize)
ax6.set_xlim([0,Lz])
ax6.set_ylim([theta_inf,theta_sup])
"""

npzfile = np.load(fr+'ca008-q65.npz')
z = npzfile['arr_0']
theta = npzfile['arr_1']
pffile = np.genfromtxt(fp2+'contactangleU2d98.txt')
zpf = pffile[:,0]
print(pffile[:,1])
tpf = correct_ca(pffile[:,1])
ax6.plot(z[0],theta_0,'ko',ms=10)
ax6.plot(z[-1],180-theta_0,'ko',ms=10)
ax6.plot([z[0],z[-1]],[theta_0,180-theta_0],'k--',lw=3)
ax6.plot(z,theta,'r-',lw=2.5)
ax6.plot(zpf,tpf,'r--',lw=2.5)
ax6.set_xlabel(r'$z$ [nm]',fontsize=labelsize)
ax6.tick_params(labelsize=ticksize)
ax6.set_xlim([0,Lz])
ax6.set_ylim([theta_inf,theta_sup])

plt.show()