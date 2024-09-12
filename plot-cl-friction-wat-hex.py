import numpy as np 
import matplotlib.pyplot as plt

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

gamma = 5.19e-2
U0 = 3.72463768115942
PF_MD = 2*np.sqrt(2)/3

cos = lambda x : np.cos(np.deg2rad(x))
sin = lambda x : np.sin(np.deg2rad(x))

ms = 15
mew = 3

theta_inf = 55
theta_sup = 110

class dataset:

    def __init__(self) :
        self.theta_0 = 0
        self.dtheta_0 = 0
        self.capillary_number = []
        self.theta_advancing = []
        self.dtheta_advancing = []
        self.theta_receding = []
        self.dtheta_receding = []

    def convert_ucl(self) :
        self.Ucl = 10*U0*self.capillary_number

    def concatenate_pre_fit(self) :
        self.cos_eq = cos(self.theta_0)
        self.sin_eq = sin(self.theta_0)
        self.cos_adv = cos(self.theta_advancing)
        self.cos_adv = np.concatenate((self.cos_eq,self.cos_adv),axis=None)
        self.sin_adv = sin(self.theta_advancing)
        self.sin_adv = np.concatenate((self.sin_eq,self.sin_adv),axis=None)
        self.ucl_adv = np.concatenate((0,self.Ucl),axis=None)
        self.cos_rec = cos(self.theta_receding)
        self.cos_rec = np.concatenate((self.cos_rec,self.cos_eq),axis=None)
        self.sin_rec = sin(self.theta_receding)
        self.sin_rec = np.concatenate((self.sin_rec,self.sin_eq),axis=None)
        self.ucl_rec = np.concatenate((-self.Ucl,0),axis=None)

    def fit(self) :
        self.pfit_adv, self.cfit_adv = np.polyfit(self.cos_adv,PF_MD*self.sin_adv*self.ucl_adv,deg=1,full=False,cov=True)
        self.pfit_rec, self.cfit_rec = np.polyfit(self.cos_rec,PF_MD*self.sin_rec*self.ucl_rec,deg=1,full=False,cov=True)
        self.mu_f_adv = -gamma/self.pfit_adv[0]
        ptmp = -gamma/(self.pfit_adv[0]+np.sqrt(self.cfit_adv[0][0]))
        mtmp = -gamma/(self.pfit_adv[0]-np.sqrt(self.cfit_adv[0][0]))
        self.dmu_f_adv = 0.5*(ptmp-mtmp)
        self.mu_f_rec = -gamma/self.pfit_rec[0]
        ptmp = -gamma/(self.pfit_rec[0]+np.sqrt(self.cfit_rec[0][0]))
        mtmp = -gamma/(self.pfit_rec[0]-np.sqrt(self.cfit_rec[0][0]))
        self.dmu_f_rec = 0.5*(ptmp-mtmp)

    def eval(self) :
        self.range_adv = np.linspace(self.theta_0,max(self.theta_advancing))
        self.range_rec = np.linspace(min(self.theta_receding),self.theta_0)
        self.ucl_fit_adv = np.polyval(self.pfit_adv,cos(self.range_adv))/(PF_MD*sin(self.range_adv))
        self.ucl_fit_rec = np.polyval(self.pfit_rec,cos(self.range_rec))/(PF_MD*sin(self.range_rec))

dataset_hydrophobic = dataset()
dataset_hydrophilic = dataset()

dataset_hydrophobic.capillary_number = np.array([0.05,0.06,0.07,0.08,0.09,0.10,0.11,0.12,0.13])
dataset_hydrophobic.theta_advancing  = [100.62650382656886,100.28930719768756,101.49072703540607,102.50682692047131,103.50630071793175,104.03118002118691,104.58598660451344,105.14585460578715,106.22988181328239]
dataset_hydrophobic.dtheta_advancing = [0.10795648499997412,0.7610132251041577,0.5496584902347124,0.7437279148235021,0.9418199651037185,1.459336357590054,0.795541402042609,1.8374534647594132,1.8563457642545629]
dataset_hydrophobic.theta_receding   = [90.67912154956836,89.74565369817059,87.6789654672491,86.17935647076085,83.95470829388582,82.15409004814482,80.9647058109186,79.68152290650399,77.50761277543687,]
dataset_hydrophobic.dtheta_receding  = [0.9654650838040482,1.2364502238828408,0.9635675640161807,0.3844651133513608,0.6448731778284227,0.020373150667040818,0.0949479487504874,0.3324788547800921,0.5175793305265373]
dataset_hydrophobic.theta_0          =  97.26240221727994
dataset_hydrophobic.dtheta_0         =  0.081444264241739
dataset_hydrophobic.convert_ucl()

dataset_hydrophilic.capillary_number = np.array([0.03,0.04,0.05,0.06,0.07,0.08])
dataset_hydrophilic.theta_receding   = [73.96950829970838,70.47764894599393,67.37654314204468,65.09460660605647,61.21391862050032,58.44974850729641]
dataset_hydrophilic.dtheta_receding  = [0.0013671030469524226,0.5711529177676056,0.4770425560956042,0.6274126344333553,0.2062195528770019,1.1554748868083315]
dataset_hydrophilic.theta_advancing  = [82.22954655518518,84.91415544814703,84.70763536223221,86.81887291740787,87.66708305164146,88.30398882083733]
dataset_hydrophilic.dtheta_advancing = [1.2324469568179524,0.9557547810030016,1.50942127501294,2.0291532455867127,1.7770515210732754,2.3619278558780437]
dataset_hydrophilic.theta_0          = 80.91084141031868
dataset_hydrophilic.dtheta_0         = 0.09505831156180015
dataset_hydrophilic.convert_ucl()

dataset_hydrophobic.concatenate_pre_fit()
dataset_hydrophilic.concatenate_pre_fit()

dataset_hydrophobic.fit()
dataset_hydrophilic.fit()

dataset_hydrophobic.eval()
dataset_hydrophilic.eval()

plt.plot([theta_inf,theta_sup],[0,0],'k:',linewidth=mew)

plt.errorbar(dataset_hydrophobic.theta_advancing,dataset_hydrophobic.Ucl,xerr=dataset_hydrophobic.dtheta_advancing,
             marker='o',mfc='None',mec='blue',ms=ms,mew=mew,elinewidth=mew,ls='None',ecolor='b')
plt.errorbar(dataset_hydrophobic.theta_receding,-dataset_hydrophobic.Ucl,xerr=dataset_hydrophobic.dtheta_receding,
             marker='o',mfc='None',mec='blue',ms=ms,mew=mew,elinewidth=mew,ls='None',ecolor='b')
plt.errorbar(dataset_hydrophobic.theta_0,0,xerr=dataset_hydrophobic.dtheta_0,
             marker='o',mfc='None',mec='blue',ms=ms,mew=mew,elinewidth=mew,ls='None',ecolor='b')

plt.plot(dataset_hydrophobic.range_adv, dataset_hydrophobic.ucl_fit_adv, 'b--', linewidth=mew)
plt.plot(dataset_hydrophobic.range_rec, dataset_hydrophobic.ucl_fit_rec, 'b--', linewidth=mew)

plt.errorbar(dataset_hydrophilic.theta_advancing,dataset_hydrophilic.Ucl,xerr=dataset_hydrophilic.dtheta_advancing,
             marker='D',mfc='None',mec='red',ms=0.9*ms,mew=mew,elinewidth=mew,ls='None',ecolor='r')
plt.errorbar(dataset_hydrophilic.theta_receding,-dataset_hydrophilic.Ucl,xerr=dataset_hydrophilic.dtheta_receding,
             marker='D',mfc='None',mec='red',ms=0.9*ms,mew=mew,elinewidth=mew,ls='None',ecolor='r')
plt.errorbar(dataset_hydrophilic.theta_0,0,xerr=dataset_hydrophilic.dtheta_0,
             marker='D',mfc='None',mec='red',ms=0.9*ms,mew=mew,elinewidth=mew,ls='None',ecolor='r')

plt.plot(dataset_hydrophilic.range_adv, dataset_hydrophilic.ucl_fit_adv, 'r--', linewidth=mew)
plt.plot(dataset_hydrophilic.range_rec, dataset_hydrophilic.ucl_fit_rec, 'r--', linewidth=mew)

tickfs = 30
labelfs = 35

plt.xlim([theta_inf,theta_sup])
plt.tick_params(axis='both',labelsize=tickfs)
plt.xlabel(r'$\theta$ [$^\circ$]',fontsize=labelfs)
plt.ylabel(r'u$_{cl}$ [m/s]',fontsize=labelfs)
plt.show()