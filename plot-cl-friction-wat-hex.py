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

dataset_hydrophobic.capillary_number = np.array([0.05,
                                                0.06,
                                                0.07,
                                                0.08,
                                                0.09,
                                                0.10,
                                                0.11,
                                                0.12,
                                                0.13])
dataset_hydrophobic.theta_advancing  = [100.62650382656886,
                                        100.28930719768756,
                                        101.49072703540607,
                                        102.50682692047131,
                                        103.50630071793175,
                                        104.03118002118691,
                                        104.58598660451344,
                                        105.14585460578715,
                                        106.22988181328239]
dataset_hydrophobic.dtheta_advancing = [0.10795648499997412,
                                        0.7610132251041577,
                                        0.5496584902347124,
                                        0.7437279148235021,
                                        0.9418199651037185,
                                        1.459336357590054,
                                        0.795541402042609,
                                        1.8374534647594132,
                                        1.8563457642545629]
dataset_hydrophobic.theta_receding   = [90.67912154956836,
                                        89.74565369817059,
                                        87.6789654672491,
                                        86.17935647076085,
                                        83.95470829388582,
                                        82.15409004814482,
                                        80.9647058109186,
                                        79.68152290650399,
                                        77.50761277543687,]
dataset_hydrophobic.dtheta_receding  = [0.9654650838040482,
                                        1.2364502238828408,
                                        0.9635675640161807,
                                        0.3844651133513608,
                                        0.6448731778284227,
                                        0.020373150667040818,
                                        0.0949479487504874,
                                        0.3324788547800921,
                                        0.5175793305265373]
dataset_hydrophobic.theta_0 = 97.26240221727994
dataset_hydrophobic.dtheta_0 = 0.081444264241739
dataset_hydrophobic.convert_ucl()

dataset_hydrophilic.capillary_number = np.array([0.01,
                                                0.03,
                                                0.04,
                                                0.05,
                                                0.06,
                                                0.07,
                                                0.08])
dataset_hydrophilic.theta_receding   = [76.93072551910545,
                                        74.237022303511,
                                        70.8822230622249,
                                        67.22561955933739,
                                        65.55813199791437,
                                        61.408313551193515,
                                        59.00934712879015]
dataset_hydrophilic.dtheta_receding  = [0.5647588336768976,
                                        0.10132095467086089,
                                        0.4596274419461608,
                                        0.23743793157125026,
                                        0.0909553049175571,
                                        0.7221324382111121,
                                        1.0932876063604624]
dataset_hydrophilic.theta_advancing  = [80.0767422214623,
                                        81.68949952534248,
                                        85.25406195064082,
                                        85.2973740206753,
                                        86.90787109090178,
                                        87.32580050217328,
                                        88.02408034293134]
dataset_hydrophilic.dtheta_advancing = [0.7164041367418221,
                                        1.2555269125202528,
                                        1.2271383492379755,
                                        1.4286264613596344,
                                        2.0144925788573715,
                                        1.9023527098953252,
                                        2.2310940809217286]
dataset_hydrophilic.theta_0          = 79.16475132276848
dataset_hydrophilic.dtheta_0         = 1.045510073126497
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

### PLOTTING PF RESULTS ###

U_ref_pf_q60 = []
U_ref_pf_q65 = []
theta_1_q60 = []
theta_2_q60 = []
theta_1_q65 = []
theta_2_q65 = []

U_ref_pf_q60.append(4.1)
theta_1_q60.append(103.179)
theta_2_q60.append(87.2626)
U_ref_pf_q60.append(3.35)
theta_1_q60.append(102.025)
theta_2_q60.append(89.0394)
U_ref_pf_q60.append(2.61)
theta_1_q60.append(100.935)
theta_2_q60.append(90.8438)
U_ref_pf_q60.append(1.86)
theta_1_q60.append(99.8451)
theta_2_q60.append(92.651)

U_ref_pf_q65.append(2.98)
theta_1_q65.append(85.2574)
theta_2_q65.append(66.6926)
U_ref_pf_q65.append(2.61)
theta_1_q65.append(84.5328)
theta_2_q65.append(68.2297)
U_ref_pf_q65.append(1.86)
theta_1_q65.append(83.0139)
theta_2_q65.append(71.4021)
U_ref_pf_q65.append(1.12)
theta_1_q65.append(81.5284)
theta_2_q65.append(74.5444)

U_ref_pf_q60 = np.array(U_ref_pf_q60)
U_ref_pf_q65 = np.array(U_ref_pf_q65)

plt.plot(theta_1_q60, U_ref_pf_q60, 'ks',
    mec='black',ms=0.8*ms,mew=mew,ls='None',
    label='PF')
plt.plot(theta_2_q60, -U_ref_pf_q60, 'ks',
    mec='black',ms=0.8*ms,mew=mew,ls='None')
plt.plot(theta_1_q65, U_ref_pf_q65, 'ks',
    mec='black',ms=0.8*ms,mew=mew,ls='None')
plt.plot(theta_2_q65, -U_ref_pf_q65, 'ks',
    mec='black',ms=0.8*ms,mew=mew,ls='None')
plt.legend(fontsize=25)

### ################### ###

plt.show()