import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt
import scipy.optimize as opt
import numpy.linalg as alg


def array_from_file( filename ):
    my_list = []
    with open(filename, 'r') as f:
        for line in f:
            my_list.append(float(line.split()[0]))
    return np.array(my_list)

# Rational polynomial fit
def rational(x, p, q) :
    return np.polyval(p, x) / np.polyval(q + [1.0], x)
def rational_4_2(x, p0, p1, p2, p3, q1) :
    return rational(x, [p0, p1, p2, p3], [q1,])
def ratioder_4_2(x, p0, p1, p2, p3, q1) :
    return rational(x, [2*p0*q1, 3*p0+p1*q1, 2*p1, p2-p3*q1], [q1*q1, 2*q1,])

cos = lambda t : np.cos(np.deg2rad(t))
acos = lambda c : np.rad2deg(np.arccos(c))

# Nondimensional quantities
# Initial droplet radius [nm]
R0 = 20
# Visco-capillary time [ps]
tau = 303.5
# Characteristic spreading velocity
V0= R0/tau
print("V0="+str(V0)+"nm/ps")

observables = ['angle_fit', 'angle_l', 'angle_r', 'foot_l', 'foot_r',  'radius_fit']

def pad(v1, v2) :
    vout = v2
    vout[0:len(v1)] = v1
    return vout


class spreading_replicas :

    def __init__(self, folder_names, exc_n=0) :

        self.time = array_from_file(folder_names[0]+'/time.txt')/tau

        n_obs = len( self.time )

        self.obs_arrays = dict()
        self.obs_series = dict()

        self.exc_n = exc_n

        for obs in observables :

            self.obs_arrays[obs] = np.zeros( ( len(folder_names), len(self.time) ) )
            for i in range(len(folder_names)) :
                tmp = array_from_file(folder_names[i]+'/'+obs+'.txt')
                if len(tmp)<n_obs and i>0 :
                    tmp = pad(tmp, self.obs_arrays[obs][0,:])
                self.obs_arrays[obs][i,:] = tmp[:len(self.obs_arrays[obs][i,:])]
            self.obs_series[obs] = np.mean(self.obs_arrays[obs], axis=0)

        self.radius = 0.5 * ( self.obs_series['foot_r'] - self.obs_series['foot_l'] )
        self.radius /= R0

        self.angle = 0.5 * ( self.obs_series['angle_r'] + self.obs_series['angle_l'] )

    def fit_cl(self, fit_fun) :
        popt, pcov = opt.curve_fit(fit_fun, self.time, self.radius)
        self.contact_line_fit = fit_fun(self.time, *popt)
        self.p_fit = popt

    def fit_cl_constraint(self, fit_fun, fit_der, npar, w=10000) :
        # Constraint the derivative to remain nonpositive
        def constr_fun(x, *args) :
            return fit_fun(x, *args) + w*fit_der(x, *args)>0
        popt, pcov = opt.curve_fit(constr_fun, self.time, self.radius, p0=self.p_fit)
        self.contact_line_fit = fit_fun(self.time, *popt)
        self.p_fit = popt

    def compute_velocity(self, deg_num, deg_tot) :
        p_0 = np.array(self.p_fit[0:deg_num])
        p_1 = np.polyder(p_0, m=1)
        q_0 = np.concatenate((self.p_fit[deg_num:deg_tot],[1.0]))
        q_1 = np.polyder(q_0, m=1)
        def v_fit(t) :
            num = ( np.polyval(p_1,t)*np.polyval(q_0,t) - np.polyval(p_0,t)*np.polyval(q_1,t) )
            den = ( np.polyval(q_0,t) )**2
            return num/den
        self.velocity_fit = v_fit(self.time)


class resampler :

    def __init__(self, vel, ang) :

        self.vel = vel
        self.ang = ang
        self.n0 = len(self.vel)

    def resample(self, nbins=None) :

        if nbins==None :
            nbins = int(np.sqrt(self.n0))

        bin_array = np.linspace(np.min(self.vel), np.max(self.vel), nbins)
        histogram = np.zeros(nbins, dtype=int)
        binned_vel = np.digitize(self.vel, bin_array)-1
        for i in range(self.n0) :
            histogram[binned_vel[i]] += 1
        if len(np.where(histogram==0)[0]) > 0 :
            print("[densmap] Warning! You are taking too many bins, try reduce nbins.")
        hat_nb = np.min(histogram[np.nonzero(histogram)])
        print("[densmap] obs. per bin = "+str(hat_nb))
        sampled_idx = np.array([], dtype=int)
        for j in range(nbins) :
            if histogram[j]>0 :
                idx = np.where(binned_vel==j)[0]
                idx_select = idx[rng.randint(0,len(idx),hat_nb)]
                sampled_idx = np.append(sampled_idx,idx_select)

        return self.vel[sampled_idx], self.ang[sampled_idx]
            

def production() :

    # Visualization stuff
    nshow = 10
    ms = 12
    elw = 3

    rep_0p = spreading_replicas(['/home/michele/densmap/SpreadingDataGlycerol/0pR1', 
                                '/home/michele/densmap/SpreadingDataGlycerol/0pR2',
                                '/home/michele/densmap/SpreadingDataGlycerol/0pR3',
                                '/home/michele/densmap/SpreadingDataGlycerol/0pR4',
                                '/home/michele/densmap/SpreadingDataGlycerol/0pR5'], exc_n=50)
    rep_0p.fit_cl(rational_4_2)
    rep_0p.fit_cl_constraint(rational_4_2, ratioder_4_2, 5, w=10000)
    rep_0p.compute_velocity(4,5)

    rep_20p = spreading_replicas(['/home/michele/densmap/SpreadingDataGlycerol/20pR1', 
                                '/home/michele/densmap/SpreadingDataGlycerol/20pR2',
                                '/home/michele/densmap/SpreadingDataGlycerol/20pR3',
                                '/home/michele/densmap/SpreadingDataGlycerol/20pR4',
                                '/home/michele/densmap/SpreadingDataGlycerol/20pR5'], exc_n=50)
    rep_20p.fit_cl(rational_4_2)
    rep_20p.fit_cl_constraint(rational_4_2, ratioder_4_2, 5, w=10000)
    rep_20p.compute_velocity(4,5)

    rep_40p = spreading_replicas(['/home/michele/densmap/SpreadingDataGlycerol/40pR1', 
                                '/home/michele/densmap/SpreadingDataGlycerol/40pR2',
                                '/home/michele/densmap/SpreadingDataGlycerol/40pR3',
                                '/home/michele/densmap/SpreadingDataGlycerol/40pR4',
                                '/home/michele/densmap/SpreadingDataGlycerol/40pR5'], exc_n=150)
    rep_40p.fit_cl(rational_4_2)
    rep_40p.fit_cl_constraint(rational_4_2, ratioder_4_2, 5, w=10000)
    rep_40p.compute_velocity(4,5)

    rep_60p = spreading_replicas(['/home/michele/densmap/SpreadingDataGlycerol/60pR2', 
                                '/home/michele/densmap/SpreadingDataGlycerol/60pR2',
                                '/home/michele/densmap/SpreadingDataGlycerol/60pR3',
                                '/home/michele/densmap/SpreadingDataGlycerol/60pR4',
                                '/home/michele/densmap/SpreadingDataGlycerol/60pR5'], exc_n=100)
    rep_60p.fit_cl(rational_4_2)
    rep_60p.fit_cl_constraint(rational_4_2, ratioder_4_2, 5, w=10000)
    rep_60p.compute_velocity(4,5)

    rep_80p = spreading_replicas(['/home/michele/densmap/SpreadingDataGlycerol/80pR2', 
                                '/home/michele/densmap/SpreadingDataGlycerol/80pR2',
                                '/home/michele/densmap/SpreadingDataGlycerol/80pR3',
                                '/home/michele/densmap/SpreadingDataGlycerol/80pR4',
                                '/home/michele/densmap/SpreadingDataGlycerol/80pR5'], exc_n=50)
    rep_80p.fit_cl(rational_4_2)
    rep_80p.fit_cl_constraint(rational_4_2, ratioder_4_2, 5, w=10000)
    rep_80p.compute_velocity(4,5)

    eta_w = 0.69
    eta_st_0p  = 0.69/eta_w
    eta_st_20p = 1.33/eta_w
    eta_st_40p = 2.51/eta_w
    eta_st_60p = 7.10/eta_w
    eta_st_80p = 45.7/eta_w

    eta_st_0p_ste = 0.01/eta_w
    eta_st_20p_ste = 0.02/eta_w
    eta_st_40p_ste = 0.04/eta_w
    eta_st_60p_ste = 0.20/eta_w
    eta_st_80p_ste = 1.11/eta_w

    gamma_0p  = 5.59e-2
    gamma_20p = 5.56e-2
    gamma_40p = 5.56e-2
    gamma_60p = 5.60e-2
    gamma_80p = 5.49e-2

    # OLD DATA
    cos0_0p  = cos(48.93)
    cos0_20p = cos(47.62)
    cos0_40p = cos(47.12)
    cos0_60p = cos(51.44)
    cos0_80p = cos(53.40)

    # 60p and 80p have not converged yet
    cos0_hat = (cos0_0p+cos0_20p+cos0_40p)/3

    # Number of k steps
    nk = 25

    mu_f_st_0p  = np.zeros(nk)
    mu_f_st_20p = np.zeros(nk)
    mu_f_st_40p = np.zeros(nk)
    mu_f_st_60p = np.zeros(nk)
    mu_f_st_80p = np.zeros(nk)

    theta_0_0p  = np.zeros(nk)
    theta_0_20p = np.zeros(nk)
    theta_0_40p = np.zeros(nk)
    theta_0_60p = np.zeros(nk)
    theta_0_80p = np.zeros(nk)

    # Ficticious capillary number, used only to fit
    caw_0p  = V0*eta_w*rep_0p.velocity_fit[rep_0p.exc_n:]/gamma_0p
    caw_20p = V0*eta_w*rep_20p.velocity_fit[rep_20p.exc_n:]/gamma_20p
    caw_40p = V0*eta_w*rep_40p.velocity_fit[rep_40p.exc_n:]/gamma_40p
    caw_60p = V0*eta_w*rep_60p.velocity_fit[rep_60p.exc_n:]/gamma_60p
    caw_80p = V0*eta_w*rep_80p.velocity_fit[rep_80p.exc_n:]/gamma_80p

    cos_0p  = cos(rep_0p.angle[rep_0p.exc_n:])
    cos_20p = cos(rep_20p.angle[rep_20p.exc_n:])
    cos_40p = cos(rep_40p.angle[rep_40p.exc_n:])
    cos_60p = cos(rep_60p.angle[rep_60p.exc_n:])
    cos_80p = cos(rep_80p.angle[rep_80p.exc_n:])

    res0p  = resampler(eta_st_0p*caw_0p, cos_0p)
    res20p = resampler(eta_st_20p*caw_20p, cos_20p)
    res40p = resampler(eta_st_40p*caw_40p, cos_40p)
    res60p = resampler(eta_st_60p*caw_60p, cos_60p)
    res80p = resampler(eta_st_80p*caw_80p, cos_80p)

    lagmul = 0.1
    pini = np.array([2,cos0_hat])

    for k in range(nk) :

        v0, c0 = res0p.resample(250)
        v20, c20 = res20p.resample(250)
        v40, c40 = res40p.resample(250)
        v60, c60 = res60p.resample(250)
        v80, c80 = res80p.resample(250)

        def objfun_0p(p) :
            return alg.norm(p[0]*v0+c0-p[1]) + lagmul*np.abs(p[1]-cos0_0p)
        def objfun_20p(p) :
            return alg.norm(p[0]*v20+c20-p[1]) + lagmul*np.abs(p[1]-cos0_20p)
        def objfun_40p(p) :
            return alg.norm(p[0]*v40+c40-p[1]) + lagmul*np.abs(p[1]-cos0_40p) 
        def objfun_60p(p) :
            return alg.norm(p[0]*v60+c60-p[1]) + lagmul*np.abs(p[1]-cos0_60p) 
        def objfun_80p(p) :
            return alg.norm(p[0]*v80+c80-p[1]) + lagmul*np.abs(p[1]-cos0_80p) 

        coeff_0p  = opt.minimize( objfun_0p, pini )
        coeff_20p = opt.minimize( objfun_20p, pini )
        coeff_40p = opt.minimize( objfun_40p, pini )
        coeff_60p = opt.minimize( objfun_60p, pini )
        coeff_80p = opt.minimize( objfun_80p, pini )

        mu_f_st_0p[k]  = coeff_0p.x[0]
        mu_f_st_20p[k] = coeff_20p.x[0]
        mu_f_st_40p[k] = coeff_40p.x[0]
        mu_f_st_60p[k] = coeff_60p.x[0]
        mu_f_st_80p[k] = coeff_80p.x[0]

        theta_0_0p[k]  = acos(coeff_0p.x[1])
        theta_0_20p[k] = acos(coeff_20p.x[1])
        theta_0_40p[k] = acos(coeff_40p.x[1])
        theta_0_60p[k] = acos(coeff_60p.x[1])
        theta_0_80p[k] = acos(coeff_80p.x[1])

    mu_f_st_0p_unc = 0.5*(np.max(mu_f_st_0p)-np.min(mu_f_st_0p))
    mu_f_st_20p_unc = 0.5*(np.max(mu_f_st_20p)-np.min(mu_f_st_20p))
    mu_f_st_40p_unc = 0.5*(np.max(mu_f_st_40p)-np.min(mu_f_st_40p))
    mu_f_st_60p_unc = 0.5*(np.max(mu_f_st_60p)-np.min(mu_f_st_60p))
    mu_f_st_80p_unc = 0.5*(np.max(mu_f_st_80p)-np.min(mu_f_st_80p))

    theta_0_0p_unc  = 0.5*(np.max(theta_0_0p)-np.min(theta_0_0p))
    theta_0_20p_unc = 0.5*(np.max(theta_0_20p)-np.min(theta_0_20p))
    theta_0_40p_unc = 0.5*(np.max(theta_0_40p)-np.min(theta_0_40p))
    theta_0_60p_unc = 0.5*(np.max(theta_0_60p)-np.min(theta_0_60p))
    theta_0_80p_unc = 0.5*(np.max(theta_0_80p)-np.min(theta_0_80p))

    mu_f_st_0p  = np.mean(mu_f_st_0p)
    mu_f_st_20p = np.mean(mu_f_st_20p)
    mu_f_st_40p = np.mean(mu_f_st_40p)
    mu_f_st_60p = np.mean(mu_f_st_60p)
    mu_f_st_80p = np.mean(mu_f_st_80p)

    theta_0_0p  = np.mean(theta_0_0p)
    theta_0_20p = np.mean(theta_0_20p)
    theta_0_40p = np.mean(theta_0_40p)
    theta_0_60p = np.mean(theta_0_60p)
    theta_0_80p = np.mean(theta_0_80p)

    print('mu_f_st (0p)  = '+str(mu_f_st_0p)+' +/- '+str(mu_f_st_0p_unc))
    print('mu_f_st (20p) = '+str(mu_f_st_20p)+' +/- '+str(mu_f_st_20p_unc))
    print('mu_f_st (40p) = '+str(mu_f_st_40p)+' +/- '+str(mu_f_st_40p_unc))
    print('mu_f_st (60p) = '+str(mu_f_st_60p)+' +/- '+str(mu_f_st_60p_unc))
    print('mu_f_st (80p) = '+str(mu_f_st_80p)+' +/- '+str(mu_f_st_80p_unc))

    print('theta_0 (0p)  = '+str(theta_0_0p)+' +/- '+str(theta_0_0p_unc))
    print('theta_0 (20p) = '+str(theta_0_20p)+' +/- '+str(theta_0_20p_unc))
    print('theta_0 (40p) = '+str(theta_0_40p)+' +/- '+str(theta_0_40p_unc))
    print('theta_0 (60p) = '+str(theta_0_60p)+' +/- '+str(theta_0_60p_unc))
    print('theta_0 (80p) = '+str(theta_0_80p)+' +/- '+str(theta_0_80p_unc))

    plt.subplot(2, 1, 1)

    plt.ylabel(r'$x_{cl}$ [nm]', fontsize=37.5)

    plt.plot((1e-3)*tau*rep_0p.time, R0*rep_0p.radius, '-', color='silver', linewidth=2.5)
    plt.plot((1e-3)*tau*rep_20p.time, R0*rep_20p.radius, '-', color='silver', linewidth=2.5)
    plt.plot((1e-3)*tau*rep_40p.time, R0*rep_40p.radius, '-', color='silver', linewidth=2.5)
    plt.plot((1e-3)*tau*rep_60p.time, R0*rep_60p.radius, '-', color='silver', linewidth=2.5)
    plt.plot((1e-3)*tau*rep_80p.time, R0*rep_80p.radius, '-', color='silver', linewidth=2.5)
    plt.plot((1e-3)*tau*rep_0p.time, R0*rep_0p.contact_line_fit, 'c--', linewidth=2.5, label=r'$\alpha_g=0.0$')
    plt.plot((1e-3)*tau*rep_20p.time, R0*rep_20p.contact_line_fit, 'm--', linewidth=2.5, label=r'$\alpha_g=0.2$')
    plt.plot((1e-3)*tau*rep_40p.time, R0*rep_40p.contact_line_fit, 'b--', linewidth=2.5, label=r'$\alpha_g=0.4$')
    plt.plot((1e-3)*tau*rep_60p.time, R0*rep_60p.contact_line_fit, 'r--', linewidth=2.5, label=r'$\alpha_g=0.6$')
    plt.plot((1e-3)*tau*rep_80p.time, R0*rep_80p.contact_line_fit, 'g--', linewidth=2.5, label=r'$\alpha_g=0.8$')

    plt.legend(fontsize=22.5)
    plt.xticks([])
    plt.yticks(fontsize=30)

    plt.subplot(2, 1, 2)

    plt.ylabel(r'$\theta$ [deg]', fontsize=37.5)
    plt.xlabel(r'$t$ [ns]', fontsize=35)

    for i in range(5) :
        plt.plot((1e-3)*tau*rep_0p.time, 0.5*(rep_0p.obs_arrays['angle_r'][i,:]+rep_0p.obs_arrays['angle_l'][i,:]), 
            '.', color='silver', markersize=5.0)
        plt.plot((1e-3)*tau*rep_20p.time, 0.5*(rep_20p.obs_arrays['angle_r'][i,:]+rep_20p.obs_arrays['angle_l'][i,:]), 
            '.', color='silver', markersize=5.0)
        plt.plot((1e-3)*tau*rep_40p.time, 0.5*(rep_40p.obs_arrays['angle_r'][i,:]+rep_40p.obs_arrays['angle_l'][i,:]), 
            '.', color='silver', markersize=5.0)
        plt.plot((1e-3)*tau*rep_60p.time, 0.5*(rep_60p.obs_arrays['angle_r'][i,:]+rep_60p.obs_arrays['angle_l'][i,:]), 
            '.', color='silver', markersize=5.0)
        plt.plot((1e-3)*tau*rep_80p.time, 0.5*(rep_80p.obs_arrays['angle_r'][i,:]+rep_80p.obs_arrays['angle_l'][i,:]), 
            '.', color='silver', markersize=5.0)

    plt.plot((1e-3)*tau*rep_0p.time, rep_0p.angle, 'c-', linewidth=2.0, label=r'$\alpha_g=0.0$')
    plt.plot((1e-3)*tau*rep_20p.time, rep_20p.angle, 'm-', linewidth=2.0, label=r'$\alpha_g=0.2$')
    plt.plot((1e-3)*tau*rep_40p.time, rep_40p.angle, 'b-', linewidth=2.0, label=r'$\alpha_g=0.4$')
    plt.plot((1e-3)*tau*rep_60p.time, rep_60p.angle, 'r-', linewidth=2.0, label=r'$\alpha_g=0.6$')
    plt.plot((1e-3)*tau*rep_80p.time, rep_80p.angle, 'g-', linewidth=2.0, label=r'$\alpha_g=0.8$')
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.show()

    plt.subplot(1, 2, 1)

    plt.ylabel(r'$u_{cl}$ [nm/ns]', fontsize=40)
    plt.xlabel(r'$-\cos\theta$', fontsize=40)

    V0E3 = (1e3)*V0
    plt.plot(-cos_0p[0::nshow], V0E3*caw_0p[0::nshow], 'cv', linewidth=2.0, label=r'$\alpha_g=0.0$',
        markeredgewidth=elw, markersize=1.2*ms, markerfacecolor='w')
    plt.plot(-cos_20p[0::nshow], V0E3*caw_20p[0::nshow], 'mH', linewidth=2.0, label=r'$\alpha_g=0.2$',
        markeredgewidth=elw, markersize=1.2*ms, markerfacecolor='w')
    plt.plot(-cos_40p[0::nshow], V0E3*caw_40p[0::nshow], 'bo', linewidth=2.0, label=r'$\alpha_g=0.4$',
        markeredgewidth=elw, markersize=1.1*ms, markerfacecolor='w')
    plt.plot(-cos_60p[0::nshow], V0E3*caw_60p[0::nshow], 'rD', linewidth=2.0, label=r'$\alpha_g=0.6$',
        markeredgewidth=elw, markersize=ms, markerfacecolor='w')
    plt.plot(-cos_80p[0::nshow], V0E3*caw_80p[0::nshow], 'gs', linewidth=2.0, label=r'$\alpha_g=0.8$',
        markeredgewidth=elw, markersize=1.1*ms, markerfacecolor='w')

    range0 = np.linspace(np.min(cos_0p), cos(theta_0_0p), 10)
    plt.plot(-range0, V0E3*(-range0/mu_f_st_0p + cos(theta_0_0p)/mu_f_st_0p)/eta_st_0p, 'k-', linewidth=4.5)
    print("<cos/u> 0% = "+str(-(mu_f_st_0p*eta_st_0p)/V0E3))

    range20 = np.linspace(np.min(cos_20p), cos(theta_0_20p), 10)
    plt.plot(-range20, V0E3*(-range20/mu_f_st_20p + cos(theta_0_20p)/mu_f_st_20p)/eta_st_20p, 'k-', linewidth=4.5)
    print("<cos/u> 20% = "+str(-(mu_f_st_20p*eta_st_20p)/V0E3))

    range40 = np.linspace(np.min(cos_40p), cos(theta_0_40p), 10)
    plt.plot(-range40, V0E3*(-range40/mu_f_st_40p + cos(theta_0_40p)/mu_f_st_40p)/eta_st_40p, 'k-', linewidth=4.5)
    print("<cos/u> 40% = "+str(-(mu_f_st_40p*eta_st_40p)/V0E3))

    range60 = np.linspace(np.min(cos_60p), cos(theta_0_60p), 10)
    plt.plot(-range60, V0E3*(-range60/mu_f_st_60p + cos(theta_0_60p)/mu_f_st_60p)/eta_st_60p, 'k-', linewidth=4.5)
    print("<cos/u> 60% = "+str(-(mu_f_st_60p*eta_st_60p)/V0E3))

    range80 = np.linspace(np.min(cos_80p), cos(theta_0_80p), 10)
    plt.plot(-range80, V0E3*(-range80/mu_f_st_80p + cos(theta_0_80p)/mu_f_st_80p)/eta_st_80p, 'k-', linewidth=4.5)
    print("<cos/u> 80% = "+str(-(mu_f_st_80p*eta_st_80p)/V0E3))

    plt.legend(fontsize=30)
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)

    plt.subplots_adjust(left=0.075, bottom=0.125, right=0.975, top=0.925, wspace=0.3, hspace=0.025)

    ms = 15
    cs = 10
    elw = 3.25
    ct = 2.75

    plt.subplot(1, 2, 2)
    switch_point = 0.5*(eta_st_40p+eta_st_60p)

    scale = np.polyfit(np.log([eta_st_0p,eta_st_20p,eta_st_40p,eta_st_60p,eta_st_80p]), \
        np.log([eta_st_0p*mu_f_st_0p/mu_f_st_0p,eta_st_20p*mu_f_st_20p/mu_f_st_0p,eta_st_40p*mu_f_st_40p/mu_f_st_0p,\
        eta_st_60p*mu_f_st_60p/mu_f_st_0p,eta_st_80p*mu_f_st_80p/mu_f_st_0p]),deg=1)

    print(scale[0])

    plt.plot([eta_st_0p, eta_st_80p], [eta_st_0p, eta_st_80p], 'k--', 
        linewidth=6, label=r'$\sim\eta$ (Duvivier et al.)')
    plt.plot([eta_st_0p, eta_st_80p], [np.sqrt(eta_st_0p), np.sqrt(eta_st_80p)], 'k-', 
        linewidth=6, label=r'$\sim\eta^{1/2}$ (Carlson et al.)')
    plt.plot([eta_st_0p, eta_st_80p], [eta_st_0p**(scale[0]), eta_st_80p**(scale[0])], 'k:', 
        linewidth=6, label=r'$\sim\eta^a$, $a=0.66$')
    plt.loglog(eta_st_0p, eta_st_0p*mu_f_st_0p/mu_f_st_0p, 'cv', 
            markersize=2*ms, markerfacecolor='w', markeredgewidth=1.5*elw)
    plt.loglog(eta_st_20p, eta_st_20p*mu_f_st_20p/mu_f_st_0p, 'mH', 
            markersize=2*ms, markerfacecolor='w', markeredgewidth=1.5*elw)
    plt.loglog(eta_st_40p, eta_st_40p*mu_f_st_40p/mu_f_st_0p, 'bo', 
            markersize=1.75*ms, markerfacecolor='w', markeredgewidth=1.5*elw)
    plt.loglog(eta_st_60p, eta_st_60p*mu_f_st_60p/mu_f_st_0p, 'rD', 
            markersize=1.75*ms, markerfacecolor='w', markeredgewidth=1.5*elw)
    plt.loglog(eta_st_80p, eta_st_80p*mu_f_st_80p/mu_f_st_0p, 'gs', 
            markersize=1.75*ms, markerfacecolor='w', markeredgewidth=1.5*elw)

    plt.tick_params(axis='both', labelsize=35)
    plt.xlabel(r'$\eta/\eta_w$', fontsize=40)
    plt.ylabel(r'$\mu_f/\mu_{f,w}$', fontsize=40)
    plt.legend(fontsize=30)

    plt.subplots_adjust(left=0.075, bottom=0.125, right=0.975, top=0.925, wspace=0.2, hspace=0.025)
    plt.show()

    #####################################################
    ### EFFECTIVE VISCOSITY AND CONTACT LINE FRICTION ###
    #####################################################

    # Ratio t_hb_gly and t_hb_wat
    Chi = 3.0

    alpha_g_eff_20p = 0.187
    alpha_g_eff_40p = 0.354
    alpha_g_eff_60p = 0.509
    alpha_g_eff_80p = 0.721

    eta_eff_20p = 1.147
    eta_eff_40p = 2.094
    eta_eff_60p = 4.411
    eta_eff_80p = 19.74

    mu_f_0p  = mu_f_st_0p*eta_w*eta_st_0p
    mu_f_20p = mu_f_st_20p*eta_w*eta_st_20p
    mu_f_40p = mu_f_st_40p*eta_w*eta_st_40p
    mu_f_60p = mu_f_st_60p*eta_w*eta_st_60p
    mu_f_80p = mu_f_st_80p*eta_w*eta_st_80p

    vol_ratio_g = 1.07
    vol_ratio_w = 5.06

    mu_f_eff_20p = mu_f_20p * ( Chi*vol_ratio_g/alpha_g_eff_20p + vol_ratio_w/(1-alpha_g_eff_20p) )
    mu_f_eff_40p = mu_f_40p * ( Chi*vol_ratio_g/alpha_g_eff_40p + vol_ratio_w/(1-alpha_g_eff_40p) )
    mu_f_eff_60p = mu_f_60p * ( Chi*vol_ratio_g/alpha_g_eff_60p + vol_ratio_w/(1-alpha_g_eff_60p) )
    mu_f_eff_80p = mu_f_80p * ( Chi*vol_ratio_g/alpha_g_eff_80p + vol_ratio_w/(1-alpha_g_eff_80p) )

    print("mu_f_eff_20p = "+str(mu_f_eff_20p))
    print("mu_f_eff_40p = "+str(mu_f_eff_40p))
    print("mu_f_eff_60p = "+str(mu_f_eff_60p))
    print("mu_f_eff_80p = "+str(mu_f_eff_80p))

    fig, ax = plt.subplots()

    plt.plot([eta_eff_20p/eta_w, eta_eff_80p/eta_w], [10*eta_eff_20p/eta_w, 10*eta_eff_80p/eta_w], 'k-', 
        linewidth=5, label=r'$\sim\eta^*$')

    plt.loglog(eta_eff_20p/eta_w, mu_f_eff_20p/mu_f_0p, 'mH', label=r'$\alpha_g=0.2$',
            markersize=1.75*ms, markerfacecolor='w', markeredgewidth=1.5*elw)
    plt.loglog(eta_eff_40p/eta_w, mu_f_eff_40p/mu_f_0p, 'bo', label=r'$\alpha_g=0.4$',
            markersize=1.75*ms, markerfacecolor='w', markeredgewidth=1.5*elw)
    plt.loglog(eta_eff_60p/eta_w, mu_f_eff_60p/mu_f_0p, 'rD', label=r'$\alpha_g=0.6$',
            markersize=1.5*ms, markerfacecolor='w', markeredgewidth=1.5*elw)
    plt.loglog(eta_eff_80p/eta_w, mu_f_eff_80p/mu_f_0p, 'gs', label=r'$\alpha_g=0.8$',
            markersize=1.5*ms, markerfacecolor='w', markeredgewidth=1.5*elw)

    plt.tick_params(axis='both', labelsize=35)
    plt.xlabel(r'$\eta^*/\eta_w$', fontsize=40)
    plt.ylabel(r'$\mu_f^*/\mu_{f,w}$', fontsize=40)
    plt.legend(fontsize=30)
    ax.set_box_aspect(1)
    plt.show()


def test_tip4p() :

    # Visualization stuff
    nshow = 10
    ms = 12
    elw = 3

    rep_t4 = spreading_replicas(['/home/michele/densmap/SpreadingDataGlycerol/T4R1', 
                                '/home/michele/densmap/SpreadingDataGlycerol/T4R2',
                                '/home/michele/densmap/SpreadingDataGlycerol/T4R3',
                                '/home/michele/densmap/SpreadingDataGlycerol/T4R4',
                                '/home/michele/densmap/SpreadingDataGlycerol/T4R5'], exc_n=50)
    rep_t4.fit_cl(rational_4_2)
    rep_t4.fit_cl_constraint(rational_4_2, ratioder_4_2, 5, w=10000)
    rep_t4.compute_velocity(4,5)

    rep_0p = spreading_replicas(['/home/michele/densmap/SpreadingDataGlycerol/0pR1', 
                                '/home/michele/densmap/SpreadingDataGlycerol/0pR2',
                                '/home/michele/densmap/SpreadingDataGlycerol/0pR3',
                                '/home/michele/densmap/SpreadingDataGlycerol/0pR4',
                                '/home/michele/densmap/SpreadingDataGlycerol/0pR5'], exc_n=50)
    rep_0p.fit_cl(rational_4_2)
    rep_0p.fit_cl_constraint(rational_4_2, ratioder_4_2, 5, w=10000)
    rep_0p.compute_velocity(4,5)

    eta_w = 0.69
    eta_st_0p = 1
    esa_st_t4 = 0.855/eta_w
    gamma_0p = 5.59e-2
    gamma_t4 = 6.14e-2
    cos0  = cos(48.93)

    caw_0p = V0*eta_w*rep_0p.velocity_fit[rep_0p.exc_n:]/gamma_0p
    cos_0p = cos(rep_0p.angle[rep_0p.exc_n:])
    res0p = resampler(eta_st_0p*caw_0p, cos_0p)

    caw_t4 = V0*eta_w*rep_t4.velocity_fit[rep_t4.exc_n:]/gamma_t4
    cos_t4 = cos(rep_t4.angle[rep_t4.exc_n:])
    rest4 = resampler(esa_st_t4*caw_t4, cos_t4)

    lagmul = 10
    pini = np.array([2,cos0])
    
    plt.subplot(2, 1, 1)
    plt.ylabel(r'$x_{cl}$ [nm]', fontsize=37.5)
    plt.plot((1e-3)*tau*rep_t4.time, R0*rep_t4.radius-R0*rep_t4.contact_line_fit[0], '-', color='silver', linewidth=2.5)
    plt.plot((1e-3)*tau*rep_t4.time, R0*rep_t4.contact_line_fit-R0*rep_t4.contact_line_fit[0], 'k--', linewidth=2.5)
    plt.plot((1e-3)*tau*rep_0p.time, R0*rep_0p.radius-R0*rep_0p.contact_line_fit[0], '-', color='silver', linewidth=2.5)
    plt.plot((1e-3)*tau*rep_0p.time, R0*rep_0p.contact_line_fit-R0*rep_0p.contact_line_fit[0], 'c--', linewidth=2.5)
    plt.xticks([])
    plt.yticks(fontsize=30)
    plt.subplot(2, 1, 2)
    plt.ylabel(r'$\theta$ [deg]', fontsize=37.5)
    plt.xlabel(r'$t$ [ns]', fontsize=35)
    for i in range(5) :
        plt.plot((1e-3)*tau*rep_t4.time, 0.5*(rep_t4.obs_arrays['angle_r'][i,:]+rep_t4.obs_arrays['angle_l'][i,:]), 
            '.', color='silver', markersize=5.0)
        plt.plot((1e-3)*tau*rep_0p.time, 0.5*(rep_0p.obs_arrays['angle_r'][i,:]+rep_0p.obs_arrays['angle_l'][i,:]), 
            '.', color='silver', markersize=5.0)
    plt.plot((1e-3)*tau*rep_t4.time, rep_t4.angle, 'k-', linewidth=2.0, label=r'TIP4P/2005')
    plt.plot((1e-3)*tau*rep_0p.time, rep_0p.angle, 'c-', linewidth=2.0, label=r'SPC/E')
    plt.legend(fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.show()


def test_pure_glycerol() :

    # Visualization stuff
    nshow = 10
    ms = 12
    elw = 3

    rep_g = spreading_replicas(['/home/michele/densmap/SpreadingDataGlycerol/100pR1', 
                                '/home/michele/densmap/SpreadingDataGlycerol/100pR2',
                                '/home/michele/densmap/SpreadingDataGlycerol/100pR3',
                                '/home/michele/densmap/SpreadingDataGlycerol/100pR4',
                                '/home/michele/densmap/SpreadingDataGlycerol/100pR5'], exc_n=50)
    rep_g.fit_cl(rational_4_2)
    rep_g.fit_cl_constraint(rational_4_2, ratioder_4_2, 5, w=10000)
    rep_g.compute_velocity(4,5)

    rep_w = spreading_replicas(['/home/michele/densmap/SpreadingDataGlycerol/0pR1', 
                                '/home/michele/densmap/SpreadingDataGlycerol/0pR2',
                                '/home/michele/densmap/SpreadingDataGlycerol/0pR3',
                                '/home/michele/densmap/SpreadingDataGlycerol/0pR4',
                                '/home/michele/densmap/SpreadingDataGlycerol/0pR5'], exc_n=50)
    rep_w.fit_cl(rational_4_2)
    rep_w.fit_cl_constraint(rational_4_2, ratioder_4_2, 5, w=10000)
    rep_w.compute_velocity(4,5)

    eta_w     = 0.69
    eta_st_0p = 1
    gamma     = 5.59e-2
    cos0      = cos(48.93)

    caw_0p = V0*eta_w*rep_g.velocity_fit[rep_g.exc_n:]/gamma
    cos_0p = cos(rep_g.angle[rep_g.exc_n:])
    res0p  = resampler(eta_st_0p*caw_0p, cos_0p)

    lagmul = 10
    pini = np.array([2,cos0])
    
    t_gly = 4*(18.992+18.607+18.857+18.981+19.098)/5
    t_wat = 4*20.261

    plt.subplot(1, 2, 1)
    plt.ylabel(r'$x_{cl}$ [nm]', fontsize=27.5)
    plt.xlabel(r'$t$ [ns]', fontsize=27.5)
    plt.xticks(fontsize=25.0)
    plt.yticks(fontsize=25.0)
    plt.plot((1e-3)*tau*rep_w.time, R0*rep_w.contact_line_fit, 'c--', linewidth=4)
    plt.title("Pure water", fontsize=30.0)
    plt.subplot(1, 2, 2)
    plt.plot((1e-3)*tau*rep_g.time, R0*rep_g.contact_line_fit, 'k--', linewidth=4)
    plt.ylabel(r'$x_{cl}$ [nm]', fontsize=27.5)
    plt.xlabel(r'$t$ [ns]', fontsize=27.5)
    plt.xticks(fontsize=25.0)
    plt.yticks(fontsize=25.0)
    plt.title("Pure glycerol", fontsize=30.0)
    plt.show()


if __name__ == "__main__" :
    production()
    # test_tip4p()
    # test_pure_glycerol()