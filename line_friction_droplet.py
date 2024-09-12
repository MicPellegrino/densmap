import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as smg
import scipy.signal as sgn
import scipy.optimize as opt
import numpy.random

# Plotting params
plot_sampling = 1
plot_tcksize = 17.5

# Reference units
mu      = 0.877                 # mPa*s
gamma   = 57.8                  # mPa*m
U_ref   = (gamma/mu)*1e-3       # nm/ps

def array_from_file( filename ):
    my_list = []
    with open(filename, 'r') as f:
        for line in f:
            my_list.append(float(line.split()[0]))
    return np.array(my_list)

def cos( theta ) :
    return np.cos(np.deg2rad(theta))
cos_vec = np.vectorize(cos)

def sin( theta ) :
    return np.sin(np.deg2rad(theta))
sin_vec = np.vectorize(sin)

tan = lambda t : sin(t)/cos(t)

# Rational polynomial fit
def rational(x, p, q):
    return np.polyval(p, x) / np.polyval(q + [1.0], x)
def rational_3_3(x, p0, p1, p2, q1, q2):
    return rational(x, [p0, p1, p2], [q1, q2])
def rational_4_2(x, p0, p1, p2, p4, q1):
    return rational(x, [p0, p1, p2, p4], [q1,])

# Key: input file name
# Value: advancing/receding
# Use dataframes instead?
input_files = dict()
# Q1
input_files['Q1'] = { 'SpreadingData/FlatQ1' : ('adv', True),
                'SpreadingData/FlatQ1REC' : ('rec', False),
                'SpreadingData/FlatQ1REC2' : ('rec', False)}
# Q2
input_files['Q2'] = { 'SpreadingData/FlatQ2' : ('adv', True),
                'SpreadingData/FlatQ2ADV' : ('adv', False), 
                'SpreadingData/FlatQ2REC' : ('rec', False)}
# Q3
input_files['Q3'] = { 'SpreadingData/FlatQ3' : ('adv', True),
                'SpreadingData/FlatQ3ADV' : ('adv', False), 
                'SpreadingData/FlatQ3REC' : ('rec', False), 
                'SpreadingData/FlatQ3CAP' : ('adv', False), 
                'SpreadingData/FlatQ3REC2': ('rec', False)}
# Q4
input_files['Q4'] = { 'SpreadingData/FlatQ4' : ('adv', True),
                'SpreadingData/FlatQ4ADV' : ('adv', False), 
                'SpreadingData/FlatQ4REC' : ('rec', False) }

equilibrium_contact_angle = dict()
"""
equilibrium_contact_angle['Q1'] = 126.01
equilibrium_contact_angle['Q2'] = 94.9
equilibrium_contact_angle['Q3'] = 70.5
equilibrium_contact_angle['Q4'] = 39.2
"""
# Use the same values of the MCL paper for consistency? NO!
"""
equilibrium_contact_angle['Q1'] = 127
equilibrium_contact_angle['Q2'] = 95
equilibrium_contact_angle['Q3'] = 69
equilibrium_contact_angle['Q4'] = 38
"""
# Using 1st order interpolation
equilibrium_contact_angle['Q1'] = 127.50
equilibrium_contact_angle['Q2'] = 96.01
equilibrium_contact_angle['Q3'] = 71.97
equilibrium_contact_angle['Q4'] = 38.71

substrate_color = dict()
substrate_color['Q1'] = 'c'
substrate_color['Q2'] = 'g'
substrate_color['Q3'] = 'r'
substrate_color['Q4'] = 'm'

# Shear droplet data

theta = dict()
ca = dict()
U = dict()
err_theta = dict()

theta['Q1'] = np.array([130.88, 131.49, 137.93, 142.13, 129.7410333158549, 126.65, 126.06, 124.32, 125.42])
ca['Q1'] = 0.5*np.array([0.15, 0.30, 0.60, 0.90, 0.0, -0.15, -0.30, -0.60, -0.90])
err_theta['Q1'] =  np.array([1.027, 0.3170, 0.3123, 0.2346, 0, 1.745, 1.0524, 1.3838, 3.128])

theta['Q2'] = np.array([101.21042472, 104.01678041, 106.17731627, 108.88678425, 111.04525937, 97.7792911518621, 94.95432237, 92.13575175, 88.46459205, 84.38834944, 79.88718327])
ca['Q2'] = 0.5*np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.0, -0.05, -0.10, -0.15, -0.20, -0.25])
err_theta['Q2'] =  np.array([3.4160, 0.9342, 0.7004, 0.7892, 4.4365, 0, 5.1389, 1.5293,  2.194,  1.553, 6.9790])

theta['Q3'] = np.array([77.18496895, 77.85253598, 78.84062248, 81.5119225, 84.13935051, 72.34125045506745, 65.86140487, 63.75901713, 61.81662731, 58.91360746, 54.28781229])
ca['Q3'] = np.array([0.015,  0.025,  0.03 ,  0.04 ,  0.05 , 0.0, -0.015 ,-0.025 ,-0.03 , -0.04 , -0.05])
err_theta['Q3'] =  np.array([0.6232, 0.9263, 1.578,  1.773, 2.925, 0, 0.5128, 0.4506, 0.9719, 2.350, 4.213])

theta['Q4'] = np.array([42.51470603, 45.30519045, 45.4518236, 39.093364496646906, 37.05776924, 33.47130752, 29.93002039])
ca['Q4'] = 0.5*np.array([0.010, 0.015, 0.020, 0.0, -0.010, -0.015, -0.020])
err_theta['Q4'] =  np.array([0.2937,  0.8878,  0.2764, 0, 0.3322, 0.2743, 0.1396])

"""
theta['Q1'] = np.array(    [130.88, 131.49, 137.93, 142.13, 127.50, 126.65, 126.06, 124.32, 125.42])
ca['Q1']    = 0.5*np.array([0.15,   0.30,   0.60,   0.90,   0.0,    -0.15,  -0.30,  -0.60,  -0.90])
theta['Q2'] = np.array(    [107.57, 109.99, 111.87, 112.82, 115.75, 103.63, 101.45,  100.33, 97.47, 93.60, 89.72])
ca['Q2']    = 0.5*np.array([0.05,   0.10,   0.15,   0.20,   0.25,   0.0,    -0.05,   -0.10,  -0.15, -0.20, -0.25])
theta['Q3'] = np.array(    [77.18, 77.85, 78.84, 81.51, 84.14, 71.97, 65.86, 63.76, 61.82, 58.91, 54.29])
ca['Q3']    = 0.5*np.array([0.03,  0.05,  0.06 , 0.08 , 0.1 ,  0.0,   -0.03, -0.05, -0.06, -0.08, -0.1])
theta['Q4'] = np.array(    [42.51, 45.305, 45.452, 38.71, 37.058, 33.471, 29.930])
ca['Q4']    = 0.5*np.array([0.010, 0.015,  0.020,  0.0,   -0.010, -0.015, -0.020])
"""

for k in ca.keys() :
    U[k] = ca[k]*U_ref

class ContactLine :

    def __init__(self, t, fl, fr, al, ar, r, ac) :
        self.time = t
        self.foot_l = fl
        self.foot_r = fr
        self.angle_l = al
        self.angle_r = ar
        self.radius = r
        self.angle_circle = ac
        self.contact_line_pos = 0.5*(fr-fl)
        self.idx_steady=-1
        self.theta0 = 0

    def fit_cl(self, fit_fun) :
        popt, pcov = opt.curve_fit(fit_fun, self.time, self.contact_line_pos)
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
        
    def average_angles(self, N_avg) :
        ca = 0.5*(self.angle_r+self.angle_l)
        self.angle_l_avg = np.convolve(self.angle_l, np.ones(N_avg)/N_avg, mode='same')
        self.angle_r_avg = np.convolve(self.angle_r, np.ones(N_avg)/N_avg, mode='same')
        self.contact_angle = np.convolve(ca, np.ones(N_avg)/N_avg, mode='same')

    def estimate_theta0(self, N_last) :
        self.theta0 = np.mean(self.angle_circle[-N_last:])


class ContactLineModel :

    def __init__(self, input_files, 
        model_function=None, fit_cos=False, vmin=1e-4, delta_t_avg=880, T_off=500, dt=10.0, frac_steady=0.667, theta0=None) :

        self.vmin = vmin
        self.delta_t_avg = delta_t_avg
        self.N = int(T_off/dt)
        N_avg = int(delta_t_avg/dt)
        self.cl_container = []
        self.avg_theta0 = 0
        n_avg  = 0

        # Use later on...
        self.model_function = model_function
        self.fit_cos = fit_cos

        self.set_contact_line_data(input_files, N_avg, n_avg, frac_steady)

        self.compute_reduced_quantities(N_avg, theta0)

    def __str__(self) :
        return "Average theta0 = "+str(self.avg_theta0)

    def set_contact_line_data(self, input_files, N_avg, n_avg, frac_steady) :
        for k in input_files.keys() :
            
            t = array_from_file(k+'/time.txt')
            fl = array_from_file(k+'/foot_l.txt')
            fr = array_from_file(k+'/foot_r.txt')
            al = array_from_file(k+'/angle_l.txt')
            ar = array_from_file(k+'/angle_r.txt')
            r = array_from_file(k+'/radius_fit.txt')
            ac = array_from_file(k+'/angle_fit.txt')
            CL = ContactLine( t[self.N:], fl[self.N:] ,fr[self.N:], \
                                al[self.N:], ar[self.N:], r[self.N:], ac[self.N:] )
            if input_files[k][0] == 'adv' :
                CL.fit_cl(rational_4_2)
                CL.compute_velocity(4, 5)
            elif input_files[k][0] == 'rec' :
                CL.fit_cl(rational_3_3)
                CL.compute_velocity(3, 5)
            CL.idx_steady = np.argmin(np.abs(CL.velocity_fit-self.vmin))
            CL.average_angles(N_avg)

            N_eq = int(frac_steady*len(CL.time[N_avg:-N_avg]))
            CL.estimate_theta0(N_eq)
            
            if input_files[k][1] == True :
                self.avg_theta0 += CL.theta0
                n_avg += 1
            self.cl_container.append(CL)
        self.avg_theta0 /= n_avg
        self.N_avg = N_avg
        self.n_avg = n_avg

    def compute_reduced_quantities(self, N_avg, theta0=None) :
        if not(theta0==None) :
            self.avg_theta0 = theta0
        self.reduced_velocity_micro = []
        self.reduced_cosine_micro = []
        self.angle_micro = []
        for i in range(0,len(self.cl_container)) :
            nidx = min(len(self.cl_container[i].velocity_fit)-N_avg,self.cl_container[i].idx_steady)
            self.reduced_velocity_micro = np.concatenate( 
                (self.reduced_velocity_micro,
                self.cl_container[i].velocity_fit[N_avg:nidx]/U_ref), 
                axis=None )
            self.reduced_cosine_micro = np.concatenate(
                (self.reduced_cosine_micro,
                cos(self.avg_theta0)-cos(self.cl_container[i].contact_angle[N_avg:nidx])), 
                axis=None )
            self.angle_micro = np.concatenate(
                (self.angle_micro,
                self.cl_container[i].contact_angle[N_avg:nidx]),
                axis=None)

    def fit_model(self, shear_data=None, weight=0.01, mode='direct') :

        # mode='direct': just fit a model, no fear of overfitting!
        # mode='validation': minimize the prediction error (20% testing)

        s = np.ones(len(self.reduced_velocity_micro))
        if shear_data==None :
            s = s/np.sum(s)
            c = self.reduced_velocity_micro
            if self.fit_cos :
                t = self.reduced_cosine_micro
            else :
                t = self.angle_micro, shear_data[0]
        else:
            s = np.concatenate( (s, weight*np.ones(len(shear_data[1]))), axis=None )
            s = s/np.sum(s)
            c = np.concatenate( (self.reduced_velocity_micro, shear_data[1]), axis=None )
            if self.fit_cos :
                t = np.concatenate( (self.reduced_cosine_micro, shear_data[0]), axis=None )
            else :
                t = np.concatenate( (self.angle_micro, shear_data[0]), axis=None )

        mask = np.random.rand(len(c)) <= 0.8
        training_c = c[mask]
        training_t = t[mask]
        training_s = s[mask]
        testing_c = c[~mask]
        testing_t = t[~mask]
        testing_s = s[~mask]

        if mode=='direct' :
            self.popt, self.pcov = opt.curve_fit(self.model_function, t, c, sigma=s)
            err = np.sqrt( np.sum( (1/s)*( c - self.model_function(t, *self.popt) )**2 ) )

        else :
            self.popt, self.pcov = opt.curve_fit(self.model_function, training_t, training_c, sigma=training_s)
            err = np.sqrt( np.sum( (1/testing_s)*( testing_c - self.model_function(testing_t, *self.popt) )**2 ) )
        
        t_range = np.linspace(min(shear_data[0]), max(shear_data[0]), 100)

        return self.popt, err, t_range, self.model_function(t_range, *self.popt)

    def plot_data(self) :
        plt.plot(self.reduced_cosine_micro, self.reduced_velocity_micro, 'k.')
        plt.xlabel(r"$\cos(\theta_0)-\cos(\theta)$")
        plt.ylabel(r"$U_{cl}/U_{ref}$")
        plt.show()

    def plot_models(self) :
        if self.fit_cos:
            plt.plot(self.reduced_cosine_micro[0::plot_sampling], self.reduced_velocity_micro[0::plot_sampling], 'bo')
            cos_vector = np.linspace(min(self.reduced_cosine_micro), max(self.reduced_cosine_micro), 250)
            plt.plot(cos_vector, self.model_function(cos_vector, *self.popt), 'r-')
            plt.xlabel(r"$\cos(\theta_0)-\cos(\theta)$")
        else:
            plt.plot(self.angle_micro[0::plot_sampling], self.reduced_velocity_micro[0::plot_sampling], 'bo')
            angle_vector = np.linspace(min(self.angle_micro), max(self.angle_micro), 250)
            plt.plot(angle_vector, self.model_function(angle_vector, *self.popt), 'r-')
            plt.xlabel(r"$\theta$")
        plt.ylabel(r"$U_{cl}/U_{ref}$")
        plt.show()


def main_pp() :

    clm = dict()
    N_sample_plot = plot_sampling
    min_cos = 2
    max_cos = -2
    min_vel = 100
    max_vel = -100

    expa_min = 10
    expa_max = 0

    for k in input_files.keys() :    
        
        clm[k] = ContactLineModel(input_files[k], vmin=1e-4, delta_t_avg=10, frac_steady=0.75)
        print(clm[k])
        if max(clm[k].reduced_cosine_micro) > max_cos :
            max_cos = max(clm[k].reduced_cosine_micro)
        if min(clm[k].reduced_cosine_micro) < min_cos :
            min_cos = min(clm[k].reduced_cosine_micro)
        if max(clm[k].reduced_velocity_micro) > max_vel :
            max_vel = max(clm[k].reduced_velocity_micro)
        if min(clm[k].reduced_velocity_micro) < min_vel :
            min_vel = min(clm[k].reduced_velocity_micro)
        sparse_x = clm[k].angle_micro[::N_sample_plot]
        sparse_y = clm[k].reduced_velocity_micro[::N_sample_plot]
        t0lab = ""
        if k =="Q1" :
            t0lab = r'$\theta_0=127^\circ$'
        if k =="Q2" :
            t0lab = r'$\theta_0=95^\circ$'
        if k =="Q3" :
            t0lab = r'$\theta_0=69^\circ$'
        if k =="Q4" :
            t0lab = r'$\theta_0=38^\circ$'
        plt.plot(sparse_x, sparse_y, '.', \
            markersize=37.5, markeredgecolor='None', alpha=0.125, color=substrate_color[k])
        plt.plot(theta[k], ca[k], 'D', \
            label=t0lab, markersize=17.5, markeredgecolor='k', markeredgewidth=4, color=substrate_color[k])
        # Models for contact line friction
        def therm_fun(t, b0, b1) :
            return b0 * np.exp(-b1*(0.5*sin(t)+cos(t))**2) * (cos(clm[k].avg_theta0)-cos(t))
        def mkt_fun(t, a1, a3) :
            return a1*t + a3*(t**3)
        clm[k].model_function = therm_fun
        p_thermo, err, t_fit, c_fit = clm[k].fit_model(shear_data=[theta[k],ca[k]])
        plt.plot(t_fit, c_fit, 'k:', linewidth=3.0)
        mu_th = 1.0/p_thermo[0]

        expa = p_thermo[1]
        if expa > expa_max :
            expa_max = expa
        if expa <= expa_min :
            expa_min = expa

        print("mu* (therm) = "+str( mu_th ))
        print("expa = "+str(expa))
        print("---------------------------------------------")

    print("expa range = ["+str(expa_min)+","+str(expa_max)+"]")
    
    plt.plot([20, 150], [0.0, 0.0], 'k--', linewidth=3.0)
    plt.xlabel(r'$\theta$ [deg]', fontsize=50)
    plt.ylabel(r'Ca$_{cl}$ [1]', fontsize=50)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.legend(fontsize=40)
    plt.ylim([-0.6, 0.6])
    plt.show()

def main(w=0.01, mode='direct', show_plots=False, Na=40) :

    expa_range = np.linspace(0.5, 2.5,Na)
    err_vals = np.zeros(Na)

    clm = dict()

    err_opt = 0

    mu_th = []

    for k in input_files.keys() :    
            clm[k] = ContactLineModel(input_files[k], vmin=1e-4, delta_t_avg=10, frac_steady=0.75)

    for i in range(Na) :

        err_i = 0

        mu_th.append(dict())

        for k in input_files.keys() :    

            # Models for contact line friction
            def therm_fun(t, b0) :
                return b0 * np.exp(-expa_range[i]*(0.5*sin(t)+cos(t))**2) * (cos(clm[k].avg_theta0)-cos(t))
            clm[k].model_function = therm_fun
            p_thermo, err, t_fit, c_fit = clm[k].fit_model(shear_data=[theta[k],ca[k]], weight=w, mode=mode)
            mu_th_ik = 1.0/p_thermo[0]
            mu_th[-1][k] = mu_th_ik

            err_i+=err

        err_vals[i] = err_i
        if i == 0 or (err_i < err_opt):
            err_opt = err_i
            i_opt = i

    expa_opt = expa_range[i_opt]

    print("a_opt = "+str(expa_opt))

    print("mu_th_opt = "+str(mu_th[i_opt]))

    if show_plots :
        plt.plot(expa_range, err_vals, 'b-', linewidth=2.0)
        plt.plot(expa_opt, err_opt, 'rs', markersize=7.5)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel(r'$a$', fontsize=25)
        plt.ylabel('err', fontsize=25)
        plt.show()

    if show_plots :

        N_sample_plot = plot_sampling
        min_cos = 2
        max_cos = -2
        min_vel = 100
        max_vel = -100

        fig = plt.figure()
        ax = fig.add_subplot(111)

        for k in input_files.keys() :

            if max(clm[k].reduced_cosine_micro) > max_cos :
                max_cos = max(clm[k].reduced_cosine_micro)
            if min(clm[k].reduced_cosine_micro) < min_cos :
                min_cos = min(clm[k].reduced_cosine_micro)
            if max(clm[k].reduced_velocity_micro) > max_vel :
                max_vel = max(clm[k].reduced_velocity_micro)
            if min(clm[k].reduced_velocity_micro) < min_vel :
                min_vel = min(clm[k].reduced_velocity_micro)
            sparse_x = clm[k].angle_micro[::N_sample_plot]
            sparse_y = clm[k].reduced_velocity_micro[::N_sample_plot]
            t0lab = ""
            if k =="Q1" :
                t0lab = r'$\theta_0=127^\circ$'
            if k =="Q2" :
                t0lab = r'$\theta_0=95^\circ$'
            if k =="Q3" :
                t0lab = r'$\theta_0=69^\circ$'
            if k =="Q4" :
                t0lab = r'$\theta_0=38^\circ$'
            # Models for contact line friction
            def therm_fun(t, b0, b1) :
                return b0 * np.exp(-b1*(0.5*sin(t)+cos(t))**2) * (cos(clm[k].avg_theta0)-cos(t))
            t_min = min(np.mean(sparse_x)-2.5*np.std(sparse_x), min(theta[k]))
            t_max = max(np.mean(sparse_x)+2.5*np.std(sparse_x), max(theta[k]))
            t_fit = np.linspace(t_min, t_max, 100)
            # t_fit = np.linspace(min(theta[k]), max(theta[k]), 100)
            # print(mu_th[i_opt][k])
            c_fit = therm_fun(t_fit, 1/mu_th[i_opt][k], expa_opt)
            # ax.plot(sparse_x, sparse_y, '.', \
            #     markersize=37.5, markeredgecolor='None', alpha=0.125, color=substrate_color[k])
            ax.plot(sparse_x, U_ref*sparse_y, '.', \
                markersize=37.5, markeredgecolor='None', alpha=0.125, color=substrate_color[k])
            # ax.errorbar(theta[k], ca[k], xerr=err_theta[k], fmt='k.', elinewidth=2.5)
            ax.errorbar(theta[k], U_ref*ca[k], xerr=err_theta[k], fmt='k.', elinewidth=2.5)
            # ax.plot(theta[k], ca[k], 'D', \
            #     label=t0lab, markersize=17.5, markeredgecolor='k', markeredgewidth=2, color=substrate_color[k])
            ax.plot(theta[k], U_ref*ca[k], 'D', \
                label=t0lab, markersize=17.5, markeredgecolor='k', markeredgewidth=2, color=substrate_color[k])
            if k =="Q4" :
                # ax.plot(t_fit, c_fit, 'k-', linewidth=3.5, label='Eq. (2) fit')
                ax.plot(t_fit, U_ref*c_fit, 'k-', linewidth=3.5, label='model fit')
            else :
                # ax.plot(t_fit, c_fit, 'k-', linewidth=3.5)
                ax.plot(t_fit, U_ref*c_fit, 'k-', linewidth=3.5)

        ax.plot([12.5, 157.5], [0.0, 0.0], 'k--', linewidth=1.5)
        ax.set_xlabel(r'$\theta$ [deg]', fontsize=50)
        # ax.set_ylabel(r'Ca$_{cl}$ [1]', fontsize=50)
        ax.set_ylabel(r'$U_{cl}$ [nm/ps]', fontsize=50)
        ax.tick_params(axis='both', labelsize=40)
        # ax.xticks(fontsize=40)
        # ax.yticks(fontsize=40)
        ax.legend(fontsize=30)
        ax.set_ylim([-0.035, 0.035])
        ax.set_xlim([12.5, 157.5])
        ax.set_box_aspect(1)
        # ax.title('(a)', y=1.08, fontsize=45, loc='left', fontweight="bold")
        plt.show()

    return expa_opt, mu_th[i_opt]

if __name__ == "__main__" :

    # Fixing the wight, but removing 20% of the data randomly, either from shear or from spreading simulations

    """
    k_steps = 25
    weights_vec = [0.1, 0.01, 0.001]
    """
    k_steps = 1
    weights_vec = [0.01]

    a_array = np.zeros( (len(weights_vec), k_steps) )
    mu_q1_array = np.zeros( (len(weights_vec), k_steps) )
    mu_q2_array = np.zeros( (len(weights_vec), k_steps) )
    mu_q3_array = np.zeros( (len(weights_vec), k_steps) )
    mu_q4_array = np.zeros( (len(weights_vec), k_steps) )

    j = 0
    for w in weights_vec :
        print("w = "+str(w))
        a_vec_k = []
        mu_th_vec_k = []
        for k in range(k_steps) :
            print("k = "+str(k))
            # a, mu = main(w=w,  mode='validation', show_plots=True)
            a, mu = main(w=w,  mode='direct', show_plots=True)
            a_array[j,k] = a
            mu_q1_array[j,k] = mu['Q1']
            mu_q2_array[j,k] = mu['Q2']
            mu_q3_array[j,k] = mu['Q3']
            mu_q4_array[j,k] = mu['Q4']
        print("-----------")
        j += 1

    print("a")
    print("mean : ", np.mean(a_array, axis=1) )
    print("std  : ", np.std(a_array, axis=1) )

    print("mu_q1")
    print("mean : ", np.mean(mu_q1_array, axis=1))
    print("std  : ", np.std(mu_q1_array, axis=1))

    print("mu_q2")
    print("mean : ", np.mean(mu_q2_array, axis=1))
    print("std  : ", np.std(mu_q2_array, axis=1))
    
    print("mu_q3")
    print("mean : ", np.mean(mu_q3_array, axis=1))
    print("std  : ", np.std(mu_q3_array, axis=1))
    
    print("mu_q4")
    print("mean : ", np.mean(mu_q4_array, axis=1))
    print("std  : ", np.std(mu_q4_array, axis=1))