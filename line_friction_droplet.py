import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as smg
import scipy.signal as sgn
import scipy.optimize as opt

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

# Lambdas don't have a specific __name__
# mkt_3       = lambda x, a1, a3 : a1*x + a3*(x**3)
# therm_fun   = lambda t, b0, b1 : b0 * np.exp(-b1*(0.5*sin(t)+cos(t))**2) * (cos(theta_0)-cos(t))

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
equilibrium_contact_angle['Q1'] = 126.01
equilibrium_contact_angle['Q2'] = 94.9
equilibrium_contact_angle['Q3'] = 70.5
equilibrium_contact_angle['Q4'] = 39.2

substrate_color = dict()
substrate_color['Q1'] = 'c'
substrate_color['Q2'] = 'g'
substrate_color['Q3'] = 'r'
substrate_color['Q4'] = 'm'

# Shear droplet data

theta = dict()
ca = dict()
U = dict()

theta['Q1'] = np.array([130.88, 131.49, 137.93, 142.13, 129.7410333158549, 126.65, 126.06, 124.32, 125.42])
ca['Q1'] = 0.5*np.array([0.15, 0.30, 0.60, 0.90, 0.0, -0.15, -0.30, -0.60, -0.90])

theta['Q2'] = np.array([101.21042472, 104.01678041, 106.17731627, 108.88678425, 111.04525937, 97.7792911518621, 94.95432237, 92.13575175, 88.46459205, 84.38834944, 79.88718327])
ca['Q2'] = 0.5*np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.0, -0.05, -0.10, -0.15, -0.20, -0.25])

# theta['Q3'] = np.array([77.18496895, 77.85253598, 78.84062248, 81.5119225, 84.13935051, 72.34125045506745, 65.86140487, 63.75901713, 61.81662731, 58.91360746, 54.28781229])
theta['Q3'] = np.array([78.32, 81.25, 84.77, 86.10, 87.19, 70.5, 67.04, 64.84, 62.39, 55.35, 50.16])
ca['Q3'] = np.array([0.015,  0.025,  0.03 ,  0.04 ,  0.05 , 0.0, -0.015 ,-0.025 ,-0.03 , -0.04 , -0.05])

theta['Q4'] = np.array([42.51470603, 45.30519045, 45.4518236, 39.093364496646906, 37.05776924, 33.47130752, 29.93002039])
ca['Q4'] = 0.5*np.array([0.010, 0.015, 0.020, 0.0, -0.010, -0.015, -0.020])

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
            print("Reading files in folder: ", k)
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
            # Throw away a few data ...
            N_eq = int(frac_steady*len(CL.time[N_avg:-N_avg]))
            CL.estimate_theta0(N_eq)
            
            print("<theta0> = ", CL.theta0)
            
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

    def fit_model(self) :
        if self.fit_cos :
            self.popt, self.pcov = opt.curve_fit(self.model_function, self.reduced_cosine_micro, self.reduced_velocity_micro)
        else :
            self.popt, self.pcov = opt.curve_fit(self.model_function, self.angle_micro, self.reduced_velocity_micro)
        print("Fitting contact line model: "+self.model_function.__name__)
        print(self.popt)
        # print(self.pcov)
        return self.popt

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


def main() :

    clm = dict()
    N_sample_plot = plot_sampling
    min_cos = 2
    max_cos = -2
    min_vel = 100
    max_vel = -100
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
        sparse_y = clm[k].reduced_velocity_micro[::N_sample_plot]*U_ref
        """
        plt.plot(sparse_x, sparse_y, 'o', \
            label=k+r" - $\theta_0 = $"+"{:.2f}".format(clm[k].avg_theta0)+" deg", \
            markersize=17.5, markerfacecolor='None', markeredgewidth=2.5)
        """
        # plt.plot(sparse_x, sparse_y, '.', \
        #     markersize=20, markerfacecolor='None', markeredgewidth=1.5, alpha=0.6, color=substrate_color[k])
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
        """
        plt.plot(theta[k], ca[k], 'D', \
            label=t0lab, markersize=17.5, markeredgecolor='k', markeredgewidth=4, color=substrate_color[k])
        """
        plt.plot(theta[k], U[k], 'D', \
            label=t0lab, markersize=17.5, markeredgecolor='k', markeredgewidth=4, color=substrate_color[k])
        # Models for contact line friction
        def therm_fun(t, b0, b1) :
            return b0 * np.exp(-b1*(0.5*sin(t)+cos(t))**2) * (cos(clm[k].avg_theta0)-cos(t))
        def mkt_fun(t, a1, a3) :
            return a1*t + a3*(t**3)
        clm[k].model_function = therm_fun
        p_thermo = clm[k].fit_model()
        clm[k].fit_cos = True
        clm[k].model_function = mkt_fun
        p_mkt = clm[k].fit_model()
        mu_st = 1.0/p_mkt[0]
        mu_th = 1.0/p_thermo[0]
        beta = p_mkt[1] * mu_st
        print("mu* (mkt)   = "+str( mu_st ))
        print("mu* (therm) = "+str( mu_th ))
        print("beta = "+str(beta))
        print("expa = "+str(p_thermo[1]))
        print("---------------------------------------------")
        # clm.plot_models()
    plt.plot([20, 150], [0.0, 0.0], 'k--', linewidth=3.0)
    # plt.plot([0.0, 0.0], [min_vel, max_vel], 'k--', linewidth=2.0)
    plt.xlabel(r'$\theta$ [deg]', fontsize=50)
    # plt.ylabel('Ca [1]', fontsize=50)
    plt.ylabel(r'$U$ [nm/ps]', fontsize=50)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.legend(fontsize=40)
    plt.ylim([-0.035, 0.035])
    # plt.text(70, -0.3, 'receding', fontsize=45, weight='bold')
    # plt.text(40, 0.3, 'advancing', fontsize=45, weight='bold')
    plt.show()

if __name__ == "__main__" :
    main()