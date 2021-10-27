import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as smg
import scipy.signal as sgn
import scipy.optimize as opt

# Plotting params
plot_sampling = 30
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
# Use datafrrames instead?
# Q1
"""
input_files = { 'SpreadingData/FlatQ1REC' : ('rec', True),
                'SpreadingData/FlatQ1REC2' : ('rec', False)}
"""
# Q2
"""
input_files = { 'SpreadingData/FlatQ2ADV' : ('adv', True), 
                'SpreadingData/FlatQ2REC' : ('rec', True)}
"""
# Q3
"""
input_files = { 'SpreadingData/FlatQ3ADV' : ('adv', True), 
                'SpreadingData/FlatQ3REC' : ('rec', True), 
                'SpreadingData/FlatQ3CAP' : ('adv', False), 
                'SpreadingData/FlatQ3REC2': ('rec', False)}
"""
# Q4
input_files = { 'SpreadingData/FlatQ4ADV' : ('adv', True), 
                'SpreadingData/FlatQ4REC' : ('rec', True) }


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
        model_function=None, fit_cos=False, vmin=1e-4, delta_t_avg=880, T_off=500, dt=10.0, frac_steady=0.667) :

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

        self.compute_reduced_quantities(N_avg)

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

    def compute_reduced_quantities(self, N_avg) :
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
        print(self.pcov)

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

    clm = ContactLineModel(input_files, vmin=1e-4, delta_t_avg=880)
    
    print(clm)
    clm.plot_data()

    # Models for contact line friction
    theta_0 = clm.avg_theta0
    def mkt_3(x, a1, a3) :
        return a1*x + a3*(x**3)
    def therm_fun(t, b0, b1) :
        return b0 * np.exp(-b1*(0.5*sin(t)+cos(t))**2) * (cos(theta_0)-cos(t))
    
    clm.model_function = therm_fun

    clm.fit_model()
    clm.plot_models()

if __name__ == "__main__" :
    main()