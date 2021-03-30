import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy.ndimage as smg
import scipy.signal as sgn
import scipy.optimize as sc_opt

cos = lambda t : np.cos( np.deg2rad(t) )
sin = lambda t : np.sin( np.deg2rad(t) )
tan = lambda t : np.tan( np.deg2rad(t) )
cot = lambda t : 1.0/np.tan( np.deg2rad(t) )
tan_m1 = lambda t : np.rad2deg(np.arctan(t))
cos_m1 = lambda t : np.rad2deg(np.arccos(t))
sin_m1 = lambda t : np.rad2deg(np.arcsin(t))

def array_from_file( filename ):
    my_list = []
    with open(filename, 'r') as f:
        for line in f:
            my_list.append(float(line.split()[0]))
    return np.array(my_list)

# Initial droplet radius [nm]
R0 = 25.0
print("Initiad droplet radius = "+str(R0)+" [nm]")
# Corrugation spacing [1/nm]
k = 0.0
print("Corrugation spacing    = "+str(k)+" [1/nm]")
# Displacement prefactor [nondim]
beta = R0*k
print("Displacement prefactor = "+str(beta)+" [nondim]")
# Bulk viscosity [mPa*s]
mu = 0.887
print("Bulk viscosity         = "+str(mu)+" [mPa*s]")
# Surface tension [mPa*m]
gamma = 57.8
print("Surface tension        = "+str(gamma)+" [mPa*m]")
# Reference speed [nm/ps]
U_ref = (gamma/mu)*(1e-3)
print("Reference speed        = "+str(U_ref)+" [nm/ps]")
# Reference time [ps]
tau = R0/U_ref
print("Reference time         = "+str(tau)+" [ps]")
# Contact angles [deg]
theta_g_0 = 90.0
print("Initial c. a           = "+str(theta_g_0)+" [deg]")
theta_e = 38.8
print("Equilibrium c. a.      = "+str(theta_e)+" [deg]")
# Corrugation height [nm]
h = 0.0
print("Corrugation height     = "+str(h)+" [nm]")
# Roughness coefficient 'a' [nondim]
a = h*k
print("Roughness coefficient  = "+str(a)+" [nondim]")
# Initial reduced droplet area [nondim]
A0 = np.pi
fun_theta = lambda t : ( np.deg2rad(t)/(sin(t)**2) - cot(t) )
# Initial wetted distance [nondim]
x0 = np.sqrt( np.pi/fun_theta(theta_g_0) )
print("Initial c.l. distance  = "+str(x0*R0)+" [nm]")

### MD DATA ###
# Obtaining the signal from saved .txt files
folder_name = 'SpreadingData/FlatQ4'
time = array_from_file(folder_name+'/time.txt')
foot_l = array_from_file(folder_name+'/foot_l.txt')
foot_r = array_from_file(folder_name+'/foot_r.txt')
angle_l = array_from_file(folder_name+'/angle_l.txt')
angle_r = array_from_file(folder_name+'/angle_r.txt')
radius = array_from_file(folder_name+'/radius_fit.txt')
# Cutoff inertial phase
t_bin = time[1]-time[0]
t_90deg = 2280.0                  # [ps]
time_window = time[-1]-t_90deg
print("Time window            = "+str(time_window)+" [ps]")
Nb = int( t_90deg / t_bin )
time = time[Nb:]
foot_l = foot_l[Nb:]
foot_r = foot_r[Nb:]
angle_l = angle_l[Nb:]
angle_r = angle_r[Nb:]
radius = radius[Nb:]
center = 0.5*(foot_r+foot_l)
# branch_right = foot_r - center
branch_right = 0.5*radius

### SIMULATION ###
# Macroscopic angle (given by circular cap)
theta_g = lambda x : sc_opt.fsolve(lambda t : ((x**2)*fun_theta(t)-np.pi), theta_e)[0]
# Microscopic angle
phi = lambda x : theta_g(x) + tan_m1(a*cos(beta*x))
# Curvilinear coordinates measure
dsdx = lambda x : np.sqrt( 1.0 + a*a*(cos(beta*x)**2) )
# Potential (MKT or PF)
# V = lambda x : ( cos( theta_e ) - cos( phi(x) ) ) / dsdx(x)
f = lambda x : sin( phi(x) )
V = lambda x : ( cos( theta_e ) - cos( phi(x) ) ) / ( dsdx(x)*f(x) )
# Numerical integration
# Final time [ps]
Nt = int(10*len(time))
dt = 0.1*t_bin/tau

### LINE FRICTION ###
# Tolerance on the reduced error
toll = 0.0001
Kmax = 12
rel_err = 1+toll
mu_star_a = 10.5
rel_err_a = 0.00096282223226555
mu_star_b = 12.5
rel_err_b = 0.00128886989252967

k=0
while k<Kmax and rel_err>toll :
    k+=1
    mu_star = 0.5 * ( mu_star_a + mu_star_b )
    print("Iter "+str(k))
    print("Friction ratio mu_f/mu = "+str(mu_star)+" [nm]")
    # Simulation
    x_vec = []
    x = x0
    x_vec.append(x)
    for n in range(1,Nt) :
        x = x + V(x)*dt/mu_star
        x_vec.append(x)
    x_vec = np.array(x_vec)
    # Check differences
    abs_err = np.sqrt(np.sum((R0*x_vec[0::10]-branch_right)**2))/len(branch_right)
    rel_err = abs_err/R0
    if rel_err_b < rel_err_a :
        rel_err_a = rel_err
        mu_star_a = mu_star
    else :
        rel_err_b = rel_err
        mu_star_b = mu_star

print("Relative error         = "+str(rel_err)+" [nondim]")
t_vec = np.linspace(0.0, n*dt, n+1)
pnt_err = np.abs(R0*x_vec[0::10]-branch_right)

# Plot spreading radius
plt.title(r'Spreading branch, $\mu^*=$'+str(mu_star), fontsize=30.0)
plt.plot(time, branch_right, 'r-', linewidth=1.5, label='MD')
plt.plot(tau*t_vec+t_90deg, R0*x_vec, 'k--', linewidth=2.0, label='Simulated')
# plt.plot(time, pnt_err, 'g-', linewidth=1.0)
plt.xlabel('t [ps]', fontsize=30.0)
plt.ylabel('x [nm]', fontsize=30.0)
plt.xlim([time[0], time[-1]])
plt.legend(fontsize=20.0)
plt.xticks(fontsize=20.0)
plt.yticks(fontsize=20.0)
plt.show()
