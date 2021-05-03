import numpy as np
import matplotlib.pyplot as plt

def array_from_file( filename ):
    my_list = []
    with open(filename, 'r') as f:
        for line in f:
            my_list.append(float(line.split()[0]))
    return np.array(my_list)

# Plot parameters
plot_sampling = 15
plot_tcksize = 25

# From droplet spreading simulations:
muf0 = 10.19884996319109
beta = 2.753401248620512
mu_st_fun = lambda xi : muf0 / (1.0 + beta*xi**2 )

# Theta0 = 68.8deg
avg_theta_0 = 68.8
folders = [ 'ShearDynamic/Q3_Ca005', 
            'ShearDynamic/Q3_Ca008',
            'ShearDynamic/Q3_Ca010' ]
capillary_number = np.array([ 0.05, 0.08, 0.10])
# Init averaging
t_0 = 5000
adv_collect = []
rec_collect = []
avg_angle_adv = []
std_angle_adv = []
avg_angle_rec = []
std_angle_rec = []
for fn in folders :
    time = array_from_file(fn+'/time.txt')
    idx_0 = np.abs( time-t_0 ).argmin()
    tl = array_from_file(fn+'/angle_tl.txt')[idx_0:]
    br = array_from_file(fn+'/angle_br.txt')[idx_0:]
    tr = array_from_file(fn+'/angle_tr.txt')[idx_0:]
    bl = array_from_file(fn+'/angle_bl.txt')[idx_0:]
    adv = 0.5 * ( tl + br )
    adv_collect.append( adv )
    rec = 0.5 * ( tr + bl )
    rec_collect.append( rec )
    avg_angle_adv.append( np.mean( adv ) )
    std_angle_adv.append( np.std( adv ) )
    avg_angle_rec.append( np.mean( rec ) )
    std_angle_rec.append( np.std( rec ) )
avg_angle_adv = np.array( avg_angle_adv )
avg_angle_rec = np.array( avg_angle_rec )
std_angle_adv = np.array( std_angle_adv )
std_angle_rec = np.array( std_angle_rec )

# Cosine difference
capillary_number = np.concatenate( (-capillary_number, capillary_number), axis=None )
delta_cosine = np.concatenate( (avg_angle_rec, avg_angle_adv), axis=None )
cos_fun_vec = lambda t : np.cos( np.deg2rad(t) )
cos_fun_vec = np.vectorize( cos_fun_vec )
delta_cosine = cos_fun_vec( avg_theta_0*np.ones(delta_cosine.shape) ) - cos_fun_vec( delta_cosine )

# Line friction from non-linear MKT
mu_st_fun_vec = np.vectorize(mu_st_fun)
mu_f_shear = mu_st_fun_vec(delta_cosine)

# Consistency check
capillary_from_mkt = delta_cosine / mu_f_shear 

print("Ca (original) = "+str(capillary_number))
print("Ca (estimate) = "+str(capillary_from_mkt))

######### Line friction plot #########
xi_range = np.linspace(-2.0, 2.0, 500)
fig, ax = plt.subplots()
ax.set_title('Angle-dependent contact line friction', fontsize=30.0)
ax.plot(xi_range, mu_st_fun(xi_range), 'k-', linewidth=2.75, label=r'$\hat{\mu}_f^*\;/\;[1+\beta(\delta\cos)^2]$')
for dc in delta_cosine :
    if dc > 0 :
        ax.plot([dc, dc], [0.0, mu_st_fun(dc)], 'r--', linewidth=2.75)
        ax.plot(dc, mu_st_fun(dc), 'rx', markeredgewidth=2.0, markersize=12.5)
    if dc <= 0 :
        ax.plot([dc, dc], [0.0, mu_st_fun(dc)], 'b--', linewidth=2.75)
        ax.plot(dc, mu_st_fun(dc), 'bo', markersize=8.5)
ax.set_xlabel(r'$\delta\cos$ [-1]', fontsize=30.0)
ax.set_ylabel(r'$\mu_f^*$ [-1]', fontsize=30.0)
ax.legend(fontsize=20.0)
ax.tick_params(axis='x', labelsize=plot_tcksize)
ax.tick_params(axis='y', labelsize=plot_tcksize)
ax.set_xlim([-2.0, 2.0])
ax.set_ylim([0.0, 1.25*muf0])
plt.show()
######################################
