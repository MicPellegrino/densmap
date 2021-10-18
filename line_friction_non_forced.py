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

"""
muf0 = 7.106528285687076
beta = 4.45979718788757
muth = 2.6810468850069893
expa = 2.0189965532077703
"""
"""
muf0 = 7.826477525728069
beta = 3.887523096076031
muth = 3.472570598923394
expa = 0.7013225308250497
"""
"""
muf0 = 5.022451046269512
beta = 3.1945412208177566
muth = 1.7798311811670215
expa = 1.0226149350876883
"""
"""
muf0 = 5.490595931179681
beta = 4.001367517106868
muth = 3.403975947252455
expa = 0.6356135670182949
"""
muf0 = 5.587731071215932
beta = 4.570032390160676
muth = 3.403975947252455
expa = 0.6356135670182949

sin = lambda t : np.sin(np.deg2rad(t))
cos = lambda t : np.cos(np.deg2rad(t))
mu_st_fun = lambda xi : muf0 / (1.0 + beta*xi**2 )
mu_th_fun = lambda t :  muth * np.exp(expa*(0.5*sin(t)+cos(t))**2)

# Theta0 = 68.8deg
# avg_theta_0 = 71.65334344790341
avg_theta_0 = 72.29215275895473
folders = [ 'ShearDynamic/Q3_Ca005',
            'ShearDynamic/Q3_Ca006',
            'ShearDynamic/Q3_Ca008',
            'ShearDynamic/Q3_Ca010' ]
capillary_number = 0.5*np.array([ 0.05, 0.06, 0.08, 0.10])
# Init averaging
t_0 = 10000
adv_collect = []
rec_collect = []
avg_angle_adv = []
avg_angle_rec = []
std_angle_adv = []
std_angle_rec = []
std_cos_adv = []
std_cos_rec = []
n_decorr = 80
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
    std_cos_adv.append( np.std(cos(avg_theta_0)-cos(adv))/np.sqrt(len(tl)/n_decorr) )
    std_cos_rec.append( np.std(cos(avg_theta_0)-cos(rec))/np.sqrt(len(tl)/n_decorr) )
avg_angle_adv = np.array( avg_angle_adv )
avg_angle_rec = np.array( avg_angle_rec )
std_angle_adv = np.array( std_angle_adv )
std_angle_rec = np.array( std_angle_rec )
std_cos_adv = np.array( std_cos_adv )
std_cos_rec = np.array( std_cos_rec )

# Cosine difference
capillary_number = np.concatenate( (-capillary_number, capillary_number), axis=None )
delta_cosine = np.concatenate( (avg_angle_rec, avg_angle_adv), axis=None )
angle_absolute = delta_cosine
cos_fun_vec = lambda t : np.cos( np.deg2rad(t) )
cos_fun_vec = np.vectorize( cos_fun_vec )
delta_cosine = cos_fun_vec( avg_theta_0*np.ones(delta_cosine.shape) ) - cos_fun_vec( delta_cosine )

# Line friction from non-linear MKT
mu_st_fun_vec = np.vectorize(mu_st_fun)
mu_f_shear = mu_st_fun_vec(delta_cosine)

# Line friction from P&B
mu_th_fun_vec = np.vectorize(mu_th_fun)
mu_f_therm = mu_th_fun_vec(angle_absolute)

# Consistency check
capillary_from_mkt = delta_cosine / mu_f_shear
delta_from_mkt = capillary_from_mkt-(delta_cosine+np.concatenate((std_cos_rec,std_cos_adv),axis=None)) / mu_f_shear
capillary_from_thr = delta_cosine / mu_f_therm
delta_from_thr = capillary_from_thr-(delta_cosine+np.concatenate((std_cos_rec,std_cos_adv),axis=None)) / mu_f_therm

print("Ca (original) = "+str(capillary_number))
print("Ca (estimate) = "+str(capillary_from_mkt))

######### Line friction plot (MKT) #########
xi_range = np.linspace(-2.0, 2.0, 500)
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_title('Nonlinear MKT', fontsize=30.0)
ax1.plot(xi_range, mu_st_fun(xi_range), 'k-', linewidth=2.75)
s_red = True
s_blue = True
for dc in delta_cosine :
    if dc > 0 :
        if s_red :
            ax1.plot([dc, dc], [0.0, mu_st_fun(dc)], 'r--', linewidth=2.75, label='advancing')
            s_red = False
        else :
            ax1.plot([dc, dc], [0.0, mu_st_fun(dc)], 'r--', linewidth=2.75)
        ax1.plot(dc, mu_st_fun(dc), 'rx', markeredgewidth=2.0, markersize=12.5)
    if dc <= 0 :
        if s_blue :
            ax1.plot([dc, dc], [0.0, mu_st_fun(dc)], 'b--', linewidth=2.75, label='receding')
            s_blue = False
        else :
            ax1.plot([dc, dc], [0.0, mu_st_fun(dc)], 'b--', linewidth=2.75)
        ax1.plot(dc, mu_st_fun(dc), 'bo', markersize=8.5)
ax1.set_xlabel(r'$\delta\cos$ [1]', fontsize=30.0)
ax1.set_ylabel(r'$\mu_f^*$ [1]', fontsize=30.0)
ax1.legend(fontsize=20.0)
ax1.tick_params(axis='x', labelsize=plot_tcksize)
ax1.tick_params(axis='y', labelsize=plot_tcksize)
ax1.set_xlim([-2.0, 2.0])
ax1.set_ylim([0.0, 1.25*muf0])
# plt.show()
############################################

######### Line friction plot (P&B) #########
t_range = np.linspace(30.0, 110.0, 500)
# fig, ax = plt.subplots()
ax2.set_title('Johansson & Hess 2018', fontsize=30.0)
ax2.plot(t_range, mu_th_fun(t_range), 'k-', linewidth=2.75, label=r'$Johansson&Hess2018$')
s_red = True
s_blue = True
for theta in angle_absolute :
    if theta > avg_theta_0 :
        if s_red :
            ax2.plot([theta, theta], [0.0, mu_th_fun(theta)], 'r--', linewidth=2.75, label='advancing')
            s_red = False
        else :
            ax2.plot([theta, theta], [0.0, mu_th_fun(theta)], 'r--', linewidth=2.75)
        ax2.plot(theta, mu_th_fun(theta), 'rx', markeredgewidth=2.0, markersize=12.5)
    else :
        if s_blue :
            ax2.plot([theta, theta], [0.0, mu_th_fun(theta)], 'b--', linewidth=2.75, label='receding')
            s_blue = False
        else :
            ax2.plot([theta, theta], [0.0, mu_th_fun(theta)], 'b--', linewidth=2.75)
        ax2.plot(theta, mu_th_fun(theta), 'bo', markersize=8.5)
ax2.set_xlabel(r'$\theta$ [deg]', fontsize=30.0)
# ax2.set_ylabel(r'$\mu_f^*$ [-1]', fontsize=30.0)
# ax2.legend(fontsize=20.0)
ax2.tick_params(axis='x', labelsize=plot_tcksize)
ax2.tick_params(axis='y', labelsize=plot_tcksize)
# ax2.set_xlim([-2.0, 2.0])
ax2.set_ylim([0.0, 12.5])
plt.show()
############################################

####### Capillary number plot ########
xi_range = np.linspace(-2.0, 2.0, 500)
fig, ax = plt.subplots()
ax.set_title('Predicted capillary number', fontsize=35.0)
ax.plot(delta_cosine, capillary_number, 'gs', markersize=12.5, \
        markeredgewidth=2, markeredgecolor='black', label='imposed')
# ax.plot(delta_cosine, capillary_from_mkt, 'mD', markersize=12.5, label='estimate from MKT')
# ax.plot(delta_cosine, capillary_from_thr, 'cH', markersize=15.0, label='estimate from J&H')
eb1=ax.errorbar(delta_cosine, capillary_from_mkt, yerr=delta_from_mkt, fmt='mD', \
        markersize=12.5, capthick=2.5, capsize=5, linewidth=2.5, markerfacecolor="None", \
        markeredgewidth=3, label='estimate from MKT')
eb1[-1][0].set_linestyle('--')
eb2=ax.errorbar(delta_cosine, capillary_from_thr, yerr=delta_from_thr, fmt='cH', \
        markersize=15.0, capthick=2.5, capsize=5, linewidth=2.5, markerfacecolor="None", \
        markeredgewidth=3, label='estimate from J&H')
eb2[-1][0].set_linestyle(':')
ax.set_xlabel(r'$\delta\cos$ [1]', fontsize=32.5)
ax.set_ylabel(r'$\widehat{Ca}$ [1]', fontsize=32.5)
ax.legend(fontsize=30.0)
ax.tick_params(axis='x', labelsize=plot_tcksize)
ax.tick_params(axis='y', labelsize=plot_tcksize)
plt.show()
######################################
