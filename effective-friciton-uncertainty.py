import numpy as np
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt

# Visualization stuff #
nshow = 10
ms = 9
elw = 2.5
# ################### #

"""
mu_f_eff_20p = 104.40575549535964
mu_f_eff_40p = 147.06111089825868
mu_f_eff_60p = 290.16796640194497
mu_f_eff_80p = 1393.0685600514596
"""

muf_w = 3.77
eta_w = 0.69

alpha_g = np.array([0.2, 0.4, 0.6, 0.8])

vol_ratio_w = np.array([5.23, 5.17, 5.07, 4.91])
vol_ratio_g = np.array([1.07, 1.08, 1.08, 1.09])

muf   = np.array([5.43, 9.50, 18.5, 58.8])
muf_std = np.array([0.1,  0.2,  0.4,  1.3])

ag_st     = np.array([0.186, 0.356, 0.502, 0.720])
ag_st_std = np.array([0.022, 0.030, 0.032, 0.037])

eta_eff     = np.array([1.147, 2.135, 4.334, 20.97])
eta_eff_std = np.array([0.080, 0.265, 0.754, 6.702])

Chi     = 1.0
muf_eff = (Chi*vol_ratio_g/ag_st + vol_ratio_w/(1-ag_st))*muf

muf_eff_upp = []
muf_eff_low = []

m1 = (Chi*vol_ratio_g/(ag_st-ag_st_std) + vol_ratio_w/(1-ag_st+ag_st_std))*(muf+muf_std)
m2 = (Chi*vol_ratio_g/(ag_st-ag_st_std) + vol_ratio_w/(1-ag_st+ag_st_std))*(muf-muf_std)
m3 = (Chi*vol_ratio_g/(ag_st+ag_st_std) + vol_ratio_w/(1-ag_st-ag_st_std))*(muf+muf_std)
m4 = (Chi*vol_ratio_g/(ag_st+ag_st_std) + vol_ratio_w/(1-ag_st-ag_st_std))*(muf-muf_std)
for i in range(len(alpha_g)) :
    muf_eff_upp.append( max(m1[i],m2[i],m3[i],m4[i]) )
    muf_eff_low.append( min(m1[i],m2[i],m3[i],m4[i]) )

muf_eff_upp = np.array(muf_eff_upp)
muf_eff_low = np.array(muf_eff_low)

muf_eff_st = muf_eff/muf_w
muf_eff_upp_st = muf_eff_upp/muf_w
muf_eff_low_st = muf_eff_low/muf_w
print(muf_eff_st)
print(muf_eff_upp_st)
print(muf_eff_low_st)
muf_eff_err_st = np.vstack(((muf_eff_st-muf_eff_low_st),(muf_eff_upp_st-muf_eff_st)))

eta_eff_st = eta_eff/eta_w
eta_eff_upp_st = (eta_eff+eta_eff_std)/eta_w
eta_eff_low_st = (eta_eff-eta_eff_std)/eta_w
print(eta_eff_st)
print(eta_eff_upp_st)
print(eta_eff_low_st)
eta_eff_err_st = np.vstack((eta_eff_st-eta_eff_low_st,eta_eff_upp_st-eta_eff_st))


fig = plt.figure()
ax = plt.axes()

plt.plot([eta_eff_st[0], eta_eff_st[3]], [10*eta_eff_st[0], 10*eta_eff_st[3]], 'k--', 
    linewidth=4.5, label=r'$\sim\eta^*$')

plt.errorbar(eta_eff_st,muf_eff_st,yerr=muf_eff_err_st,xerr=eta_eff_err_st,fmt='k.',elinewidth=3.5,capsize=0.5*ms,capthick=3.5)

plt.plot(eta_eff_st[0],muf_eff_st[0],'mH',label=r'$\alpha_g=0.2$',markersize=1.75*ms,markerfacecolor='w',markeredgewidth=1.5*elw)
plt.plot(eta_eff_st[1],muf_eff_st[1],'bo',label=r'$\alpha_g=0.4$',markersize=1.75*ms,markerfacecolor='w',markeredgewidth=1.5*elw)
plt.plot(eta_eff_st[2],muf_eff_st[2],'rD',label=r'$\alpha_g=0.6$',markersize=1.5*ms,markerfacecolor='w',markeredgewidth=1.5*elw)
plt.plot(eta_eff_st[3],muf_eff_st[3],'gs',label=r'$\alpha_g=0.8$',markersize=1.5*ms,markerfacecolor='w',markeredgewidth=1.5*elw)

print(eta_eff_st[0])
print(muf_eff_st[0])

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xticks([2, 10])
ax.set_yticks([20, 100])

class MathTextSciFormatter(mticker.Formatter):
    def __init__(self, fmt="%1.2e"):
        self.fmt = fmt
    def __call__(self, x, pos=None):
        s = self.fmt % x
        decimal_point = '.'
        positive_sign = '+'
        tup = s.split('e')
        significand = tup[0].rstrip(decimal_point)
        sign = tup[1][0].replace(positive_sign, '')
        # exponent = tup[1][1:].lstrip('0')
        exponent = tup[1][2:]
        if exponent:
            exponent = '10^{%s%s}' % (sign, exponent)
        if significand and exponent:
            s =  r'%s{\times}%s' % (significand, exponent)
        else:
            s =  r'%s%s' % (significand, exponent)
        return "${}$".format(s)

ax.get_xaxis().set_major_formatter(MathTextSciFormatter('%1.e'))
ax.get_yaxis().set_major_formatter(MathTextSciFormatter('%1.e'))

plt.tick_params(axis='both', labelsize=35)
plt.xlabel(r'$\eta^*/\eta_w$', fontsize=40)
plt.ylabel(r'$\mu_f^*/\mu_{f,w}$', fontsize=40)
plt.legend(fontsize=30)

plt.subplots_adjust(left=0.1, right=1.1)
ax.set_box_aspect(1)

plt.show()