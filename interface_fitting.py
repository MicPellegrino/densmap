import densmap as dm
import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt

def read_interface_xvg(file_name) :
    
    x = []
    z = []
    
    file_in = open(file_name, 'r')
    
    for line in file_in :
        cols = line.split()
        if cols[0] != "#" and cols[0] != "@" :
            x.append(float(cols[0]))
            z.append(float(cols[1]))
    file_in.close()

    return x, z

Lz = 30.63400
idx_ini = 1
idx_fin = 501

x_r, z_r = read_interface_xvg("InterfaceTest/int_r_"+str(idx_ini).zfill(5)+".xvg")
x_r = np.array(x_r)
z_r = np.array(z_r)
x_avg = np.zeros(x_r.shape, dtype=float)
z_avg = np.zeros(z_r.shape, dtype=float)
x_avg2 = np.zeros(x_r.shape, dtype=float)

for idx in range(idx_ini,idx_fin) :

    x_r, z_r = read_interface_xvg("InterfaceTest/int_r_"+str(idx).zfill(5)+".xvg")
    x_r = np.array(x_r)
    z_r = np.array(z_r)
    x_r = np.mean(x_r)-x_r
    z_r = Lz-z_r

    x_l, z_l = read_interface_xvg("InterfaceTest/int_l_"+str(idx).zfill(5)+".xvg")
    x_l = np.array(x_l)
    z_l = np.array(z_l)
    x_l = x_l-np.mean(x_l)

    x_avg += 0.5*(x_r+np.flip(x_l))
    z_avg += 0.5*(z_r+np.flip(z_l))

    x_avg2 += (0.5*(x_r+np.flip(x_l)))**2

x_avg /= (idx_fin-idx_ini)
z_avg /= (idx_fin-idx_ini)

x_avg2 /= (idx_fin-idx_ini) 

std_x = np.sqrt( x_avg2 - x_avg**2 )

N = len(z_avg)
indices = np.array(range(N))
quota = 0.60
tests = 2000
q_max = int(0.21*quota*N)

err_avg = np.zeros(q_max-1)
q_vec = np.array(range(1,q_max))

for k in range(tests) :
   
    if k%100==0 :
        print("k = "+str(k))
    
    rng.shuffle(indices)
    ind_train = indices[:int(quota*N)]
    ind_valid = indices[int(quota*N):]
    
    """
    ind_train = indices[int(0.5*(1-quota)*N):N-int(0.5*(1-quota)*N)]
    ind_valid = np.concatenate((indices[:int(0.5*(1-quota)*N)],indices[N-int(0.5*(1-quota)*N):]))
    """

    err = []
    for q in range(1,q_max) :
        p = np.polyfit(z_avg[ind_train], x_avg[ind_train], q)
        x_hat = np.polyval(p, z_avg)
        err.append(np.sqrt(np.sum((x_hat[ind_valid]-x_avg[ind_valid])**2)))
    err_avg += np.array(err)

err_avg /= tests

plt.plot(x_avg,z_avg,'g-')
plt.plot(x_avg-std_x,z_avg,'g--')
plt.plot(x_avg+std_x,z_avg,'g--')
plt.plot(x_avg[ind_valid],z_avg[ind_valid],'rx')
plt.show()

plt.plot(q_vec, err_avg, 'rx--')
plt.show()

plt.plot(std_x,z_avg,'g-')
plt.show()
