import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits import mplot3d
import astropy.units as u
from datetime import datetime
from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, get_body_barycentric_posvel, ICRS
import pickle
from nbody.engine import run       

day = 86400. #*u.second
year = day*365

G = 6.67e-11 #*(u.meter**3)/(u.kilogram*u.second**2) # 6.67e-11 #

C = 299792458. #*(u.meter/u.second) #299792458. 
Ms = 1.988e30 #*(u.kilogram) # 1.988e30 

Mmerc = 0.3301e24
Mearth = 5.9722e24 
AU = 149597870700. #*u.meter
Ms = 1.988e30

#parameters for solution files management     
plot_step = 100000
buffer_lenght = 1000000
data_thin = 10

PN_order = 0
ICN_it = 2 

#nbodies = 6
dt = 0.01
dt2 = 0.5*dt
N  = 50000000
p = 0

Neff = int(N/(data_thin*plot_step)) #number of final datas in the plots
nout = int(N/buffer_lenght) #number of files generated   

#---------------------------------------#
#actual natural initial coordinates 
t0 = Time('2000-01-01T0:0:00.0', scale='tdb')
t = t0

masses = {
'sun'     : Ms, #1st planet has to be the central attractor
'mercury' : Mmerc, #2nd planet has to be the one which we want to test the GR dynamics effects on 
'venus'   : 0.815*Mearth,
'earth'   : Mearth,
'mars'    : 0.1075*Mearth,
'jupiter' : 317.8*Mearth,
#'saturn'  : 95.2*Mearth,
#'uranus'  : 14.6*Mearth,
#'neptune' : 17.2*Mearth,
#'pluto'   : 0.00218*Mearth,
}

planet_names = [
'sun',
'mercury',
'venus',
'earth',
'mars',
'jupiter',
#'saturn',
#'uranus',
#'neptune',
#'pluto' 
]

planets = []
planets_0 = []

m = np.array([masses[planet] for planet in planet_names]).astype(np.longdouble)   
Mtot = np.sum(m)
    
#3D initial conditions
x = np.array([[0 for i in range(0, len(m))] for k in range(0, Neff)]).astype(np.longdouble)
y = np.array([[0 for i in range(0, len(m))] for k in range(0, Neff)]).astype(np.longdouble)
z = np.array([[0 for i in range(0, len(m))] for k in range(0, Neff)]).astype(np.longdouble)

vx = np.array([[0 for i in range(0, len(m))] for k in range(0, Neff)]).astype(np.longdouble)
vy = np.array([[0 for i in range(0, len(m))] for k in range(0, Neff)]).astype(np.longdouble)
vz = np.array([[0 for i in range(0, len(m))] for k in range(0, Neff)]).astype(np.longdouble)

#vcm = np.array([[0 for i in range(0, len(masses))] for Neff in range(0, N)]).astype(np.longdouble)

sx = np.array([[0 for i in range(0, len(m))] for k in range(0, Neff)]).astype(np.longdouble)
sy = np.array([[0 for i in range(0, len(m))] for k in range(0, Neff)]).astype(np.longdouble)
sz = np.array([[0 for i in range(0, len(m))] for k in range(0, Neff)]).astype(np.longdouble)

#find high resolution initial points 

planet_periods = np.array([0., 87.969, 224.701, 365.256,  686.980, 4332.589, 10759.22]) 
planet_0 = []
j_per = 1

x_0 = np.array([0 for i in range(0, len(m))]).astype(np.longdouble)
y_0 = np.array([0 for i in range(0, len(m))]).astype(np.longdouble)
z_0 = np.array([0 for i in range(0, len(m))]).astype(np.longdouble)

vx_0 = np.array([0 for i in range(0, len(m))]).astype(np.longdouble)
vy_0 = np.array([0 for i in range(0, len(m))]).astype(np.longdouble)
vz_0 = np.array([0 for i in range(0, len(m))]).astype(np.longdouble)

      
for k in range(len(m)):
    for j in range (j_per):
        planets_0 = (get_body_barycentric_posvel(planet_names[k], t0 - planet_periods[k]*j*u.day, ephemeris= 'DE440'))
         
        x_0[k] += planets_0[0].x.to(u.meter).value 
        y_0[k] += planets_0[0].y.to(u.meter).value 
        z_0[k] += planets_0[0].z.to(u.meter).value 

        vx_0[k] += planets_0[1].x.to(u.meter/u.second).value 
        vy_0[k] += planets_0[1].y.to(u.meter/u.second).value 
        vz_0[k] += planets_0[1].z.to(u.meter/u.second).value

x[0][:] = x_0[:]/j_per
y[0][:] = y_0[:]/j_per
z[0][:] = z_0[:]/j_per
        
vx[0][:] = vx_0[:]/j_per
vy[0][:] = vy_0[:]/j_per
vz[0][:] = vz_0[:]/j_per       

#print(len(m))
#run the simulation
if (p == 0):
    D, t_sim = run(N, np.longdouble(dt), PN_order, m, x[0][:], y[0][:], z[0][:], m*vx[0][:], m*vy[0][:], m*vz[0][:], sx[0][:], sy[0][:], sz[0][:], ICN_it, data_thin, buffer_lenght, plot_step)
    #run(N, np.longdouble(dt), PN_order, m, x_0[:], y_0[:], z_0[:], m*vx_0[:], m*vy_0[:], m*vz_0[:], sx[0][:], sy[0][:], sz[0][:], ICN_it, data_thin, buffer_lenght)


s, H, T, V = [], [], [], []

for i in range(nout):  
    s_tot, H_tot, T_tot, V_tot = [], [], [], []
    

    s_tot.append(pickle.load(open('solution_{}_order{}.pkl'.format(i, PN_order),'rb')))
    H_tot.append(pickle.load(open('hamiltonian_{}_order{}.pkl'.format(i, PN_order),'rb')))
    T_tot.append(pickle.load(open('kinetic_{}_order{}.pkl'.format(i, PN_order),'rb')))
    V_tot.append(pickle.load(open('potential_{}_order{}.pkl'.format(i, PN_order),'rb')))        

    s.append(s_tot[0][::plot_step])
    H.append(H_tot[0][::plot_step])
    T.append(T_tot[0][::plot_step])
    V.append(V_tot[0][::plot_step])          

    index = i*100/nout 
    if (index) % 10 == 0 :
        print("Sim. deframmentation: {}%".format(index))
    
    del s_tot
    del H_tot
    del T_tot
    del V_tot   

s = np.array(s, dtype=object)
H = np.array(H, dtype=object)
T = np.array(T, dtype=object)
V = np.array(V, dtype=object)

s = np.concatenate((s[:]), axis=0)
H = np.concatenate((H[:]), axis=0)
T = np.concatenate((T[:]), axis=0)
V = np.concatenate((V[:]), axis=0) 






#collect SS data
for i in range(1, Neff):

    t = t0 + t_sim[i]*u.second

    for planet in planet_names:

        planets.append(get_body_barycentric_posvel(planet, t, ephemeris='DE440'))   

    x[i][:] = np.array([planet[0].x.to(u.meter).value for planet in planets[len(m)*(i-1) : len(m)*(i)]]).astype(np.longdouble)
    y[i][:] = np.array([planet[0].y.to(u.meter).value for planet in planets[len(m)*(i-1) : len(m)*(i)]]).astype(np.longdouble)
    z[i][:] = np.array([planet[0].z.to(u.meter).value for planet in planets[len(m)*(i-1) : len(m)*(i)]]).astype(np.longdouble)

    vx[i][:] = np.array([planet[1].x.to(u.meter/u.second).value for planet in planets[len(m)*(i-1) : len(m)*(i)]]).astype(np.longdouble)
    vy[i][:] = np.array([planet[1].y.to(u.meter/u.second).value for planet in planets[len(m)*(i-1) : len(m)*(i)]]).astype(np.longdouble)
    vz[i][:] = np.array([planet[1].z.to(u.meter/u.second).value for planet in planets[len(m)*(i-1) : len(m)*(i)]]).astype(np.longdouble)
 
    #vcm[i] = np.array([np.sum(vx[i]*m/Mtot), np.sum(vy[i]*m/Mtot), np.sum(vz[i]*m/Mtot)])

    sx[i][:] = np.zeros(len(m)).astype(np.longdouble)
    sy[i][:] = np.zeros(len(m)).astype(np.longdouble)
    sz[i][:] = np.zeros(len(m)).astype(np.longdouble)
    
    index = (i*100)/Neff
    
    if (index % 10 == 0):
    	print('SS orbits upload: {}%'.format(index))


    #print(t, (t - t0).sec, (i)*(N/Neff)*dt2, i)
    
t_s = (N)*dt2 - (N/Neff)*dt2 #sec
t_s = t_s/(60*60*24*365) #year
t_nat = (t_sim[-1])/(60*60*24*365)

print('Simulation time: {} - {}'.format(t_nat, t_s)) #, len(t_nat), len(t_sim))








N_arr = np.linspace(0, N - N/Neff, Neff)
#print(N_arr)

q = np.array([[[0 for i in range(0, 3)] for j in range(0, Neff)] for m in range(0,len(m))]).astype(np.longdouble)
p = np.array([[[0 for i in range(0, 3)] for j in range(0, Neff)] for m in range(0,len(m))]).astype(np.longdouble)
spn = np.array([[[0 for i in range(0, 3)] for j in range(0, Neff)] for m in range(0,len(m))]).astype(np.longdouble)

#print(np.shape(s))

for k in range(len(m)):
    for i in range(0, Neff):
                          
        q[k,i,:] = s[i][k]['q'][:]       
                    
        p[k,i,:] = s[i][k]['p'][:]
        
        spn[k,i,:] = s[i][k]['s'][:]


           #------------   PLOTS  -------------#

col_rainbow = cm.rainbow(np.linspace(0, 1, len(masses)))    
col_viridis = cm.viridis(np.linspace(0, 1, len(masses)))  


f = plt.figure(figsize=(6,4))
ax = f.add_subplot(111)
ax.plot(N_arr, T, label = 'Kinetic')
ax.plot(N_arr, V, label = 'Potential')
ax.plot(N_arr, H, label = 'Hamiltonian')
ax.set_xlabel('iteration')
ax.set_ylabel('Energy')
ax.grid()
plt.legend()
plt.show()

f = plt.figure(figsize=(16,6))
ax = f.add_subplot(111, projection = '3d')
for k in range(0, len(m)):
    for i in range(0, Neff):   
        ax.scatter(float(q[k,i,0]/AU), float(q[k,i,1]/AU), float(q[k,i,2]/AU), color = col_rainbow[k])  
        ax.scatter(x[i][k]/AU, y[i][k]/AU, z[i][k]/AU, color = col_viridis[k])    
        if (i==Neff-1):
            ax.scatter(float(q[k,i,0]/AU), float(q[k,i,1]/AU), float(q[k,i,2]/AU), color = col_rainbow[k], label = 'Planet {}'.format(k))
            ax.scatter(x[i][k]/AU, y[i][k]/AU, z[i][k]/AU, color = col_viridis[k], label = 'Astro-Planet {}'.format(k))
            
ax.set_xlabel('x [AU]')
ax.set_ylabel('y [AU]')
ax.set_zlabel('z [AU]')        
plt.legend()
plt.show()








r_astro = np.sqrt(x[:,1]*x[:,1] + y[:,1]*y[:,1] + z[:,1]*z[:,1])
r_sim = np.sqrt(q[1,:,0]*q[1,:,0] + q[1,:,1]*q[1,:,1] + q[1,:,2]*q[1,:,2])

r_diff = np.sqrt((x[:,1]-q[1,:,0])**2 + (y[:,1]-q[1,:,1])**2 + (z[:,1]-q[1,:,2])**2)

f = plt.figure(figsize=(16,6))

ax = f.add_subplot(111)
ax.scatter(N_arr, abs(r_astro - r_sim), label = 'Diff of mod')
ax.scatter(N_arr, r_diff, label = 'Mod of diff')
#ax.scatter(N_arr, r_sim, label = 'Simulation')
ax.set_xlabel('iteration')
ax.set_ylabel('Orbital displacement [m]')   
plt.legend()   
plt.grid()

plt.show()







AstPy_err = np.array([0., 300000., 800000., 4600,  7700000., 76000000., 267000000.]).astype(np.longdouble)  

f = plt.figure(figsize=(16,10))

ax1 = f.add_subplot(131)
for k in range(0, len(m)):
    for i in range(0, Neff):
        #ax1.scatter(N_arr[i], abs(q[k,i,0] - x[i][k]), color = col_rainbow[k])
        ax1.errorbar(N_arr[i], abs(q[k,i,0] - x[i][k]), yerr = AstPy_err[k] + D[i][k][0], fmt='o', color = col_rainbow[k])
        if (i==Neff-1):
            #ax1.scatter(N_arr[i], abs(q[k,i,0] - x[i][k]), color = col_rainbow[k])
            ax1.errorbar(N_arr[i], abs(q[k,i,0] - x[i][k]), yerr = AstPy_err[k]+ D[i][k][0], fmt='o', color = col_rainbow[k], label = 'Planet {}'.format(k))
ax1.set_xlabel('iteration')
ax1.set_ylabel(r'$\Delta x$ [m]')
plt.grid()
plt.legend()

ax2 = f.add_subplot(132)
for k in range(0, len(m)):
    for i in range(0, Neff):
        #ax2.scatter(N_arr[i], abs(q[k,i,1] - y[i][k]), color = col_rainbow[k])
        ax2.errorbar(N_arr[i], abs(q[k,i,1] - y[i][k]), yerr = AstPy_err[k]+ D[i][k][1], fmt='o', color = col_rainbow[k])
        if (i==Neff-1):
            #ax2.scatter(N_arr[i], abs(q[k,i,1] - y[i][k]), color = col_rainbow[k])
            ax2.errorbar(N_arr[i], abs(q[k,i,1] - y[i][k]), yerr = AstPy_err[k]+ D[i][k][1], fmt='o', color = col_rainbow[k], label = 'Planet {}'.format(k))
ax2.set_xlabel('iteration')
ax2.set_ylabel(r'$\Delta y$ [m]')
plt.grid()
plt.legend()

ax3 = f.add_subplot(133)
for k in range(0, len(m)):
    for i in range(0, Neff):
        #ax3.scatter(N_arr[i], abs(q[k,i,2] - z[i][k]), color = col_rainbow[k])
        ax3.errorbar(N_arr[i], abs(q[k,i,2] - z[i][k]), yerr = AstPy_err[k]+ D[i][k][2], fmt='o', color = col_rainbow[k])
        if (i==Neff-1):
            #ax3.scatter(N_arr[i], abs(q[k,i,2] - z[i][k]), color = col_rainbow[k])
            ax3.errorbar(N_arr[i], abs(q[k,i,2] - z[i][k]), yerr = AstPy_err[k]+ D[i][k][2], fmt='o', color = col_rainbow[k], label = 'Planet {}'.format(k))
ax3.set_xlabel('iteration')
ax3.set_ylabel(r'$\Delta z$ [m]')
plt.grid()
plt.legend()

plt.show()    



f = plt.figure(figsize=(16,10))

ax1 = f.add_subplot(131)
for k in range(0, len(m)):
    for i in range(0, Neff):
        ax1.scatter(N_arr[i], D[i][k][0], color = col_rainbow[k])
        if (i==Neff-1):
            ax1.scatter(N_arr[i], D[i][k][0], color = col_rainbow[k], label = 'Error Planet {}'.format(k))
ax1.set_xlabel('iteration')
ax1.set_ylabel(r'$\Delta x$ [m]')
plt.grid()
plt.legend()

ax2 = f.add_subplot(132)
for k in range(0, len(m)):
    for i in range(0, Neff):
        ax2.scatter(N_arr[i], D[i][k][1], color = col_rainbow[k])
        if (i==Neff-1):
            ax2.scatter(N_arr[i], D[i][k][1], color = col_rainbow[k], label = 'Error Planet {}'.format(k))
ax2.set_xlabel('iteration')
ax2.set_ylabel(r'$\Delta y$ [m]')
plt.grid()
plt.legend()

ax3 = f.add_subplot(133)
for k in range(0, len(m)):
    for i in range(0, Neff):
        ax3.scatter(N_arr[i], D[i][k][2], color = col_rainbow[k])
        if (i==Neff-1):
            ax3.scatter(N_arr[i], D[i][k][2], color = col_rainbow[k], label = 'Error Planet {}'.format(k))
ax3.set_xlabel('iteration')
ax3.set_ylabel(r'$\Delta z$ [m]')
plt.grid()
plt.legend()

plt.show()       

