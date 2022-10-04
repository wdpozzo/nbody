from nbody.engine import run, _H_2body
from Kep_dynamic import KepElemToCart, kepler_sol_sys
import numpy as np
import pickle

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits import mplot3d

#import pyorb

day = 86400. #*u.second
year = day*365

G = 6.67e-11 #*(u.meter**3)/(u.kilogram*u.second**2) # 6.67e-11 #

# AU**3/((d**2)*solMass) = (86400 * 86400) /( 2e30 * 1.5e11 * 1.5e11)

C = 299792458. #*(u.meter/u.second) #299792458. #
Ms = 1.988e30 #*(u.kilogram) # 1.988e30 #

Mmerc = 0.3301e24
Mearth = 5.9722e24 
AU = 149597870700. #*u.meter
Ms = 1.988e30 #kg

plot_step = 10000
buffer_lenght = 100000
data_thin = 1

PN_order = 1
ICN_it = 2

nbodies = 6
dt = 1000
N  = 1000000
p = 0

Neff = int(N/(data_thin*plot_step))
nout = int(N/buffer_lenght)       

#-----------------------S-Stars initial conditions---------------------------
'''
orb_S2 = pyorb.Orbit(
    M0 = 13.60*Ms,
    a = 1010.9591*AU,
    e = 0.8863,
    i = 134.35,
    omega = 66.450,
    Omega = 227.97,
    degrees = True,
    type = 'true',
)

pos = np.empty((3, N))

for ti in range(neff):
    pos[:,ti] = np.squeeze(orb.pos)
    orb.propagate(dt)

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
ax.plot(pos[0,:], pos[1,:], pos[2,:])
ax.set_xlabel('x [AU]')
ax.set_ylabel('y [AU')
ax.set_zlabel('z [AU]')     
ax.legend()
ax.grid()
plt.show()  
'''

x_S2, y_S2, z_S2, vx_S2, vy_S2, vz_S2 = KepElemToCart(1010.9591*AU, 16.051*365*24*60*60, 0.8863, 227.97, 134.35, 66.450, N, Neff)

x_S8, y_S8, z_S8, vx_S8, vy_S8, vz_S8 = KepElemToCart(3262.2005*AU, 92.989*365*24*60*60, 0.8028, 315.46, 74.358, 346.86, N, Neff)

x_S12, y_S12, z_S12, vx_S12, vy_S12, vz_S12 = KepElemToCart(2412.6882*AU, 59.145*365*24*60*60, 0.8883, 230.37, 33.520, 317.98, N, Neff) 

x_S1, y_S1, z_S1, vx_S1, vy_S1, vz_S1 = KepElemToCart(4828.4267*AU, 165.66*365*24*60*60, 0.5533, 342.39, 119.33, 122.23, N, Neff) 

x_S9, y_S9, z_S9, vx_S9, vy_S9, vz_S9 = KepElemToCart(2232.4459*AU, 52.081*365*24*60*60, 0.6425, 156.70, 82.532, 150.43, N, Neff) 

#x_S38, y_S38, z_S38, vx_S38, vy_S38, vz_S38, r_S38 = KepElemToCart(a, T, e, Omega, i, w, N. Neff)

#x_S55, y_S55, z_S55, vx_S55, vy_S55, vz_S55, r_S55 = KepElemToCart(a, T, e, Omega, i, w, N. Neff)  

m = np.array([4.0e6*Ms, 13.60*Ms, 13.20*Ms, 7.60*Ms, 12.40*Ms, 8.20*Ms]).astype(np.longdouble) #masses of Sgr A*, S0-2, S0-8, S0-12, S0-1, S0-9

#3D initial conditions
x = np.array([-0.6750200794271042*AU, x_S2, x_S8, x_S12, x_S1, x_S9]).astype(np.longdouble)  #-0.6750200794271042*AU
y = np.array([20.24490944238422*AU, y_S2, y_S8, y_S12, y_S1, y_S9]).astype(np.longdouble)  #20.24490944238422*AU
z = np.array([0., z_S2, z_S8, z_S12, z_S1, z_S9]).astype(np.longdouble)  

#print(x, y, x)

vx = np.array([0., vx_S2, vx_S8, vx_S12, vx_S1, vx_S9]).astype(np.longdouble)
vy = np.array([0., vy_S2, vy_S8, vy_S12, vy_S1, vy_S9]).astype(np.longdouble) 
vz = np.array([0., vz_S2, vz_S8, vz_S12, vz_S1, vz_S9]).astype(np.longdouble) 

sx = np.array([0., 0., 0., 0., 0., 0.]).astype(np.longdouble)
sy = np.array([0., 0., 0., 0., 0., 0.]).astype(np.longdouble)
sz = np.array([0., 0., 0., 0., 0., 0.]).astype(np.longdouble)

#print(m,x,y,z,vx,vy,vz,sx,sy,sz)

#------------------------------------------------------#   
Neff = int(N/(data_thin*plot_step))
nout = int(N/buffer_lenght)    
#------------------------------------------------------#

if (p == 0):
    run(N, np.longdouble(dt), PN_order, m, x, y, z, m*vx, m*vy, m*vz, sx, sy, sz, ICN_it, data_thin, buffer_lenght)

s, H, T, V = [], [], [], []

for i in range(nout):  

    s_tot, H_tot, T_tot, V_tot = [], [], [], []

    s_tot.append(pickle.load(open('solution_{}_order{}.pkl'.format(i, order),'rb')))
    H_tot.append(pickle.load(open('hamiltonian_{}_order{}.pkl'.format(i, order),'rb')))
    T_tot.append(pickle.load(open('kinetic_{}_order{}.pkl'.format(i, order),'rb')))
    V_tot.append(pickle.load(open('potential_{}_order{}.pkl'.format(i, order),'rb')))       

    s.append(s_tot[0][::plot_step])
    H.append(H_tot[0][::plot_step])
    T.append(T_tot[0][::plot_step])
    V.append(V_tot[0][::plot_step])       

    del s_tot
    del H_tot
    del T_tot
    del V_tot
    
    index = i*100/nout 
    if (index) % 10 == 0 :
        print("Data deframmentation: {}%".format(index))

s = np.array(s, dtype=object)#.flatten()
H = np.array(H, dtype=object)#.flatten()
T = np.array(T, dtype=object)#.flatten()
V = np.array(V, dtype=object)#.flatten()   

s = np.concatenate((s[:]))
H = np.concatenate((H[:]))
T = np.concatenate((T[:]))
V = np.concatenate((V[:])) 

N_arr = np.linspace(0, N, Neff)
#plotting_step = np.maximum(64, Neff//int(0.1*Neff))

f = plt.figure(figsize=(6,4))

ax = f.add_subplot(111)
ax.plot(N_arr, T, label = 'Kinetic')
ax.plot(N_arr, V, label = 'Potential')
ax.plot(N_arr, H, label = 'Hamiltonian')
ax.set_xlabel('iteration')
ax.set_ylabel('Energy')
ax.legend()
ax.grid()

plt.show()  

q = np.array([[[0 for i in range(0, 3)] for Neff in range(0, Neff)] for m in range(0,len(m))]).astype(np.longdouble)
p = np.array([[[0 for i in range(0, 3)] for Neff in range(0, Neff)] for m in range(0,len(m))]).astype(np.longdouble)
spn = np.array([[[0 for i in range(0, 3)] for Neff in range(0, Neff)] for m in range(0,len(m))]).astype(np.longdouble)

for k in range(len(m)):

    for i in range(0, Neff):
                    
        q[k,i,0] = s[i][k]['q'][0]
        q[k,i,1] = s[i][k]['q'][1]
        q[k,i,2] = s[i][k]['q'][2]
  
        p[k,i,0] = s[i][k]['p'][0]
        p[k,i,1] = s[i][k]['p'][1]
        p[k,i,2] = s[i][k]['p'][2]
  
        spn[k,i,0] = s[i][k]['s'][0]
        spn[k,i,1] = s[i][k]['s'][1]
        spn[k,i,2] = s[i][k]['s'][2] 

H_2body = []
V_2body = []
T_2body = [] 

for i in range(0, Neff):

    x_2b = np.array([q[0,i,0],q[1,i,0]]).astype(np.longdouble)
    y_2b = np.array([q[0,i,1],q[1,i,1]]).astype(np.longdouble)
    z_2b = np.array([q[0,i,2],q[1,i,2]]).astype(np.longdouble)

    px_2b = np.array([p[0,i,0],p[1,i,0]]).astype(np.longdouble)
    py_2b = np.array([p[0,i,1],p[1,i,1]]).astype(np.longdouble)
    pz_2b = np.array([p[0,i,2],p[1,i,2]]).astype(np.longdouble)

    sx_2b = np.array([spn[0,i,0],spn[1,i,0]]).astype(np.longdouble)
    sy_2b = np.array([spn[0,i,1],spn[1,i,1]]).astype(np.longdouble)
    sz_2b = np.array([spn[0,i,2],spn[1,i,2]]).astype(np.longdouble)

#print(mass,x,y,z,px,py,pz,sx,sy,sz)
             
    h, t, v = _H_2body(m, x_2b, y_2b, z_2b, px_2b, py_2b, pz_2b, sx_2b, sy_2b, sz_2b, 1)

    H_2body.append(h)
    T_2body.append(t)
    V_2body.append(v)
             
L, P_quad, a_p1, a_p2, a_p3, a_p4, q_peri, phi_shift_1, phi_shift_2 = kepler_sol_sys(p, q, Neff, H_2body, m, dt, 1)

#----------------------------#    
p_shift = a_p1 + a_p2*a_p3 + a_p4

p_shift_t = np.sum(p_shift)
p_shift_t = p_shift_t/Neff

p_shift_1 = np.sum(phi_shift_1)
p_shift_1 = p_shift_1/Neff 
#----------------------------#

colors = cm.rainbow(np.linspace(0, 1, len(masses)))    

f = plt.figure(figsize=(16,6))
ax = f.add_subplot(111, projection = '3d')

for i in range(Neff):
    if (i <= Neff-2):
        for k in range(len(masses)):
            ax.scatter(float(q[k][i][0]/AU), float(q[k][i][1]/AU), float(q[k][i][2]/AU), s = 0.1, c = colors[k], alpha=0.8)
            
    if (i == Neff-1):
        ax.scatter(float(q[0][i][0]/AU), float(q[0][i][1]/AU), float(q[0][i][2]/AU),s = 0.1, label = 'Sgr-A', c = colors[0],alpha=0.8)
        ax.scatter(float(q[1][i][0]/AU), float(q[1][i][1]/AU), float(q[1][i][2]/AU),s = 0.1, label = 'S2', c = colors[1],alpha=0.8)
        ax.scatter(float(q[2][i][0]/AU), float(q[2][i][1]/AU), float(q[2][i][2]/AU),s = 0.1, label = 'S8', c = colors[2],alpha=0.8)
        ax.scatter(float(q[3][i][0]/AU), float(q[3][i][1]/AU), float(q[3][i][2]/AU),s = 0.1, label = 'S12', c = colors[3], alpha=0.8)      
        ax.scatter(float(q[4][i][0]/AU), float(q[4][i][1]/AU), float(q[4][i][2]/AU),s = 0.1, label = 'S1', c = colors[4], alpha=0.8)  
        ax.scatter(float(q[5][i][0]/AU), float(q[5][i][1]/AU), float(q[5][i][2]/AU),s = 0.1, label = 'S9', c = colors[5], alpha=0.8)   
        
ax.set_xlabel('x [AU]')
ax.set_ylabel('y [AU]')
ax.set_zlabel('z [AU]')      
plt.legend()
plt.show()

'''     
f = plt.figure(figsize=(16,6))

ax1 = f.add_subplot(131)
ax1.plot(N_arr, q[1][:][0] - x_S2[:], label = 'PN vs Newtonian (S2)', alpha=0.8)
ax1.plot(N_arr, q[2][:][0] - x_S8[:], label = 'PN vs Newtonian (S8)', alpha=0.8)
ax1.plot(N_arr, q[3][:][0] - x_S12[:], label = 'PN vs Newtonian (S12)', alpha=0.8)
ax1.set_xlabel('Iteration')
ax1.set_ylabel(r'$\Delta x$ [m]')
plt.legend()   

ax2 = f.add_subplot(132)
ax2.plot(N_arr, q[1][:][1] - y_S2[:], label = 'PN vs Newtonian (S2)', alpha=0.8)
ax2.plot(N_arr, q[2][:][1] - y_S8[:], label = 'PN vs Newtonian (S8)', alpha=0.8)
ax2.plot(N_arr, q[3][:][1] - y_S12[:], label = 'PN vs Newtonian (S12)', alpha=0.8)
ax2.set_xlabel('Iteration')
ax2.set_ylabel(r'$\Delta y$ [m]')
plt.legend()

ax3 = f.add_subplot(133)
ax3.plot(N_arr, q[1][:][2] - z_S2[:], label = 'PN vs Newtonian (S2)', alpha=0.8)
ax3.plot(N_arr, q[2][:][2] - z_S8[:], label = 'PN vs Newtonian (S8)', alpha=0.8)
ax3.plot(N_arr, q[3][:][2] - z_S12[:], label = 'PN vs Newtonian (S12)', alpha=0.8)
ax3.set_xlabel('Iteration')
ax3.set_ylabel(r'$\Delta z$ [m]')
plt.legend()   

plt.show()
'''

print('Total theorethical shift = {} [rad/rev];\nNumerical shift (method 1) = {} [rad/rev];\nNumerical shift (method 2) = {} [rad/rev].'.format(p_shift_t, p_shift_1, phi_shift_2))
