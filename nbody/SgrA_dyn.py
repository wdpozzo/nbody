from nbody.engine import run, _H_2body
from Kep_dynamic import KepElemToCart, kepler_sol_sys
import numpy as np
import pickle
from nbody.CM_coord_system import CM_system, SpherToCart

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits import mplot3d

#import pyorb

day = 86400. #*u.second
year = day*365
arcs = 0.00000485 #[rad]

G = 6.67e-11 #*(u.meter**3)/(u.kilogram*u.second**2) # 6.67e-11 #

# AU**3/((d**2)*solMass) = (86400 * 86400) /( 2e30 * 1.5e11 * 1.5e11)

C = 299792458. #*(u.meter/u.second) #299792458. #

Mmerc = 0.3301e24
Mearth = 5.97216787e24 
AU = 149597870700. #*u.meter
Ms = 1.98840987e30 #*(u.kilogram) # 1.988e30 #

plot_step = 1000
buffer_lenght = 10000
data_thin = 10

#PN_order = 1
ICN_it = 2

dt = 100
N  = 100000
p = 1

Neff = int(N/(data_thin*plot_step))
nout = int(N/buffer_lenght)       

N_arr = np.linspace(0, N, Neff)
#-----------------------S-Stars initial conditions---------------------------

stars_names = [
'Sgr A*',
'S0-2',
'S0-8',
#'S0-12',
#'S0-1',
#'S0-38',
#'S0-55',
]


x_S2, y_S2, z_S2, vx_S2, vy_S2, vz_S2 = KepElemToCart(1023.3879*AU, 16.051*365*24*60*60, 0.8863, 227.97, 134.35, 66.450, N, Neff)

x_S8, y_S8, z_S8, vx_S8, vy_S8, vz_S8 = KepElemToCart(3302.3061*AU, 92.989*365*24*60*60, 0.8028, 315.46, 74.358, 346.86, N, Neff)

#x_S12, y_S12, z_S12, vx_S12, vy_S12, vz_S12 = KepElemToCart(2442.3499*AU, 59.145*365*24*60*60, 0.8883, 230.37, 33.520, 317.98, N, Neff) 

#x_S1, y_S1, z_S1, vx_S1, vy_S1, vz_S1 = KepElemToCart(4887.7876*AU, 165.66*365*24*60*60, 0.5533, 342.39, 119.33, 122.23, N, Neff) 

#x_S9, y_S9, z_S9, vx_S9, vy_S9, vz_S9 = KepElemToCart(2259.8917*AU, 52.081*365*24*60*60, 0.6425, 156.70, 82.532, 150.43, N, Neff) 

#x_S38, y_S38, z_S38, vx_S38, vy_S38, vz_S38, r_S38 = KepElemToCart(a, T, e, Omega, i, w, N. Neff)

#x_S55, y_S55, z_S55, vx_S55, vy_S55, vz_S55, r_S55 = KepElemToCart(a, T, e, Omega, i, w, N. Neff)  

m = np.array([4.4e6*Ms, #Sgr A*
	13.60*Ms, #S0-2
	13.20*Ms, #S0-8
	#7.60*Ms, #S0-12
	#12.40*Ms, #S0-1
	#8.20*Ms #S0-9
	]).astype(np.longdouble)

nbodies = len(m)

#---------------------3D initial conditions-------------------
x_0 = np.array([ -0.6833188045410651*AU, #0.5762929676852356*AU, 
	x_S2, 
	x_S8, 
	#x_S12, 
	#x_S1, 
	#x_S9
	]).astype(np.longdouble)  #-0.6750200794271042*AU

y_0 = np.array([ 20.493801206555094*AU, #4.610343741481885*AU, 
	y_S2, 
	y_S8, 
	#y_S12, 
	#y_S1, 
	#y_S9
	]).astype(np.longdouble)  #20.24490944238422*AU

z_0 = np.array([0., 
	z_S2, 
	z_S8, 
	#z_S12, 
	#z_S1, 
	#z_S9
	]).astype(np.longdouble)  

vx_0 = np.array([0., 
	vx_S2, 
	vx_S8, 
	#vx_S12, 
	#vx_S1, 
	#vx_S9
	]).astype(np.longdouble)

vy_0 = np.array([0., 
	vy_S2, 
	vy_S8, 
	#vy_S12, 
	#vy_S1, 
	#vy_S9
	]).astype(np.longdouble) 

vz_0 = np.array([0., 
	vz_S2, 
	vz_S8, 
	#vz_S12, 
	#vz_S1, 
	#vz_S9
	]).astype(np.longdouble) 

sx_0 = np.zeros(nbodies).astype(np.longdouble)
sy_0 = np.zeros(nbodies).astype(np.longdouble)
sz_0 = np.zeros(nbodies).astype(np.longdouble)

#----------S0-2 astronometric data (2002-2015) from  " arXiv:1708.03507v1 [astro-ph.GA] " -----------
t_obs = np.array([2002.578, 2003.447, 2003.455, 2004.511, 2004.516, 2004.574, 2005.268, 2006.490, 2006.584, 2006.726, 2006.800, 2007.205, 2007.214, 2007.255, 2007.455, 2008.145, 2008.197, 2008.268, 2008.456, 2008.598, 2008.708, 2009.299, 2009.344, 2009.501, 2009.605, 2009.611, 2009.715, 2010.444, 2010.455, 2011.400, 2012.374, 2013.488, 2015.581])

RA = np.array([0.0386*arcs, 0.0385*arcs, 0.0393*arcs, 0.0330*arcs, 0.0333*arcs, 0.0315*arcs, 0.0265*arcs, 0.0141*arcs, 0.0137*arcs, 0.0129*arcs, 0.0107*arcs, 0.0064*arcs, 0.0058*arcs, 0.0069*arcs, 0.0047*arcs, -0.0076*arcs, -0.0082*arcs, -0.0084*arcs, -0.0118*arcs, -0.0126*arcs, -0.0127*arcs, -0.0216*arcs, -0.0218*arcs, -0.0233*arcs, -0.0266*arcs, -0.0249*arcs, -0.0260*arcs, -0.0347*arcs, -0.0340*arcs, -0.0430*arcs, -0.0518*arcs, -0.0603*arcs, -0.0690*arcs])

Decl = np.array([0.0213*arcs, 0.0701*arcs, 0.0733*arcs, 0.1191*arcs, 0.1206*arcs, 0.1206*arcs, 0.1389*arcs, 0.1596*arcs, 0.1609*arcs, 0.1627*arcs, 0.1633*arcs, 0.1681*arcs, 0.1682*arcs, 0.1691*arcs, 0.1709*arcs, 0.1775*arcs, 0.1780*arcs, 0.1777*arcs, 0.1798*arcs, 0.1802*arcs, 0.1806*arcs, 0.1805*arcs, 0.1813*arcs, 0.1803*arcs, 0.1800*arcs, 0.1806*arcs, 0.1804*arcs, 0.1780*arcs, 0.1774*arcs, 0.1703*arcs, 0.1617*arcs, 0.1442*arcs, 0.1010*arcs])

D_obs_RA = np.array([0.0066*arcs, 0.0009*arcs, 0.0012*arcs, 0.0010*arcs, 0.0009*arcs, 0.0009*arcs, 0.0007*arcs, 0.0065*arcs, 0.0033*arcs, 0.0033*arcs, 0.0033*arcs, 0.0004*arcs, 0.0004*arcs, 0.0010*arcs, 0.0004*arcs, 0.0007*arcs, 0.0007*arcs, 0.0006*arcs, 0.0006*arcs, 0.0009*arcs, 0.0008*arcs, 0.0006*arcs, 0.0006*arcs, 0.0005*arcs, 0.0012*arcs, 0.0006*arcs, 0.0006*arcs, 0.0013*arcs, 0.0008*arcs, 0.0009*arcs, 0.0012*arcs, 0.0006*arcs, 0.0014*arcs])

D_obs_Decl = np.array([0.0065*arcs, 0.0010*arcs, 0.0012*arcs, 0.0008*arcs, 0.0006*arcs, 0.0009*arcs, 0.0011*arcs, 0.0065*arcs, 0.0007*arcs, 0.0007*arcs, 0.0007*arcs, 0.0007*arcs, 0.0008*arcs, 0.0007*arcs, 0.0006*arcs, 0.0012*arcs, 0.0011*arcs, 0.0008*arcs, 0.0009*arcs, 0.0010*arcs, 0.0013*arcs, 0.0009*arcs, 0.0009*arcs, 0.0008*arcs, 0.0015*arcs, 0.0008*arcs, 0.0008*arcs, 0.0021*arcs, 0.0013*arcs, 0.0017*arcs, 0.0016*arcs, 0.0019*arcs, 0.0010*arcs])

#------------------------------------------------------#   
Neff = int(N/(data_thin*plot_step))
nout = int(N/buffer_lenght)    
#------------------------------------------------------#
	
x_S02_obs, y_S02_obs, z_S02_obs = SpherToCart(RA, Decl, 1691371411.2*AU)

Dx_S02_obs, Dy_S02_obs, Dz_S02_obs = SpherToCart(D_obs_RA, D_obs_Decl, 1691371411.2*AU)

#print(x_S02_obs, y_S02_obs, z_S02_obs)
#print(Dx_S02_obs, Dy_S02_obs, Dz_S02_obs)

'''
if (p == 0):
    run(N, np.longdouble(dt), PN_order, m, x, y, z, m*vx, m*vy, m*vz, sx, sy, sz, ICN_it, data_thin, buffer_lenght)

s, H, T, V = [], [], [], []

for i in range(nout):  

    s_tot, H_tot, T_tot, V_tot = [], [], [], []

    s_tot.append(pickle.load(open('solution_{}_order{}.pkl'.format(i, order),'rb')))
    H_tot.append(pickle.load(open('hamiltonian_{}_order{}.pkl'.format(i, order),'rb')))
    T_tot.append(pickle.load(open('kinetic_{}_order{}.pkl'.format(i, order),'rb')))
    V_tot.append(pickle.load(open('potential_{}_order{}.pkl'.format(i, order),'rb')))
    t_tot.append(pickle.load(open('time_{}_order{}.pkl'.format(i, order),'rb')))
    D_tot.append(pickle.load(open('error_{}_order{}.pkl'.format(i, order),'rb')))

    s.append(s_tot[0][::plot_step])
    H.append(H_tot[0][::plot_step])
    T.append(T_tot[0][::plot_step])
    V.append(V_tot[0][::plot_step])
    D.append(D_tot[0][::plot_step])
    t_sim.append(t_tot[::plot_step]) 		    

    del s_tot
    del H_tot
    del T_tot
    del V_tot
    del D_tot
    del t_tot
    
    index = i*100/nout 
    if (index) % 10 == 0 :
        print("Data deframmentation: {}%".format(index))

s = np.array(s, dtype=object)#.flatten()
H = np.array(H, dtype=object)#.flatten()
T = np.array(T, dtype=object)#.flatten()
V = np.array(V, dtype=object)#.flatten()
D = np.array(D_1PN, dtype=object)
t_sim = np.array(t_sim_1PN, dtype=object)

s = np.concatenate((s[:]))
H = np.concatenate((H[:]))
T = np.concatenate((T[:]))
V = np.concatenate((V[:]))
D = np.concatenate((D[:]), axis=0)
t_sim = np.concatenate((t_sim[:]), axis=0)

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

print('Total theorethical shift = {} [rad/rev];\nNumerical shift (method 1) = {} [rad/rev];\nNumerical shift (method 2) = {} [rad/rev].'.format(p_shift_t, p_shift_1, phi_shift_2))
'''

if (p == 0):
	run(N, np.longdouble(dt), 0, m, x_0, y_0, z_0, m*vx_0, m*vy_0, m*vz_0, sx_0, sy_0, sz_0, ICN_it, data_thin, buffer_lenght, plot_step)

s_N, H_N, T_N, V_N, D_N, t_sim_N  = [], [], [], [], [], []

for i in range(nout):  
	s_tot, H_tot, T_tot, V_tot, D_tot, t_tot  = [], [], [], [], [], []
	
	s_tot.append(pickle.load(open('solution_{}_order{}.pkl'.format(i, 0),'rb')))
	H_tot.append(pickle.load(open('hamiltonian_{}_order{}.pkl'.format(i, 0),'rb')))
	T_tot.append(pickle.load(open('kinetic_{}_order{}.pkl'.format(i, 0),'rb')))
	V_tot.append(pickle.load(open('potential_{}_order{}.pkl'.format(i, 0),'rb')))	
	t_tot.append(pickle.load(open('time_{}_order{}.pkl'.format(i, 0),'rb')))
	D_tot.append(pickle.load(open('error_{}_order{}.pkl'.format(i, 0),'rb'))) 
  
	s_N.append(s_tot[0][::plot_step])
	H_N.append(H_tot[0][::plot_step])
	T_N.append(T_tot[0][::plot_step])
	V_N.append(V_tot[0][::plot_step])
	D_N.append(D_tot[0][::plot_step])          
	t_sim_N.append(t_tot[::plot_step]) 

	del s_tot
	del H_tot
	del T_tot
	del V_tot
	del D_tot     
	del t_tot   
	
	index_0 = i*100/nout 
	if (index_0) % 10 == 0 :
		print("Data deframmentation: order 0 - {}%".format(index_0))
	
s_N = np.array(s_N, dtype=object)#.flatten()
H_N = np.array(H_N, dtype=object)#.flatten()
T_N = np.array(T_N, dtype=object)#.flatten()
V_N = np.array(V_N, dtype=object)#.flatten()
D_N = np.array(D_N, dtype=object)
t_sim_N = np.array(t_sim_N, dtype=object)

s_N = np.concatenate((s_N[:]))
H_N = np.concatenate((H_N[:]))
T_N = np.concatenate((T_N[:]))
V_N = np.concatenate((V_N[:])) 		
D_N = np.concatenate((D_N[:]), axis=0)
t_sim_N = np.concatenate((t_sim_N[:]), axis=0)

if (p == 0):		    
	run(N, np.longdouble(dt), 1, m, x_0, y_0, z_0, m*vx_0, m*vy_0, m*vz_0, sx_0, sy_0, sz_0, ICN_it, data_thin, buffer_lenght, plot_step)

s_1PN, H_1PN, T_1PN, V_1PN, D_1PN, t_sim_1PN = [], [], [], [], [], []

for i in range(nout):  
	s_tot, H_tot, T_tot, V_tot, D_tot, t_tot  = [], [], [], [], [], []
	
	s_tot.append(pickle.load(open('solution_{}_order{}.pkl'.format(i, 1),'rb')))
	H_tot.append(pickle.load(open('hamiltonian_{}_order{}.pkl'.format(i, 1),'rb')))
	T_tot.append(pickle.load(open('kinetic_{}_order{}.pkl'.format(i, 1),'rb')))
	V_tot.append(pickle.load(open('potential_{}_order{}.pkl'.format(i, 1),'rb')))       		
	t_tot.append(pickle.load(open('time_{}_order{}.pkl'.format(i, 1),'rb')))
	D_tot.append(pickle.load(open('error_{}_order{}.pkl'.format(i, 1),'rb')))   

	s_1PN.append(s_tot[0][::plot_step])
	H_1PN.append(H_tot[0][::plot_step])
	T_1PN.append(T_tot[0][::plot_step])
	V_1PN.append(V_tot[0][::plot_step])
	D_1PN.append(D_tot[0][::plot_step])          
	t_sim_1PN.append(t_tot[::plot_step]) 		

	del s_tot
	del H_tot
	del T_tot
	del V_tot
	del D_tot     
	del t_tot   

	index_1 = i*100/nout 
	if (index_1) % 10 == 0 :
		print("Data deframmentation: order 1 - {}%".format(index_1))			
	
s_1PN = np.array(s_1PN, dtype=object)#.flatten()
H_1PN = np.array(H_1PN, dtype=object)#.flatten()
T_1PN = np.array(T_1PN, dtype=object)#.flatten()
V_1PN = np.array(V_1PN, dtype=object)#.flatten()
D_1PN = np.array(D_1PN, dtype=object)
t_sim_1PN = np.array(t_sim_1PN, dtype=object)

s_1PN = np.concatenate((s_1PN[:]))
H_1PN = np.concatenate((H_1PN[:]))
T_1PN = np.concatenate((T_1PN[:]))
V_1PN = np.concatenate((V_1PN[:])) 
D_1PN = np.concatenate((D_1PN[:]), axis=0)
t_sim_1PN = np.concatenate((t_sim_1PN[:]), axis=0)

#------------------------Numerical PN-order confrontation --------------------------

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits import mplot3d

    
f = plt.figure(figsize=(16,14))

ax = f.add_subplot(1,3,1)
ax.plot(N_arr, V_N, label= "Newtonian")
ax.plot(N_arr, V_1PN, label= "1PN")
#ax.plot(range(Neff), V_2PN, label= "2PN")
ax.set_xlabel('iteration', fontsize="x-large")
ax.set_ylabel('Potential energy [J]', fontsize="x-large")
ax.grid()
ax.legend(fontsize="large")
#colors = iter(cm.rainbow(np.linspace(0, 1, nbodies)))

ax2 = f.add_subplot(1,3,2)
ax2.plot(N_arr, T_N, label= "Newtonian")
ax2.plot(N_arr, T_1PN, label= "1PN")
#ax2.plot(range(Neff), T_2PN, label= "2PN")
ax2.set_xlabel('iteration', fontsize="x-large")
ax2.set_ylabel('Kinetic energy [J]', fontsize="x-large")	
ax2.grid()
ax2.legend(fontsize="large")
#colors = iter(cm.rainbow(np.linspace(0, 1, nbodies)))

ax3 = f.add_subplot(1,3,3)
ax3.plot(N_arr, H_N, label= "Newtonian")
ax3.plot(N_arr, H_1PN, label= "1PN")
#ax.plot(range(Neff), H_2PN, label= "2PN")
ax3.set_xlabel('iteration', fontsize="x-large")
ax3.set_ylabel('Energy [J]', fontsize="x-large")
ax3.grid()
ax3.legend(fontsize="large")

plt.show()

#f.savefig('/home/FGeroni/Università/PROGETTI/Tesi/ThesisProject/LaTex/Immagini/SimulationsHamiltonianNorm.pdf', bbox_inches='tight')

H_1PN_N = []
#H_2PN_N = []

arr = []
arr = [1 for i in range(Neff)] 

for i in range(0, Neff):

	H_1PN_N.append((H_1PN[i]/H_N[i]))    
	#H_2PN_N.append(H_2PN[i]/H_N[i])

f = plt.figure(figsize=(16,14))

ax = f.add_subplot(111)
ax.plot(N_arr, arr, label= "Normalized N Hamiltonian")
ax.plot(N_arr, H_1PN_N, label= "Normalized 1PN Hamiltonian")
#ax.plot(range(Neff), H_2PN_N, label= "2PN")
ax.set_xlabel('iteration', fontsize="x-large")
ax.set_ylabel(r'$H/H_{N}$', fontsize="x-large")
ax.grid()
ax.legend(fontsize="large")
plt.show()
#colors = iter(cm.rainbow(np.linspace(0, 1, nbodies)))
#f.savefig('/home/FGeroni/Università/PROGETTI/Tesi/ThesisProject/LaTex/Immagini/SimulationsHamiltonianNorm.pdf', bbox_inches='tight')

#prepare data to evaluate the radius
q_N = np.array([[[0 for i in range(0, 3)] for Neff in range(0, Neff)] for m in range(0,nbodies)], dtype='longdouble')
p_N = np.array([[[0 for i in range(0, 3)] for Neff in range(0, Neff)] for m in range(0,nbodies)], dtype='longdouble')
spn_N = np.array([[[0 for i in range(0, 3)] for Neff in range(0, Neff)] for m in range(0,nbodies)], dtype='longdouble')

q_1PN = np.array([[[0 for i in range(0, 3)] for Neff in range(0, Neff)] for m in range(0,nbodies)], dtype='longdouble')
p_1PN = np.array([[[0 for i in range(0, 3)] for Neff in range(0, Neff)] for m in range(0,nbodies)], dtype='longdouble')
spn_1PN = np.array([[[0 for i in range(0, 3)] for Neff in range(0, Neff)] for m in range(0,nbodies)], dtype='longdouble')

for j in range(nbodies):

	for i in range(Neff):

		q_N[j][i][0] = s_N[i][j]['q'][0]
		q_N[j][i][1] = s_N[i][j]['q'][1]
		q_N[j][i][2] = s_N[i][j]['q'][2]
		
		q_1PN[j][i][0]  = s_1PN[i][j]['q'][0] 
		q_1PN[j][i][1]  = s_1PN[i][j]['q'][1] 
		q_1PN[j][i][2]  = s_1PN[i][j]['q'][2] 
		
		p_N[j][i][0]  = s_N[i][j]['p'][0] 
		p_N[j][i][1]  = s_N[i][j]['p'][1] 
		p_N[j][i][2]  = s_N[i][j]['p'][2] 
		
		p_1PN[j][i][0]  = s_1PN[i][j]['p'][0] 
		p_1PN[j][i][1]  = s_1PN[i][j]['p'][1] 
		p_1PN[j][i][2]  = s_1PN[i][j]['p'][2] 
		
		spn_N[j][i][0]  = s_N[i][j]['s'][0] 
		spn_N[j][i][1]  = s_N[i][j]['s'][1] 
		spn_N[j][i][2]  = s_N[i][j]['s'][2] 

		spn_1PN[j][i][0]  = s_1PN[i][j]['s'][0] 
		spn_1PN[j][i][1]  = s_1PN[i][j]['s'][1] 
		spn_1PN[j][i][2]  = s_1PN[i][j]['s'][2] 


D_tot_q = np.array([[[0 for j in range(3)] for i in range(Neff)] for k in range(nbodies)]).astype(np.longdouble)

for i in range(Neff):
	for j in range(nbodies):
		D_tot_q[j,i,0] = np.sqrt(D_1PN[i][j][0]*D_1PN[i][j][0] + D_N[i][j][0]*D_N[i][j][0])
		D_tot_q[j,i,1] = np.sqrt(D_1PN[i][j][1]*D_1PN[i][j][1] + D_N[i][j][1]*D_N[i][j][1])
		D_tot_q[j,i,2] = np.sqrt(D_1PN[i][j][2]*D_1PN[i][j][2] + D_N[i][j][2]*D_N[i][j][2])

H_2body_N = []
V_2body_N = []
T_2body_N = []
	
H_2body_1PN = []
V_2body_1PN = []
T_2body_1PN = []    

mass = np.array([m[0], m[1]]).astype(np.longdouble)	

for i in range(Neff):
	x_N = np.array([q_N[0,i,0], q_N[1,i,0]]).astype(np.longdouble)
	y_N = np.array([q_N[0,i,1], q_N[1,i,1]]).astype(np.longdouble)
	z_N = np.array([q_N[0,i,2], q_N[1,i,2]]).astype(np.longdouble)
	
	px_N = np.array([p_N[0,i,0], p_N[1,i,0]]).astype(np.longdouble)
	py_N = np.array([p_N[0,i,1], p_N[1,i,1]]).astype(np.longdouble)
	pz_N = np.array([p_N[0,i,2], p_N[1,i,2]]).astype(np.longdouble)
		
	sx_N = np.array([spn_N[0,i,0], spn_N[1,i,0]]).astype(np.longdouble)
	sy_N = np.array([spn_N[0,i,1], spn_N[1,i,1]]).astype(np.longdouble)
	sz_N = np.array([spn_N[0,i,2], spn_N[1,i,2]]).astype(np.longdouble)
		
	x_1PN = np.array([q_1PN[0,i,0], q_1PN[1,i,0]]).astype(np.longdouble)
	y_1PN = np.array([q_1PN[0,i,1], q_1PN[1,i,1]]).astype(np.longdouble)
	z_1PN = np.array([q_1PN[0,i,2], q_1PN[1,i,2]]).astype(np.longdouble)
	
	px_1PN = np.array([p_1PN[0,i,0], p_1PN[1,i,0]]).astype(np.longdouble)
	py_1PN = np.array([p_1PN[0,i,1], p_1PN[1,i,1]]).astype(np.longdouble)
	pz_1PN = np.array([p_1PN[0,i,2], p_1PN[1,i,2]]).astype(np.longdouble)
		
	sx_1PN = np.array([spn_1PN[0,i,0], spn_1PN[1,i,0]]).astype(np.longdouble)
	sy_1PN = np.array([spn_1PN[0,i,1], spn_1PN[1,i,1]]).astype(np.longdouble)
	sz_1PN = np.array([spn_1PN[0,i,2], spn_1PN[1,i,2]]).astype(np.longdouble)
	  
	h, t, v = _H_2body(mass, x_N, y_N, z_N, px_N, py_N, pz_N, sx_N, sy_N, sz_N, 0)
	H_2body_N.append(h)
	T_2body_N.append(t)
	V_2body_N.append(v)

	h, t, v = _H_2body(mass, x_1PN, y_1PN, z_1PN, px_1PN, py_1PN, pz_1PN, sx_1PN, sy_1PN, sz_1PN, 1)
	H_2body_1PN.append(h)
	T_2body_1PN.append(t)
	V_2body_1PN.append(v)		

#coordinate Newton
q_N_rel, p_N_rel, q_N_cm, p_N_cm = CM_system(p_N[0], p_N[1], q_N[0], q_N[1], Neff, m[0], m[1])

L_N, P_quad_N, a_p1_N, a_p2_N, a_p3_N, a_p4_N, q_peri_N, phi_shift_N, phi_shift_test_N, peri_indexes_N, D_shift_N = kepler_sol_sys(p_N, q_N, D_N, Neff, H_2body_N, m, dt, 0)

#coordinate 1PN
q_1PN_rel, p_1PN_rel, q_1PN_cm, p_1PN_cm = CM_system(p_1PN[0], p_1PN[1], q_1PN[0], q_1PN[1], Neff, m[0], m[1])

L_1PN, P_quad_1PN, a_p1_1PN, a_p2_1PN, a_p3_1PN, a_p4_1PN, q_peri_1PN, phi_shift_1PN, phi_shift_test_1PN, peri_indexes_1PN, D_shift_1PN = kepler_sol_sys(p_1PN, q_1PN, D_1PN, Neff, H_2body_1PN, m, dt, 1)

r_N = []
r_1PN = []
#r_2PN = []      

r_N = np.sqrt(q_N_rel[:,0]*q_N_rel[:,0] + q_N_rel[:,1]*q_N_rel[:,1] + q_N_rel[:,2]*q_N_rel[:,2])
r_1PN = np.sqrt(q_1PN_rel[:,0]*q_1PN_rel[:,0] + q_1PN_rel[:,1]*q_1PN_rel[:,1] + q_1PN_rel[:,2]*q_1PN_rel[:,2])
	
col_rainbow = cm.rainbow(np.linspace(0, 1, nbodies))   


D_tot_q_peri = np.array([[0 for j in range(3)] for i in range(len(peri_indexes_N))]).astype(np.longdouble)

n_peri = len(peri_indexes_N)
for i in range(n_peri):
	index_N = int(peri_indexes_N[i])
	index_1PN = int(peri_indexes_1PN[i])
	D_tot_q_peri[i,0] = np.sqrt(D_1PN[index_1PN][1][0]*D_1PN[index_1PN][1][0] + D_N[index_N][1][0]*D_N[index_N][1][0])
	D_tot_q_peri[i,1] = np.sqrt(D_1PN[index_1PN][1][1]*D_1PN[index_1PN][1][1] + D_N[index_N][1][1]*D_N[index_N][1][1])
	D_tot_q_peri[i,2] = np.sqrt(D_1PN[index_1PN][1][2]*D_1PN[index_1PN][1][2] + D_N[index_N][1][2]*D_N[index_N][1][2])


#---------------------------------PLOTS----------------------------------------------
f = plt.figure(figsize=(16,14))
ax = f.add_subplot(111, projection = '3d')
for k in range(nbodies):  
	ax.plot(q_N[k,:,0], q_N[k,:,1], q_N[k,:,2], label= "{} (N)".format(stars_names[k]), alpha=1)
	ax.plot(q_1PN[k,:,0], q_1PN[k,:,1], q_1PN[k,:,2], label= "{} (1PN)".format(stars_names[k]), alpha=1)
ax.set_xlabel('x [km]', fontsize="x-large")
ax.set_ylabel('y [km]', fontsize="x-large")
ax.set_zlabel('z [km]', fontsize="x-large")
ax.legend(fontsize="large")
#f.savefig('/home/FGeroni/Università/PROGETTI/Tesi/ThesisProject/LaTex/Immagini/SimulationsOrbits.pdf', bbox_inches='tight')
plt.show()

f = plt.figure(figsize=(16,14))
ax = f.add_subplot(111, projection = '3d')
ax.plot(q_1PN[0,:,0], q_1PN[0,:,1], q_1PN[0,:,2], label= "Sgr A*")
ax.plot(q_N[1,:,0], q_N[1,:,1], q_N[1,:,2], label= "N S0-2 orbit")
ax.plot(q_1PN[1,:,0], q_1PN[1,:,1], q_1PN[1,:,2], label= "1PN S0-2 orbit")
ax.errorbar(x_S02_obs, y_S02_obs, z_S02_obs, yerr = Dy_S02_obs, xerr = Dx_S02_obs, zerr= Dz_S02_obs, marker='.', label = "Astronometric data (2002-2015)" )
ax.set_xlabel('x [m]', fontsize="x-large")
ax.set_ylabel('y [m]', fontsize="x-large")
ax.set_zlabel('z [m]', fontsize="x-large")
plt.grid()
plt.legend(fontsize="large")
plt.show() 

f = plt.figure(figsize=(16,14))
ax = f.add_subplot(111)
ax.plot(q_1PN[0,:,0], q_1PN[0,:,1], label= "Sgr A*")
ax.plot(q_N[1,:,0], q_N[1,:,1], label= "N S0-2 orbit")
ax.plot(q_1PN[1,:,0], q_1PN[1,:,1], label= "1PN S0-2 orbit")
ax.errorbar(x_S02_obs, y_S02_obs, yerr = Dy_S02_obs, xerr = Dx_S02_obs, marker='.', label = "Astronometric data (2002-2015)" )
ax.set_xlabel('x [m]', fontsize="x-large")
ax.set_ylabel('y [m]', fontsize="x-large")
plt.grid()
plt.legend(fontsize="large")
plt.show()   


f = plt.figure(figsize=(16,14))

ax1 = f.add_subplot(131)
for k in range(nbodies):  
	ax1.errorbar(N_arr, q_N[k,:,0] - q_1PN[k,:,0], yerr = D_tot_q[k,:,0], fmt= 'o', label= "{} N vs.1PN".format(stars_names[k]), alpha=1, color = col_rainbow[k])
ax1.set_xlabel('iteration', fontsize="x-large")
ax1.set_ylabel('Orbit difference: x [m]', fontsize="x-large")
plt.grid()
plt.legend(fontsize="large")

ax2 = f.add_subplot(132)
for k in range(nbodies):  
	ax2.errorbar(N_arr, q_N[k,:,1] - q_1PN[k,:,1], yerr = D_tot_q[k,:,1], fmt= 'o', label= "{} N vs.1PN".format(stars_names[k]), alpha=1, color = col_rainbow[k])
ax2.set_xlabel('iteration', fontsize="x-large")
ax2.set_ylabel('Orbit difference: y [m]', fontsize="x-large")
plt.grid()
plt.legend(fontsize="large")

ax3 = f.add_subplot(133)
for k in range(nbodies):  
	ax3.errorbar(N_arr, q_N[k,:,2] - q_1PN[k,:,2], yerr = D_tot_q[k,:,2], fmt= 'o', label= "{} N vs.1PN".format(stars_names[k]), alpha=1, color = col_rainbow[k])
ax3.set_xlabel('iteration', fontsize="x-large")
ax3.set_ylabel('Orbit difference: z [m]', fontsize="x-large")
plt.grid()
plt.legend(fontsize="large")

plt.show()  


f = plt.figure(figsize=(16,16))

ax1 = f.add_subplot(131)
for k in range(nbodies):  
	ax1.plot(N_arr, q_N[k,:,0] - q_1PN[k,:,0], label= "{} N vs.1PN".format(stars_names[k]), color = col_rainbow[k])
ax1.set_xlabel('iteration', fontsize="x-large")
ax1.set_ylabel('Orbit difference: x [m]', fontsize="x-large")
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.grid()
plt.legend(fontsize="large")

ax2 = f.add_subplot(132)
for k in range(nbodies):  
	ax2.plot(N_arr, q_N[k,:,1] - q_1PN[k,:,1], label= "{} N vs.1PN".format(stars_names[k]), color = col_rainbow[k])
ax2.set_xlabel('iteration', fontsize="x-large")
ax2.set_ylabel('Orbit difference: y [m]', fontsize="x-large")
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.grid()
plt.legend(fontsize="large")

ax3 = f.add_subplot(133)
for k in range(nbodies):  
	ax3.plot(N_arr, q_N[k,:,2] - q_1PN[k,:,2], label= "{} N vs.1PN".format(stars_names[k]), color = col_rainbow[k])
ax3.set_xlabel('iteration', fontsize="x-large")
ax3.set_ylabel('Orbit difference: z [m]', fontsize="x-large")
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.grid()
plt.legend(fontsize="large")

plt.show()  


f = plt.figure(figsize=(16,14))

ax1 = f.add_subplot(131)
for k in range(nbodies):  
	ax1.plot(N_arr, D_tot_q[k,:,0], label= "Numerical error {}".format(stars_names[k]), color = col_rainbow[k])
ax1.set_xlabel('iteration', fontsize="x-large")
ax1.set_ylabel(r'$\Delta x$ [m]', fontsize="x-large")
plt.grid()
plt.legend(fontsize="large")

ax2 = f.add_subplot(132)
for k in range(nbodies):  
	ax2.plot(N_arr, D_tot_q[k,:,1], label= "Numerical error {}".format(stars_names[k]), color = col_rainbow[k])
ax2.set_xlabel('iteration', fontsize="x-large")
ax2.set_ylabel(r'$\Delta y$ [m]', fontsize="x-large")
plt.grid()
plt.legend(fontsize="large")

ax3 = f.add_subplot(133)
for k in range(nbodies):  
	ax3.plot(N_arr, D_tot_q[k,:,2], label= "Numerical error {}".format(stars_names[k]), color = col_rainbow[k])
ax3.set_xlabel('iteration', fontsize="x-large")
ax3.set_ylabel(r'$\Delta z$ [m]', fontsize="x-large")
plt.grid()
plt.legend(fontsize="large")

plt.show() 


f = plt.figure(figsize=(16,14))
ax1 = f.add_subplot(1,1,1)
ax1.plot(N_arr, r_1PN - r_N, label= "S0-2 1PN vs. N orbit")
#ax.plot(range(Neff), H_2PN_N, label= "2PN")
ax1.set_xlabel('iteration', fontsize="x-large")
ax1.set_ylabel('Relative displacement [m]', fontsize="x-large")
ax1.grid()
ax1.legend(fontsize="large")	

plt.show()	


f = plt.figure(figsize=(16,14))

ax1 = f.add_subplot(1,2,1)
ax1.plot(N_arr, H_N, label= "N")
ax1.plot(N_arr, H_1PN, label= "1PN")
ax1.set_xlabel('iteration', fontsize="x-large")
ax1.set_ylabel('Energy [J]', fontsize="x-large")
ax1.grid()
ax1.legend(fontsize="large")

ax2 = f.add_subplot(1,2,2)
ax2.plot(N_arr, P_quad_1PN - P_quad_N, label= "1PN vs. N")
ax2.set_xlabel('iteration', fontsize="x-large")
ax2.set_ylabel('Quadrupole radiation [J/s]', fontsize="x-large")
ax2.grid()
ax2.legend(fontsize="large")		

plt.show()

col_rainbow = cm.rainbow(np.linspace(0, 1, len(q_peri_N)))    
col_viridis = cm.viridis(np.linspace(0, 1, len(q_peri_1PN)))  


f = plt.figure(figsize=(16,14))
ax = f.add_subplot(111, projection = '3d')
for i in range(0, len(q_peri_N)): 
	ax.plot(q_peri_N[i,0], q_peri_N[i,1], q_peri_N[i,2], 'o', label = 'Perihelion n. {} (N)'.format(i), color = col_rainbow[i])
for i in range(0, len(q_peri_1PN)): 
	ax.plot(q_peri_1PN[i,0], q_peri_1PN[i,1], q_peri_1PN[i,2], 'o', label = 'Perihelion n. {} (1PN)'.format(i), color = col_viridis[i])
ax.set_xlabel('x [m]', fontsize="x-large")
ax.set_ylabel('y [m]', fontsize="x-large")
ax.set_zlabel('z [m]', fontsize="x-large")
plt.legend(fontsize="large")
plt.show()   


f = plt.figure(figsize=(16,14))

ax1 = f.add_subplot(131)
ax1.errorbar(np.linspace(0, N, len(q_peri_N)), q_peri_N[:,0] - q_peri_1PN[:,0], yerr= D_tot_q_peri[:,0], fmt= 'o', label='Perihelion N vs. 1PN')
ax1.set_xlabel('iteration', fontsize="x-large")
ax1.set_ylabel('Displacement x [m]', fontsize="x-large")
plt.grid()
plt.legend(fontsize="large")

ax2 = f.add_subplot(132)
ax2.errorbar(np.linspace(0, N, len(q_peri_N)), q_peri_N[:,1] - q_peri_1PN[:,1], yerr= D_tot_q_peri[:,1], fmt= 'o', label='Perihelion N vs. 1PN')

ax2.set_xlabel('iteration', fontsize="x-large")
ax2.set_ylabel('Displacement y [m]', fontsize="x-large")
plt.grid()
plt.legend(fontsize="large")

ax3 = f.add_subplot(133)
ax3.errorbar(np.linspace(0, N, len(q_peri_N)), q_peri_N[:,2] - q_peri_1PN[:,2], yerr= D_tot_q_peri[:,2], fmt= 'o', label='Perihelion N vs. 1PN')
ax3.set_xlabel('iteration', fontsize="x-large")
ax3.set_ylabel('Displacement z [m]', fontsize="x-large")
plt.grid()
plt.legend(fontsize="large")

plt.show()  


# Perihelion shift calculations
p_s_N = a_p1_N + a_p2_N*a_p3_N + a_p4_N
p_s_t_N = np.sum(p_s_N)
p_s_t_N = p_s_t_N/Neff

p_shift_N = np.sum(phi_shift_N)
p_shift_N = p_shift_N/Neff 

p_s_1PN = a_p1_1PN + a_p2_1PN*a_p3_1PN + a_p4_1PN
p_s_t_1PN = np.sum(p_s_1PN)
p_s_t_1PN = p_s_t_1PN/Neff

p_shift_1PN = np.sum(phi_shift_1PN)
p_shift_1PN = p_shift_1PN/Neff 


f = plt.figure(figsize=(16,14))

ax1 = f.add_subplot(111)
ax1.plot(N_arr, p_s_1PN - p_s_N, label = 'Total', alpha=0.9)
ax1.plot(N_arr, a_p1_1PN - a_p1_N, label = 'Standard effect 1PN vs. N', alpha=0.9)
ax1.plot(N_arr, (a_p2_1PN*a_p3_1PN) - (a_p2_N*a_p3_N), label = 'Planetary effect 1PN vs. N', alpha=0.9)
#ax3.plot(N_arr, a_p3, label = 'Coupling of Sun and other planets', alpha=0.9)
ax1.plot(N_arr, a_p4_1PN - a_p4_N, label = 'Gravitomagnetic effect 1PN vs. N', alpha=0.9)
ax1.set_xlabel('iterations', fontsize="x-large")
#ax1.set_xscale('log')
ax1.set_ylabel('Perihelion shift [rad/revolution]', fontsize="x-large")
#ax1.set_yscale('log')
ax1.grid()
plt.legend(fontsize="large") 

plt.show()

f = plt.figure(figsize=(16,14))

ax1 = f.add_subplot(111)
ax1.plot(N_arr, p_s_1PN, label = 'Total', alpha=0.9)
ax1.plot(N_arr, a_p1_1PN, label = 'Standard effect', alpha=0.9)
ax1.plot(N_arr, (a_p2_1PN*a_p3_1PN), label = 'Planetary effect', alpha=0.9)
#ax3.plot(N_arr, a_p3, label = 'Coupling of Sun and other planets', alpha=0.9)
ax1.plot(N_arr, a_p4_1PN, label = 'Gravitomagnetic effect', alpha=0.9)
ax1.set_xlabel('iterations', fontsize="x-large")
#ax1.set_xscale('log')
ax1.set_ylabel('Perihelion shift [rad/revolution]', fontsize="x-large")
#ax1.set_yscale('log')
ax1.grid()
plt.legend(fontsize="large") 

plt.show()


print('Newtonian order:\nGR standard shift = {} [rad/rev];\nCoupling with other planets shift = {} [rad/rev];\nGravitomagnetic shift = {} [rad/rev];\nTotal theorethical shift = {} [rad/rev];\nNumerical shift = {} [rad/rev];\nNumerical shift (test) = {} +- {} [rad/rev].\n'.format(np.sum(a_p1_N)/Neff, (np.sum(a_p2_N)/Neff)*(np.sum(a_p3_N)/Neff), np.sum(a_p4_N)/Neff, p_s_t_N, p_shift_N, phi_shift_test_N, D_shift_N))

print('1PN order:\n GR standard shift = {} [rad/rev];\nCoupling with other planets shift = {} [rad/rev];\nGravitomagnetic shift = {} [rad/rev];\nTotal theorethical shift = {} [rad/rev];\nNumerical shift = {} [rad/rev];\nNumerical shift (test) = {} +- {} [rad/rev].\n'.format(np.sum(a_p1_1PN)/Neff, (np.sum(a_p2_1PN)/Neff)*(np.sum(a_p3_1PN)/Neff), np.sum(a_p4_1PN)/Neff, p_s_t_1PN, p_shift_1PN, phi_shift_test_1PN, D_shift_1PN))

print("Shift difference (test): {} +- {} [rad/rev];\nShift difference: {} [rad/rev]".format(abs(phi_shift_test_1PN - phi_shift_test_N), abs(D_shift_N + D_shift_1PN), abs(p_shift_1PN - p_shift_N)))
