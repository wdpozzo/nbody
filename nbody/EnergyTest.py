import numpy as np
from nbody.engine import run
import pickle

'''
File to test the behaviour of the integrator for all implemented orders so far.

NB. --> the behaviour also depends on dt: the right dynamic will be achieved only if the dt is small enough
'''

#hand compile (for now) with the charateristichs of the simulation
N = 3000000
dt = 0.2

Neff = N//10
nbodies = 2

#creating data arrays
m = np.array((2,1)).astype(np.longdouble)

x = np.array((2,1)).astype(np.longdouble)
y = np.array((2,1)).astype(np.longdouble)
z = np.array((2,1)).astype(np.longdouble)

vx = np.array((2,1)).astype(np.longdouble)
vy = np.array((2,1)).astype(np.longdouble)
vz = np.array((2,1)).astype(np.longdouble)
    
sx = np.array((2,1)).astype(np.longdouble)
sy = np.array((2,1)).astype(np.longdouble)
sz = np.array((2,1)).astype(np.longdouble)

#set a seed if needed
if 1:
    np.random.seed(39)

#initial conditions (randomized)
m = np.random.uniform(1e-3,1e-1,size = nbodies).astype(np.longdouble)

x = np.random.uniform(-200.0,200.0,size = nbodies).astype(np.longdouble)
y = np.random.uniform(-200.0,200.0,size = nbodies).astype(np.longdouble)
z = np.random.uniform(-200.0,200.0,size = nbodies).astype(np.longdouble)

vx = np.random.uniform(-0.001,0.001,size = nbodies).astype(np.longdouble)
vy = np.random.uniform(-0.001,0.001,size = nbodies).astype(np.longdouble)
vz = np.random.uniform(-0.001,0.001,size = nbodies).astype(np.longdouble)

sx = np.random.uniform(-1.0,1.0,size = nbodies).astype(np.longdouble)
sy = np.random.uniform(-1.0,1.0,size = nbodies).astype(np.longdouble)
sz = np.random.uniform(-1.0,1.0,size = nbodies).astype(np.longdouble)

#print(m,x,y,z,vx,vy,vz,sx,sy,sz)

'''
#initial conditions (change manually, care to be coherent)
m[0], m[1] = 10., 1e-4 #Msun*(10e6), 10*Msun

x[0], x[1] = 500., -800.
y[0], y[1] = -800., 500.
z[0], z[1] = 0., 0.

vx[0], vx[1] = 1e-5, -1e-3
vy[0], vy[1] = 5e-5, -2e-2
vz[0], vz[1] = 0., 0.

sx[0], sx[1] = 0., 0.
sy[0], sy[1] = 0., 0.
sz[0], sz[1] = 0., 0.
'''

#integration for various Newtonian orders

s_N,H_N = run(N, np.longdouble(dt), 0, m, x, y, z, m*vx, m*vy, m*vz, sx, sy, sz)
s_N = np.array(s_N, dtype=object)
pickle.dump(s_N, open('solution.p','wb'))
pickle.dump(H_N, open('hamiltonian.p','wb'))
        
s_1PN,H_1PN = run(N, np.longdouble(dt), 1, m, x, y, z, m*vx, m*vy, m*vz, sx, sy, sz)
s_1PN   = np.array(s_1PN, dtype=object)
pickle.dump(s_1PN, open('solution.p','wb'))
pickle.dump(H_1PN, open('hamiltonian.p','wb'))
        
s_2PN,H_2PN = run(N, np.longdouble(dt), 2, m, x, y, z, m*vx, m*vy, m*vz, sx, sy, sz)
s_2PN   = np.array(s_2PN, dtype=object)
pickle.dump(s_2PN, open('solution.p','wb'))
pickle.dump(H_2PN, open('hamiltonian.p','wb'))

#Energies, normalized energies, radii and orbits in the different cases

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits import mplot3d
        
f = plt.figure(figsize=(6,4))
ax = f.add_subplot(111)
ax.plot(range(Neff), H_N, label= "Newtonian")
ax.plot(range(Neff), H_1PN, label= "1PN")
ax.plot(range(Neff), H_2PN, label= "2PN")
ax.set_xlabel('iteration')
ax.set_ylabel('Hamiltonian')
ax.grid()
ax.legend()
#colors = iter(cm.rainbow(np.linspace(0, 1, nbodies)))
f.savefig('/home/FGeroni/Università/PROGETTI/Tesi/ThesisProject/LaTex/Immagini/SimulationsHamiltonian.pdf', bbox_inches='tight')

H_1PN_N = []
H_2PN_N = []

arr=[]
arr = [1 for i in range(Neff)] 

for i in range(0, Neff):

    H_1PN_N.append(H_1PN[i]/H_N[i])    
    H_2PN_N.append(H_2PN[i]/H_N[i])

f = plt.figure(figsize=(6,4))
ax = f.add_subplot(111)
ax.plot(range(Neff), arr, label= "Newtonian")
ax.plot(range(Neff), H_1PN_N, label= "1PN")
ax.plot(range(Neff), H_2PN_N, label= "2PN")
ax.set_xlabel('iteration')
ax.set_ylabel('Normalized Hamiltonian')
ax.grid()
ax.legend()
#colors = iter(cm.rainbow(np.linspace(0, 1, nbodies)))
f.savefig('/home/FGeroni/Università/PROGETTI/Tesi/ThesisProject/LaTex/Immagini/SimulationsHamiltonianNorm.pdf', bbox_inches='tight')

#prepare data to evaluate the radius
qs_N = [[] for x in range(nbodies)]
qs_1PN = [[] for x in range(nbodies)]
qs_2PN = [[] for x in range(nbodies)]

for i in range(0,Neff):

    for j in range(nbodies):
        qs_N[j].append(s_N[i][j]['q'])
        
    for j in range(nbodies): 
        qs_1PN[j].append(s_1PN[i][j]['q'])
        
    for j in range(nbodies):
        qs_2PN[j].append(s_2PN[i][j]['q'])        

q1_N = np.array(qs_N[0])
q2_N = np.array(qs_N[1])  
q_N = q1_N - q2_N

q1_1PN = np.array(qs_1PN[0])
q2_1PN = np.array(qs_1PN[1])  
q_1PN = q1_1PN - q2_1PN

q1_2PN = np.array(qs_2PN[0])
q2_2PN = np.array(qs_2PN[1])  
q_2PN = q1_2PN - q2_2PN

r_N = []
r_1PN = []
r_2PN = []      
    
for i in range(0, Neff):
    r_N.append(np.sqrt(q_N[i,0]**2 + q_N[i,1]**2 + q_N[i,2]**2))

    r_1PN.append(np.sqrt(q_1PN[i,0]**2 + q_1PN[i,1]**2 + q_1PN[i,2]**2))

    r_2PN.append(np.sqrt(q_2PN[i,0]**2 + q_2PN[i,1]**2 + q_2PN[i,2]**2))


f = plt.figure(figsize=(6,4))
ax = f.add_subplot(111)
ax.plot(range(Neff), r_N, label= "Newtonian")
ax.plot(range(Neff), r_1PN, label= "1PN")
ax.plot(range(Neff), r_2PN, label= "2PN")
ax.set_xlabel('iteration')
ax.set_ylabel('Orbital radius')
ax.grid()
ax.legend()
#colors = iter(cm.rainbow(np.linspace(0, 1, nbodies)))
f.savefig('/home/FGeroni/Università/PROGETTI/Tesi/ThesisProject/LaTex/Immagini/SimulationsOrbRadius.pdf', bbox_inches='tight')

f = plt.figure(figsize=(6,4))
ax = f.add_subplot(111, projection = '3d')
#colors = cm.rainbow(np.linspace(0, 1, nbodies[0]))    
ax.plot(q_N[:,0], q_N[:,1], q_N[:,2], label= "Newtonian", alpha=1, lw=1)
ax.plot(q_1PN[:,0], q_1PN[:,1], q_1PN[:,2], label= "1PN", alpha=1, lw=1)
ax.plot(q_2PN[:,0], q_2PN[:,1], q_2PN[:,2], label= "2PN", alpha=1, lw=1)
ax.set_xlabel('x [km]')
ax.set_ylabel('y [km]')
ax.set_zlabel('z [km]')
ax.legend()

f.savefig('/home/FGeroni/Università/PROGETTI/Tesi/ThesisProject/LaTex/Immagini/SimulationsOrbits.pdf', bbox_inches='tight')

plt.show()
