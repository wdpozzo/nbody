import numpy as np
from nbody.engine import run
import pickle

G = 6.67e-11 #*(u.meter**3)/(u.kilogram*u.second**2) # 6.67e-11 #

# AU**3/((d**2)*solMass) = (86400 * 86400) /( 2e30 * 1.5e11 * 1.5e11)

C = 299792458. #*(u.meter/u.second) #299792458. #
Ms = 1.988e30 #*(u.kilogram) # 1.988e30 #

Mmerc = 0.3301e24
Mearth = 5.9722e24 
AU = 149597870700. #*u.meter
Ms = 1.988e30


plot_step = 20
buffer_lenght = 10000000
data_thin = 40

ICN_it = 2

dt = 0.5
N  = 200000000
Neff = int(N/(data_thin*plot_step))
nout = int(N/buffer_lenght)    

    
'''
File to test the behaviour of the integrators implemented orders so far.

NB. --> the behaviour also depends on dt: the right solution will be achieved only if the dt is small enough
'''

#integration for various Newtonian orders
def energy_test(N, dt, m, x, y, z, px, py, pz, sx, sy, sz, nout):
	
	nbodies = len(m)
	
	run(N, np.longdouble(dt), 0, m, x, y, z, px, py, pz, sx, sy, sz, ICN_it, data_thin, buffer_lenght)

	s_N, H_N, T_N, V_N = [], [], [], []

	for i in range(nout):  
		s_tot, H_tot, T_tot, V_tot = [], [], [], []
		
		s_tot.append(pickle.load(open('solution_{}.pkl'.format(i),'rb')))
		H_tot.append(pickle.load(open('hamiltonian_{}.pkl'.format(i),'rb')))
		T_tot.append(pickle.load(open('kinetic_{}.pkl'.format(i),'rb')))
		V_tot.append(pickle.load(open('potential_{}.pkl'.format(i),'rb')))       
		
		s_N.append(s_tot[0][::plot_step])
		H_N.append(H_tot[0][::plot_step])
		T_N.append(T_tot[0][::plot_step])
		V_N.append(V_tot[0][::plot_step])
		
		del s_tot
		del H_tot
		del T_tot
		del V_tot
		
		if (1+i) % (10*nout)//100 == 0 :
			print("Data deframmentation: order 0 - {}%".format((100*i)/nout))
		
	s_N = np.array(s_N, dtype=object)#.flatten()
	H_N = np.array(H_N, dtype=object)#.flatten()
	T_N = np.array(T_N, dtype=object)#.flatten()
	V_N = np.array(V_N, dtype=object)#.flatten()
	
	s_N = np.concatenate((s_N[:]))
	H_N = np.concatenate((H_N[:]))
	T_N = np.concatenate((T_N[:]))
	V_N = np.concatenate((V_N[:])) 		
		    
	run(N, np.longdouble(dt), 1, m, x, y, z, px, py, pz, sx, sy, sz, ICN_it, data_thin, buffer_lenght)
	
	s_1PN, H_1PN, T_1PN, V_1PN = [], [], [], []

	for i in range(nout):  
		s_tot, H_tot, T_tot, V_tot = [], [], [], []
		
		s_tot.append(pickle.load(open('solution_{}.pkl'.format(i),'rb')))
		H_tot.append(pickle.load(open('hamiltonian_{}.pkl'.format(i),'rb')))
		T_tot.append(pickle.load(open('kinetic_{}.pkl'.format(i),'rb')))
		V_tot.append(pickle.load(open('potential_{}.pkl'.format(i),'rb')))       		
		
		s_1PN.append(s_tot[0][::plot_step])
		H_1PN.append(H_tot[0][::plot_step])
		T_1PN.append(T_tot[0][::plot_step])
		V_1PN.append(V_tot[0][::plot_step])	   
		
		del s_tot
		del H_tot
		del T_tot
		del V_tot
		
		if (1+i) % (10*nout)//100 == 0 :
			print("Data deframmentation: order 1 - {}%".format((100*i)/nout))			
		
	s_1PN = np.array(s_1PN, dtype=object)#.flatten()
	H_1PN = np.array(H_1PN, dtype=object)#.flatten()
	T_1PN = np.array(T_1PN, dtype=object)#.flatten()
	V_1PN = np.array(V_1PN, dtype=object)#.flatten()
	
	s_1PN = np.concatenate((s_1PN[:]))
	H_1PN = np.concatenate((H_1PN[:]))
	T_1PN = np.concatenate((T_1PN[:]))
	V_1PN = np.concatenate((V_1PN[:])) 
	
	'''	    
	run(N, np.longdouble(dt), 2, m, x, y, z, px, py, pz, sx, sy, sz, ICN_it, data_thin, buffer_lenght)
	
	s_2PN, H_2PN, T_2PN, V_2PN = [], [], [], []

	for i in range(nout):  
		s_tot, H_tot, T_tot, V_tot = [], [], [], []
		
		s_tot.append(pickle.load(open('solution_{}.pkl'.format(i),'rb')))
		H_tot.append(pickle.load(open('hamiltonian_{}.pkl'.format(i),'rb')))
		T_tot.append(pickle.load(open('kinetic_{}.pkl'.format(i),'rb')))
		V_tot.append(pickle.load(open('potential_{}.pkl'.format(i),'rb')))       
		
		s_2PN.append(s_tot[0][::plot_step])
		H_2PN.append(H_tot[0][::plot_step])
		T_2PN.append(T_tot[0][::plot_step])
		V_2PN.append(V_tot[0][::plot_step])
		
		del s_tot
		del H_tot
		del T_tot
		del V_tot
		
		if (1+i) % (10*nout)//100 == 0 :
			print("Data deframmentation: order 2 - {}%".format((100*i)/nout))
		
		s_2PN = np.array(s_2PN, dtype=object)#.flatten()
		H_2PN = np.array(H_2PN, dtype=object)#.flatten()
		T_2PN = np.array(T_2PN, dtype=object)#.flatten()
		V_2PN = np.array(V_2PN, dtype=object)#.flatten()
		
		s_2PN = np.concatenate((s_2PN[:]))
		H_2PN = np.concatenate((H_2PN[:]))
		T_2PN = np.concatenate((T_2PN[:]))
		V_2PN = np.concatenate((V_2PN[:])) 
	'''
		
	#Energies, normalized energies, radii and orbits in the different cases

	import matplotlib.pyplot as plt
	import matplotlib.cm as cm
	from mpl_toolkits import mplot3d
		    
	f = plt.figure(figsize=(6,4))
	ax = f.add_subplot(111)
	ax.plot(range(Neff), H_N, label= "Newtonian")
	ax.plot(range(Neff), H_1PN, label= "1PN")
	#ax.plot(range(Neff), H_2PN, label= "2PN")
	ax.set_xlabel('iteration')
	ax.set_ylabel('Hamiltonian')
	ax.grid()
	ax.legend()
	f.savefig('/home/FGeroni/Università/PROGETTI/Tesi/ThesisProject/LaTex/Immagini/SimulationsHamiltonian.pdf', bbox_inches='tight')

	h_n = []
	h_1pn = []
	#h_2pn = []      
		
	for i in range(0, Neff):
		h_n.append(T_N[i] + V_N[i])

		h_1pn.append(T_1PN[i] + V_1PN[i])

		#h_2pn.append(T_2PN[i] + V_2PN[i])

	f = plt.figure(figsize=(6,4))

	ax = f.add_subplot(1,3,1)
	ax.plot(range(Neff), V_N, label= "Newtonian")
	ax.plot(range(Neff), V_1PN, label= "1PN")
	#ax.plot(range(Neff), V_2PN, label= "2PN")
	ax.set_xlabel('iteration')
	ax.set_ylabel('Potential')
	ax.grid()
	ax.legend()
	#colors = iter(cm.rainbow(np.linspace(0, 1, nbodies)))

	ax2 = f.add_subplot(1,3,2)
	ax2.plot(range(Neff), T_N, label= "Newtonian")
	ax2.plot(range(Neff), T_1PN, label= "1PN")
	#ax2.plot(range(Neff), T_2PN, label= "2PN")
	ax2.set_xlabel('iteration')
	ax2.set_ylabel('Kinetic energy')
	ax2.grid()
	ax2.legend()
	#colors = iter(cm.rainbow(np.linspace(0, 1, nbodies)))

	ax3 = f.add_subplot(1,3,3)
	ax3.plot(range(Neff), h_n, label= "Newtonian")
	ax3.plot(range(Neff), h_1pn, label= "1PN")
	#ax3.plot(range(Neff), h_2pn, label= "2PN")
	ax3.set_xlabel('iteration')
	ax3.set_ylabel('Hamiltonian')
	ax3.grid()
	ax3.legend()
	#colors = iter(cm.rainbow(np.linspace(0, 1, nbodies)))

	f.savefig('/home/FGeroni/Università/PROGETTI/Tesi/ThesisProject/LaTex/Immagini/SimulationsHamiltonianNorm.pdf', bbox_inches='tight')

	print(len(T_N), len(V_N), len(T_N + V_N))

	H_1PN_N = []
	#H_2PN_N = []

	arr = []
	arr = [1 for i in range(Neff)] 

	for i in range(0, Neff):

		H_1PN_N.append(H_1PN[i]/H_N[i])    
		#H_2PN_N.append(H_2PN[i]/H_N[i])

	f = plt.figure(figsize=(6,4))
	ax = f.add_subplot(111)
	ax.plot(range(Neff), arr, label= "Newtonian")
	ax.plot(range(Neff), H_1PN_N, label= "1PN")
	#ax.plot(range(Neff), H_2PN_N, label= "2PN")
	ax.set_xlabel('iteration')
	ax.set_ylabel('Normalized Hamiltonian')
	ax.grid()
	ax.legend()
	#colors = iter(cm.rainbow(np.linspace(0, 1, nbodies)))
	f.savefig('/home/FGeroni/Università/PROGETTI/Tesi/ThesisProject/LaTex/Immagini/SimulationsHamiltonianNorm.pdf', bbox_inches='tight')

	#prepare data to evaluate the radius
	qs_N = [[] for x in range(nbodies)]
	qs_1PN = [[] for x in range(nbodies)]
	#qs_2PN = [[] for x in range(nbodies)]

	for i in range(0,Neff):

		for j in range(nbodies):
		    qs_N[j].append(s_N[i][j]['q'])
		    
		for j in range(nbodies): 
		    qs_1PN[j].append(s_1PN[i][j]['q'])
		    
		#for j in range(nbodies):
		    #qs_2PN[j].append(s_2PN[i][j]['q'])        

	q1_N = np.array(qs_N[0])
	q2_N = np.array(qs_N[1])  
	q_N = q1_N - q2_N

	q1_1PN = np.array(qs_1PN[0])
	q2_1PN = np.array(qs_1PN[1])  
	q_1PN = q1_1PN - q2_1PN
	
	'''
	q1_2PN = np.array(qs_2PN[0])
	q2_2PN = np.array(qs_2PN[1])  
	q_2PN = q1_2PN - q2_2PN
	'''
	
	r_N = []
	r_1PN = []
	#r_2PN = []      
		
	for i in range(0, Neff):
		r_N.append(np.sqrt(q_N[i,0]**2 + q_N[i,1]**2 + q_N[i,2]**2))

		r_1PN.append(np.sqrt(q_1PN[i,0]**2 + q_1PN[i,1]**2 + q_1PN[i,2]**2))

		#r_2PN.append(np.sqrt(q_2PN[i,0]**2 + q_2PN[i,1]**2 + q_2PN[i,2]**2))


	f = plt.figure(figsize=(6,4))
	ax = f.add_subplot(111)
	ax.plot(range(Neff), r_N, label= "Newtonian")
	ax.plot(range(Neff), r_1PN, label= "1PN")
	#ax.plot(range(Neff), r_2PN, label= "2PN")
	ax.set_xlabel('iteration')
	ax.set_ylabel('Orbital radius')
	ax.grid()
	ax.legend()
	#colors = iter(cm.rainbow(np.linspace(0, 1, nbodies)))
	f.savefig('/home/FGeroni/Università/PROGETTI/Tesi/ThesisProject/LaTex/Immagini/SimulationsOrbRadius.pdf', bbox_inches='tight')

	f = plt.figure(figsize=(6,4))
	ax = f.add_subplot(111, projection = '3d')
	#ax.title(r'$m_1 =$ {}, $m_2 =$ {}'.format(m[0], m[1]))
	#colors = cm.rainbow(np.linspace(0, 1, nbodies[0]))    
	ax.plot(q_N[:,0], q_N[:,1], q_N[:,2], label= "Newtonian", alpha=1, lw=1)
	ax.plot(q_1PN[:,0], q_1PN[:,1], q_1PN[:,2], label= "1PN", alpha=1, lw=1)
	#ax.plot(q_2PN[:,0], q_2PN[:,1], q_2PN[:,2], label= "2PN", alpha=1, lw=1)
	ax.set_xlabel('x [km]')
	ax.set_ylabel('y [km]')
	ax.set_zlabel('z [km]')
	ax.legend()

	f.savefig('/home/FGeroni/Università/PROGETTI/Tesi/ThesisProject/LaTex/Immagini/SimulationsOrbits.pdf', bbox_inches='tight')

	plt.show()
	
	return 
	

#custom initial coordinates
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

'''
m[0], m[1] = 1.e0*Mmerc, 1.e0*Ms

x[0], x[1] = -69.818e9, 0.*AU
y[0], y[1] = 0.*AU, 0.*AU
z[0], z[1] = 0.0*AU, 0.0*AU

vx[0], vx[1] = 0., 0.
vy[0], vy[1] = 38.86e3, 0.
vz[0], vz[1] = 0., 0.

sx[0], sx[1] = 0., 0.
sy[0], sy[1] = 0., 0.
sz[0], sz[1] = 0., 0.
'''

m[0], m[1] = 4.e0*Ms, 1.e0*Ms

x[0], x[1] = -0.9*AU, 0.9*AU
y[0], y[1] = 0.*AU, 0.*AU
z[0], z[1] = 0.*AU, 0.*AU

vx[0], vx[1] = +0.7e0, -1.2e0
vy[0], vy[1] = +1.86e1, -2.56e1
vz[0], vz[1] = 0., 0.

sx[0], sx[1] = 0., 0.
sy[0], sy[1] = 0., 0.
sz[0], sz[1] = 0., 0.

energy_test(N, dt, m, x, y, z, m*vx, m*vy, m*vz, sx, sy, sz, nout)

