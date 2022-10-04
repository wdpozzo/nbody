import numpy as np
from nbody.engine import run, _H_2body
import pickle
from Kep_dynamic import kepler, kepler_sol_sys
from nbody.CM_coord_system import CM_system

import astropy.units as u
from datetime import datetime
from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, get_body_barycentric_posvel

G = 6.67e-11 #*(u.meter**3)/(u.kilogram*u.second**2) # 6.67e-11 #

#AU**3/((d**2)*solMass) = (86400 * 86400) /( 2e30 * 1.5e11 * 1.5e11)

C = 299792458. #*(u.meter/u.second) #299792458. #
Ms = 1.988e30 #*(u.kilogram) # 1.988e30 #

Mmerc = 0.3301e24
Mearth = 5.9722e24 
AU = 149597870700. #*u.meter
Ms = 1.988e30

plot_step = 100000
buffer_lenght = 2000000
data_thin = 5

ICN_it = 2

dt = 0.05
N  = 1500000000
p = 0

Neff = int(N/(data_thin*plot_step))
nout = int(N/buffer_lenght)    
  
N_arr = np.linspace(0, N, Neff)
  
'''
File to test the behaviour of the integrators implemented so far.

NB. --> the behaviour also depends on dt: the right solution will be achieved only if the dt is small enough
'''

#integration for various Newtonian orders
def energy_test(N, dt, m, x, y, z, px, py, pz, sx, sy, sz, nout, planet_names):
	
	nbodies = len(m)
	
	if (p == 0):
		run(N, np.longdouble(dt), 0, m, x, y, z, px, py, pz, sx, sy, sz, ICN_it, data_thin, buffer_lenght)

	s_N, H_N, T_N, V_N = [], [], [], []

	for i in range(nout):  
		s_tot, H_tot, T_tot, V_tot = [], [], [], []
		
		s_tot.append(pickle.load(open('solution_{}_order{}.pkl'.format(i, 0),'rb')))
		H_tot.append(pickle.load(open('hamiltonian_{}_order{}.pkl'.format(i, 0),'rb')))
		T_tot.append(pickle.load(open('kinetic_{}_order{}.pkl'.format(i, 0),'rb')))
		V_tot.append(pickle.load(open('potential_{}_order{}.pkl'.format(i, 0),'rb')))	

		s_N.append(s_tot[0][::plot_step])
		H_N.append(H_tot[0][::plot_step])
		T_N.append(T_tot[0][::plot_step])
		V_N.append(V_tot[0][::plot_step])

		del s_tot
		del H_tot
		del T_tot
		del V_tot
		
		index_0 = i*100/nout 
		if (index_0) % 10 == 0 :
			print("Data deframmentation: order 0 - {}%".format(index_0))

		
	s_N = np.array(s_N, dtype=object)#.flatten()
	H_N = np.array(H_N, dtype=object)#.flatten()
	T_N = np.array(T_N, dtype=object)#.flatten()
	V_N = np.array(V_N, dtype=object)#.flatten()
	
	s_N = np.concatenate((s_N[:]))
	H_N = np.concatenate((H_N[:]))
	T_N = np.concatenate((T_N[:]))
	V_N = np.concatenate((V_N[:])) 		

	if (p == 0):		    
		run(N, np.longdouble(dt), 1, m, x, y, z, px, py, pz, sx, sy, sz, ICN_it, data_thin, buffer_lenght)
	
	s_1PN, H_1PN, T_1PN, V_1PN = [], [], [], []
        
	for i in range(nout):  
		s_tot, H_tot, T_tot, V_tot = [], [], [], []
		
		s_tot.append(pickle.load(open('solution_{}_order{}.pkl'.format(i, 1),'rb')))
		H_tot.append(pickle.load(open('hamiltonian_{}_order{}.pkl'.format(i,1),'rb')))
		T_tot.append(pickle.load(open('kinetic_{}_order{}.pkl'.format(i, 1),'rb')))
		V_tot.append(pickle.load(open('potential_{}_order{}.pkl'.format(i, 1),'rb')))       		

		s_1PN.append(s_tot[0][::plot_step])
		H_1PN.append(H_tot[0][::plot_step])
		T_1PN.append(T_tot[0][::plot_step])
		V_1PN.append(V_tot[0][::plot_step])
		

		del s_tot
		del H_tot
		del T_tot
		del V_tot

		index_1 = i*100/nout 
		if (index_1) % 10 == 0 :
			print("Data deframmentation: order 1 - {}%".format(index_1))			
		
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
	#print(H_N, T_N, V_N)
	
	#Numerical PN-order confrontation 

	import matplotlib.pyplot as plt
	import matplotlib.cm as cm
	from mpl_toolkits import mplot3d
		    
	f = plt.figure(figsize=(6,4))

	ax = f.add_subplot(1,3,1)
	ax.plot(N_arr, V_N, label= "Newtonian")
	ax.plot(N_arr, V_1PN, label= "1PN")
	#ax.plot(range(Neff), V_2PN, label= "2PN")
	ax.set_xlabel('iteration')
	ax.set_ylabel('Potential')
	ax.grid()
	ax.legend()
	#colors = iter(cm.rainbow(np.linspace(0, 1, nbodies)))

	ax2 = f.add_subplot(1,3,2)
	ax2.plot(N_arr, T_N, label= "Newtonian")
	ax2.plot(N_arr, T_1PN, label= "1PN")
	#ax2.plot(range(Neff), T_2PN, label= "2PN")
	ax2.set_xlabel('iteration')
	ax2.set_ylabel('Kinetic energy')
	ax2.grid()
	ax2.legend()
	#colors = iter(cm.rainbow(np.linspace(0, 1, nbodies)))

	ax3 = f.add_subplot(1,3,3)
	ax3.plot(N_arr, H_N, label= "Newtonian")
	ax3.plot(N_arr, H_1PN, label= "1PN")
	#ax.plot(range(Neff), H_2PN, label= "2PN")
	ax3.set_xlabel('iteration')
	ax3.set_ylabel('Hamiltonian')
	ax3.grid()
	ax3.legend()

	plt.show()

	#f.savefig('/home/FGeroni/Università/PROGETTI/Tesi/ThesisProject/LaTex/Immagini/SimulationsHamiltonianNorm.pdf', bbox_inches='tight')

	H_1PN_N = []
	#H_2PN_N = []

	arr = []
	arr = [1 for i in range(Neff)] 

	for i in range(0, Neff):

		H_1PN_N.append(np.sign(H_1PN[i])*(H_1PN[i]/H_N[i]))    
		#H_2PN_N.append(H_2PN[i]/H_N[i])

	f = plt.figure(figsize=(8,6))
	
	ax = f.add_subplot(111)
	ax.plot(N_arr, arr, label= "Newtonian")
	ax.plot(N_arr, H_1PN_N, label= "1PN")
	#ax.plot(range(Neff), H_2PN_N, label= "2PN")
	ax.set_xlabel('iteration')
	ax.set_ylabel(r'$H/H_{N}$')
	ax.grid()
	ax.legend()
	plt.show()
	#colors = iter(cm.rainbow(np.linspace(0, 1, nbodies)))
	#f.savefig('/home/FGeroni/Università/PROGETTI/Tesi/ThesisProject/LaTex/Immagini/SimulationsHamiltonianNorm.pdf', bbox_inches='tight')
	
	#prepare data to evaluate the radius
	q_N = np.array([[[0 for i in range(0, 3)] for Neff in range(0, Neff)] for m in range(0,len(m))], dtype='longdouble')
	p_N = np.array([[[0 for i in range(0, 3)] for Neff in range(0, Neff)] for m in range(0,len(m))], dtype='longdouble')
	spn_N = np.array([[[0 for i in range(0, 3)] for Neff in range(0, Neff)] for m in range(0,len(m))], dtype='longdouble')
	
	q_1PN = np.array([[[0 for i in range(0, 3)] for Neff in range(0, Neff)] for m in range(0,len(m))], dtype='longdouble')
	p_1PN = np.array([[[0 for i in range(0, 3)] for Neff in range(0, Neff)] for m in range(0,len(m))], dtype='longdouble')
	spn_1PN = np.array([[[0 for i in range(0, 3)] for Neff in range(0, Neff)] for m in range(0,len(m))], dtype='longdouble')
	
	for j in range(nbodies):
	
		for i in range(0, Neff):

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
			
		#for j in range(nbodies):
		    #qs_2PN[j].append(s_2PN[i][j]['q'])     

	#2 body case	       
	if (nbodies==2): 

		#Newtonian order quantities
		q_N_rel, p_N_rel, q_N_cm, p_N_cm = CM_system(p_N[0], p_N[1], q_N[0], q_N[1], Neff, m[0], m[1])
		
		r_dif_N, q_an_rel_N, r_kepler_N, L_N, a_p_N, P_quad_N, phi_shift_N, q_peri_N, peri_indexes_N, phi_shift_test_N = kepler(q_N[0], q_N[1], p_N[0], p_N[1], N, Neff, H_N, m, dt, 0)

		#1PN order quantities
		q_1PN_rel, p_1PN_rel, q_1PN_cm, p_1PN_cm = CM_system(p_1PN[0], p_1PN[1], q_1PN[0], q_1PN[1], Neff, m[0], m[1])
		
		r_dif_1PN, q_an_rel_1PN, r_kepler_1PN, L_1PN, a_p_1PN, P_quad_1PN, phi_shift_1PN, q_peri_1PN, peri_indexes_1PN, phi_shift_test_1PN = kepler(q_1PN[0], q_1PN[1], p_1PN[0], p_1PN[1], N, Neff, H_N, m, dt, 1)
		

		#plots	
		r_N = []
		r_1PN = []
		#r_2PN = []      
			

		r_N = np.sqrt(q_N_rel[:,0]*q_N_rel[:,0] + q_N_rel[:,1]*q_N_rel[:,1]+ q_N_rel[:,2]*q_N_rel[:,2])
		r_1PN = np.sqrt(q_1PN_rel[:,0]*q_1PN_rel[:,0] + q_1PN_rel[:,1]*q_1PN_rel[:,1] + q_1PN_rel[:,2]*q_1PN_rel[:,2])

			#r_2PN.append(np.sqrt(q_2PN[i,0]**2 + q_2PN[i,1]**2 + q_2PN[i,2]**2))

		#r_N_merc = np.sqrt(q_N[1,:,0]*q_N[1,:,0] + q_N[1,:,1]*q_N[1,:,1]+ q_N[1,:,2]*q_N[1,:,2])
		#r_1PN_merc = np.sqrt(q_1PN[1,:,0]*q_1PN[1,:,0] + q_1PN[1,:,1]*q_1PN[1,:,1] + q_1PN[1,:,2]*q_1PN[1,:,2])

		f = plt.figure(figsize=(6,4))
		ax = f.add_subplot(111)
		ax.plot(N_arr, abs(r_N - r_1PN), label= "N orbit vs. 1PN orbit")
		#ax.plot(range(Neff), r_2PN, label= "2PN")
		ax.set_xlabel('iteration')
		ax.set_ylabel('Orbital radius difference [m]')
		ax.grid()
		ax.legend()
		#colors = iter(cm.rainbow(np.linspace(0, 1, nbodies)))
		#f.savefig('/home/FGeroni/Università/PROGETTI/Tesi/ThesisProject/LaTex/Immagini/SimulationsOrbRadius.pdf', bbox_inches='tight')
		plt.show()



		f = plt.figure(figsize=(16,10))

		ax1 = f.add_subplot(131)
		ax1.scatter(N_arr, abs(q_N[1,:,0] - q_1PN[1,:,0]), label = 'Newtonian vs. 1PN')
		ax1.set_xlabel('iteration')
		ax1.set_ylabel(r'$\Delta x$ [m]')
		plt.grid()
		plt.legend()

		ax2 = f.add_subplot(132)
		ax2.scatter(N_arr, abs(q_N[1,:,1] - q_1PN[1,:,1]), label = 'Newtonian vs. 1PN')
		ax2.set_xlabel('iteration')
		ax2.set_ylabel(r'$\Delta y$ [m]')
		plt.grid()
		plt.legend()

		ax3 = f.add_subplot(133)
		ax3.scatter(N_arr, abs(q_N[1,:,2] - q_1PN[1,:,2]), label = 'Newtonian vs. 1PN')
		ax3.set_xlabel('iteration')
		ax3.set_ylabel(r'$\Delta z$ [m]')
		plt.grid()
		plt.legend()

		plt.show()       


		f = plt.figure(figsize=(6,4))
		ax = f.add_subplot(1,2,1)
		ax.plot(N_arr, L_N - L_1PN, label= 'Newtonian vs. 1PN')
		#ax.plot(range(Neff), r_2PN, label= "2PN")
		ax.set_xlabel('iteration')
		ax.set_ylabel('Angolar momentum')
		ax.grid()
		ax.legend()
		#colors = iter(cm.rainbow(np.linspace(0, 1, nbodies)))
		#f.savefig('/home/FGeroni/Università/PROGETTI/Tesi/ThesisProject/LaTex/Immagini/SimulationsOrbRadius.pdf', bbox_inches='tight')
	
		ax = f.add_subplot(1,2,2)
		ax.plot(N_arr, P_quad_N, label= "Quadrupole radiation (N)")
		ax.plot(N_arr, P_quad_1PN, label= "Quadrupole radiation (1PN)")
		#ax.plot(range(Neff), r_2PN, label= "2PN")
		ax.set_xlabel('iteration')
		ax.set_ylabel('Power emission [J/s]')
		ax.grid()
		ax.legend()		
		
		plt.show()
	
		

		p_s_N = np.sum(phi_shift_N)
		p_s_N = p_s_N/Neff 
		
		a_shift_N = np.sum(a_p_N)
		a_shift_N = a_shift_N/Neff 
		
		p_s_1PN = np.sum(phi_shift_1PN)
		p_s_1PN = p_s_1PN/Neff 
		
		a_shift_1PN = np.sum(a_p_1PN)
		a_shift_1PN = a_shift_1PN/Neff 
		
		
		print("Standard GR shift: {} [rad/rev]".format(a_shift_N))
		print("Shift difference: {} [rad/rev]".format(abs(p_s_1PN - p_s_N)))
		print("Shift difference (test): {} [rad/rev]".format(abs(phi_shift_test_1PN - phi_shift_test_N)))

		
	#N body case	   		
	if (nbodies > 2):

		H_2body_N = []
		V_2body_N = []
		T_2body_N = []
		
		H_2body_1PN = []
		V_2body_1PN = []
		T_2body_1PN = []    
		
		for i in range(0, Neff):
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
		
		#print(mass,x,y,z,px,py,pz,sx,sy,sz)        
		mass = np.array([m[0], m[1]]).astype(np.longdouble)
		 
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
		
		L_N, P_quad_N, a_p1_N, a_p2_N, a_p3_N, a_p4_N, q_peri_N, phi_shift_N, phi_shift_test_N = kepler_sol_sys(p_N, q_N, Neff, H_2body_N, m, dt, 0)

		#coordinate 1PN
		q_1PN_rel, p_1PN_rel, q_1PN_cm, p_1PN_cm = CM_system(p_1PN[0], p_1PN[1], q_1PN[0], q_1PN[1], Neff, m[0], m[1])
		
		L_1PN, P_quad_1PN, a_p1_1PN, a_p2_1PN, a_p3_1PN, a_p4_1PN, q_peri_1PN, phi_shift_1PN, phi_shift_test_1PN = kepler_sol_sys(p_1PN, q_1PN, Neff, H_2body_1PN, m, dt, 1)
		
		r_N = []
		r_1PN = []
		#r_2PN = []      

		r_N = np.sqrt(q_N_rel[:,0]*q_N_rel[:,0] + q_N_rel[:,1]*q_N_rel[:,1] + q_N_rel[:,2]*q_N_rel[:,2])
		r_1PN = np.sqrt(q_1PN_rel[:,0]*q_1PN_rel[:,0] + q_1PN_rel[:,1]*q_1PN_rel[:,1] + q_1PN_rel[:,2]*q_1PN_rel[:,2])
			
		col_rainbow = cm.rainbow(np.linspace(0, 1, len(masses)))   

		#PLOTS
		f = plt.figure(figsize=(18,18))
		ax = f.add_subplot(111, projection = '3d')
		for k in range(len(m)):  

			if (k<2):
				ax.plot(q_N[k,:,0], q_N[k,:,1], q_N[k,:,2], label= "{} (N)".format(planet_names[k]), alpha=1, color = col_rainbow[k])
				ax.plot(q_1PN[k,:,0], q_1PN[k,:,1], q_1PN[k,:,2], label= "{} (1PN)".format(planet_names[k]), alpha=1, color = col_rainbow[k])
			else :
				ax.plot(q_N[k,:,0], q_N[k,:,1], q_N[k,:,2], label= "{} (N)".format(planet_names[k]), alpha=0.5, color = col_rainbow[k])
				ax.plot(q_1PN[k,:,0], q_1PN[k,:,1], q_1PN[k,:,2], label= "{} (1PN)".format(planet_names[k]), alpha=0.5, color = col_rainbow[k])

		ax.set_xlabel('x [km]')
		ax.set_ylabel('y [km]')
		ax.set_zlabel('z [km]')
		ax.legend()
		#f.savefig('/home/FGeroni/Università/PROGETTI/Tesi/ThesisProject/LaTex/Immagini/SimulationsOrbits.pdf', bbox_inches='tight')
		plt.show()


		f = plt.figure(figsize=(16,10))

		ax1 = f.add_subplot(131)
		for k in range(len(m)):  

			if (k<2):
				ax1.scatter(N_arr, abs(q_N[k,:,0] - q_1PN[k,:,0]), label= "{} orbit difference".format(planet_names[k]), alpha=1, color = col_rainbow[k])
			else :
				ax1.scatter(N_arr, abs(q_N[k,:,0] - q_1PN[k,:,0]), label= "{} orbit difference".format(planet_names[k]), alpha=0.5, color = col_rainbow[k])
		ax1.set_xlabel('iteration')
		ax1.set_ylabel(r'$\Delta x$ [m]')
		plt.grid()
		plt.legend()

		ax2 = f.add_subplot(132)
		for k in range(len(m)):  

			if (k<2):
				ax2.scatter(N_arr, abs(q_N[k,:,1] - q_1PN[k,:,1]), label= "{} orbit difference".format(planet_names[k]), alpha=1, color = col_rainbow[k])
			else :
				ax2.scatter(N_arr, abs(q_N[k,:,1] - q_1PN[k,:,1]), label= "{} orbit difference".format(planet_names[k]), alpha=0.5, color = col_rainbow[k])
		ax2.set_xlabel('iteration')
		ax2.set_ylabel(r'$\Delta y$ [m]')
		plt.grid()
		plt.legend()

		ax3 = f.add_subplot(133)
		for k in range(len(m)):  

			if (k<2):
				ax3.scatter(N_arr, abs(q_N[k,:,2] - q_1PN[k,:,2]), label= "{} orbit difference".format(planet_names[k]), alpha=1, color = col_rainbow[k])
			else :
				ax3.scatter(N_arr, abs(q_N[k,:,2] - q_1PN[k,:,2]), label= "{} orbit difference".format(planet_names[k]), alpha=0.5, color = col_rainbow[k])
		ax3.set_xlabel('iteration')
		ax3.set_ylabel(r'$\Delta z$ [m]')
		plt.grid()
		plt.legend()

		plt.show()  


		f = plt.figure(figsize=(14,8))	
		ax1 = f.add_subplot(1,1,1)
		ax1.plot(N_arr, abs(r_N - r_1PN), label= "N orbit vs. 1PN orbit")
		#ax.plot(range(Neff), H_2PN_N, label= "2PN")
		ax1.set_xlabel('iteration')
		ax1.set_ylabel('Orbital displacement [m]')
		ax1.grid()
		ax1.legend()	
		plt.show()	


		f = plt.figure(figsize=(18,12))
		
		ax1 = f.add_subplot(1,2,1)
		ax1.plot(N_arr, H_N, label= "Hamiltonian (N)")
		ax1.plot(N_arr, H_1PN, label= "Hamiltonian (1PN)")
		#ax.plot(range(Neff), H_2PN_N, label= "2PN")
		ax1.set_xlabel('iteration')
		ax1.set_ylabel('Energy [J]')
		#ax1.invert_yaxis()
		ax1.grid()
		ax1.legend()
		#colors = iter(cm.rainbow(np.linspace(0, 1, nbodies)))
		#f.savefig('/home/FGeroni/Università/PROGETTI/Tesi/ThesisProject/LaTex/Immagini/SimulationsOrbRadius.pdf', bbox_inches='tight')

		ax2 = f.add_subplot(1,2,2)
		ax2.plot(N_arr, P_quad_N, label= "Quadrupole radiation (N)")
		ax2.plot(N_arr, P_quad_1PN, label= "Quadrupole radiation (1PN)")
		#ax.plot(range(Neff), r_2PN, label= "2PN")
		ax2.set_xlabel('iteration')
		ax2.set_ylabel('Power emission [J/s]')
		ax2.grid()
		ax2.legend()		

		plt.show()
		
		col_rainbow = cm.rainbow(np.linspace(0, 1, len(q_peri_N)))    
		col_viridis = cm.viridis(np.linspace(0, 1, len(q_peri_1PN)))  

		f = plt.figure(figsize=(16,6))
		ax = f.add_subplot(111, projection = '3d')
		#ax.plot(q_N[1,:,0], q_N[1,:,1], q_N[1,:,2], label = 'Numerical solution', alpha=0.5)
		#ax.plot(q_1PN[1,:,0], q_1PN[1,:,1], q_1PN[1,:,2], label = 'Numerical solution', alpha=0.5)
		for i in range(0, len(q_peri_N)): 
			ax.plot(q_peri_N[i,0], q_peri_N[i,1], q_peri_N[i,2], 'o', label = 'Perihelion orbit {} (N)'.format(i), color = col_rainbow[i])
		for i in range(0, len(q_peri_1PN)): 
			ax.plot(q_peri_1PN[i,0], q_peri_1PN[i,1], q_peri_1PN[i,2], 'o', label = 'Perihelion orbit {} (1PN)'.format(i), color = col_viridis[i])
		ax.set_xlabel('x [m]')
		ax.set_ylabel('y [m]')
		ax.set_zlabel('z [m]')
		plt.legend()
		plt.show()   

		f = plt.figure(figsize=(16,10))

		ax1 = f.add_subplot(131)
		ax1.scatter(np.linspace(0, N, len(q_peri_N)), abs(q_peri_N[:,0] - q_peri_1PN[:,0]), alpha=1)
		ax1.set_xlabel('iteration')
		ax1.set_ylabel(r'$\Delta x$ [m]')
		plt.grid()
		plt.legend()

		ax2 = f.add_subplot(132)
		ax2.scatter(np.linspace(0, N, len(q_peri_N)), abs(q_peri_N[:,1] - q_peri_1PN[:,1]), alpha=1)
		ax2.set_xlabel('iteration')
		ax2.set_ylabel(r'$\Delta y$ [m]')
		plt.grid()
		plt.legend()

		ax3 = f.add_subplot(133)
		ax3.scatter(np.linspace(0, N, len(q_peri_N)), abs(q_peri_N[:,2] - q_peri_1PN[:,2]), alpha=1)
		ax3.set_xlabel('iteration')
		ax3.set_ylabel(r'$\Delta z$ [m]')
		plt.grid()
		plt.legend()

		plt.show()  

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
		
		
		print('Newtonian order:\nGR standard shift = {} [rad/rev];\nCoupling with other planets shift = {} [rad/rev];\nGravitomagnetic shift = {} [rad/rev];\nTotal theorethical shift = {} [rad/rev];\nNumerical shift = {} [rad/rev];\nNumerical shift (test) = {} [rad/rev].\n'.format(np.sum(a_p1_N)/Neff, (np.sum(a_p2_N)/Neff)*(np.sum(a_p3_N)/Neff), np.sum(a_p4_N)/Neff, p_s_t_N, p_shift_N, phi_shift_test_N))
		
		print('1PN order:\n GR standard shift = {} [rad/rev];\nCoupling with other planets shift = {} [rad/rev];\nGravitomagnetic shift = {} [rad/rev];\nTotal theorethical shift = {} [rad/rev];\nNumerical shift = {} [rad/rev];\nNumerical shift (test) = {} [rad/rev].\n'.format(np.sum(a_p1_1PN)/Neff, (np.sum(a_p2_1PN)/Neff)*(np.sum(a_p3_1PN)/Neff), np.sum(a_p4_1PN)/Neff, p_s_t_1PN, p_shift_1PN, phi_shift_test_1PN))
		
		print("Shift difference (test): {} [rad/rev];\nShift difference: {} [rad/rev]".format(abs(phi_shift_test_1PN - phi_shift_test_N), abs(p_shift_1PN - p_shift_N)))		
					
	return 
	

#natural initial coordinates
  
t = Time(datetime.now())   
masses = {
'sun'     : Ms, #1st planet has to be the central attractor
'mercury' : Mmerc, #2nd planet has to be the one which we want to test the GR dynamics effects on 
'earth'   : Mearth,
'mars'    : 0.1075*Mearth,
'venus'   : 0.815*Mearth,
'jupiter' : 317.8*Mearth,
#'saturn'  : 95.2*Mearth,
#'uranus'  : 14.6*Mearth,
#'neptune' : 17.2*Mearth,
#'pluto'   : 0.00218*Mearth,
}

planet_names = [
'sun',
'mercury',
'earth',
'mars',
'venus',
'jupiter',
#'saturn',
#'uranus',
#'neptune',
]

planets = []
for planet in planet_names:
	planets.append(get_body_barycentric_posvel(planet,t))

nbodies = len(planets)
#print(planets)
m = np.array([masses[planet] for planet in planet_names]).astype(np.longdouble)

print('m=', m)
Mtot = np.sum(m)

x = np.array([planet[0].x.to(u.meter).value for planet in planets]).astype(np.longdouble)
y = np.array([planet[0].y.to(u.meter).value for planet in planets]).astype(np.longdouble)
z = np.array([planet[0].z.to(u.meter).value for planet in planets]).astype(np.longdouble)

vx = np.array([planet[1].x.to(u.meter/u.second).value for planet in planets]).astype(np.longdouble)
vy = np.array([planet[1].y.to(u.meter/u.second).value for planet in planets]).astype(np.longdouble)
vz = np.array([planet[1].z.to(u.meter/u.second).value for planet in planets]).astype(np.longdouble)

vcm = np.array([np.sum(vx*m/Mtot), np.sum(vy*m/Mtot), np.sum(vz*m/Mtot)])
#print(vcm, np.linalg.norm(vcm))

sx = np.zeros(len(m)).astype(np.longdouble)
sy = np.zeros(len(m)).astype(np.longdouble)
sz = np.zeros(len(m)).astype(np.longdouble)


'''
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


m[0], m[1] = 1.e0*Ms, 1.e0*Mmerc

x[0], x[1] = 0.*AU, 69.818e9
y[0], y[1] = 0.*AU, 0.*AU
z[0], z[1] = 0.0*AU, 0.0*AU

vx[0], vx[1] = 0., 0.
vy[0], vy[1] = 0., 38.86e3
vz[0], vz[1] = 0., 0.

sx[0], sx[1] = 0., 0.
sy[0], sy[1] = 0., 0.
sz[0], sz[1] = 0., 0.

planet_names = ['sun', 'mercury']
'''

'''
m[0], m[1] = 2.e-1*Ms, 0.8e-2*Ms

x[0], x[1] = -1.0*AU, 1.*AU
y[0], y[1] = 0.*AU, 0.*AU
z[0], z[1] = 0.*AU, 0.*AU

vx[0], vx[1] = -4.2e0, +9.2e1
vy[0], vy[1] = +3.86e0, +7.56e1
vz[0], vz[1] = 0., 0.

sx[0], sx[1] = 0., 0.
sy[0], sy[1] = 0., 0.
sz[0], sz[1] = 0., 0.
'''

energy_test(N, dt, m, x, y, z, m*vx, m*vy, m*vz, sx, sy, sz, nout, planet_names)

