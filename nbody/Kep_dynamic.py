import numpy as np
import math
#from __future__ import print_function, division
from PyAstronomy import pyasl
from nbody.CM_coord_system import CM_system
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

#from scipy.special import lambertw

'''
This will become a cython code one day...


cimport cython
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
'''

G = 6.67e-11 #(6.67e-11*u.m**3/(u.kg*u.s**2)).to(u.AU**3/(u.d**2*u.solMass)).value #* (86400 * 86400) /( 2e30 * 1.5e11 * 1.5e11)
C = 299792458. #(3.0e8*(u.m/u.s)).to(u.AU/u.d).value
Ms = 1.988e30 #(2e30*u.kg).to(u.solMass).value
#GM = 1.32712440018e20
C2 = C*C

'''
#work-in-progress --> lambert integrator to solve the trascendental equation for E (eccentric anomaly)

n_branch = 1

def f(t, n):
	n = n*math.sin(t)
	
	return t*math.exp(math.log(1 - n/t))

def g(k, n):
	r = k/n
	
	return -k/(lambertw(-r*math.exp(-r), n_branch) + r)
'''	

'''
#work-in-progress --> ellipse projection from 3D to 2D
	
def gen_3d_frame(p1, p2, q1, q2, Neff, m):

	q_rel, p_rel, q_cm, p_cm = CM_system(p1, p2, q1, q2, Neff, m[0], m[1])		
	
	for i in range(Neff):
	
		psi_orb = np.float(math.atan2(np.sqrt(q_rel[i,1]*q_rel[i,1] + q_rel[i,2]*q_rel[i,2]), q_rel[i,0]))
		theta_orb = np.float(math.atan2(np.sqrt(q_rel[i,0]*q_rel[i,0] + q_rel[i,1]*q_rel[i,1]), q_rel[i,1])) # orbital angle
		phi_orb = np.float(math.atan2(q_rel[i,1], q_rel[i,0]))

		x1 = math.cos(phi_orb)*math.cos(psi_orb) - math.sin(phi_orb)*math.cos(theta_orb)*math.sin(psi_orb)
		x2 = math.sin(phi_orb)*math.cos(psi_orb) + math.cos(phi_orb)*math.cos(theta_orb)*math.sin(psi_orb)
		x3 = math.sin(theta_orb)*math.sin(psi_orb)

		y1 = - math.cos(phi_orb)*math.sin(psi_orb) - math.sin(phi_orb)*math.cos(theta_orb)*math.cos(psi_orb)
		y2 = - math.sin(phi_orb)*math.sin(psi_orb) + math.cos(phi_orb)*math.cos(theta_orb)*math.cos(psi_orb)
		y3 = math.sin(theta_orb)*math.cos(psi_orb)

		z1 = math.sin(theta_orb)*math.cos(phi_orb)
		z2 = - math.sin(theta_orb)*math.cos(phi_orb)
		z3 = math.cos(theta_orb)

		q1[i,0] = (x1 + x2 + x3)*q1[i,0]
		q1[i,1] = (y1 + y2 + y3)*q1[i,1]
		q1[i,2] = (z1 + z2 + z3)*q1[i,2]
		
		q2[i,0] = (x1 + x2 + x3)*q2[i,0]
		q2[i,1] = (y1 + y2 + y3)*q2[i,1]
		q2[i,2] = (z1 + z2 + z3)*q2[i,2]

		p1[i,0] = (x1 + x2 + x3)*p1[i,0]
		p1[i,1] = (y1 + y2 + y3)*p1[i,1]
		p1[i,2] = (z1 + z2 + z3)*p1[i,2]

		p2[i,0] = (x1 + x2 + x3)*p2[i,0]
		p2[i,1] = (y1 + y2 + y3)*p2[i,1]
		p2[i,2] = (z1 + z2 + z3)*p2[i,2]

		#h, t, v = _H_2body(np.ndarray(m[0:1]), np.ndarray(q1[i,0], q2[i,0]), np.ndarray(q1[i,1], q2[i,1]), np.ndarray(q1[i,2], q2[i,2]), np.ndarray(p1[i,0], p2[i,0]), np.ndarray(p1[i,1], p2[i,1]), np.ndarray(p1[i,2], p2[i,2]), np.ndarray(s1[i,0], s2[i,0]), np.ndarray(s1[i,1], s2[i,1]), np.ndarray((s1[i,2], s2[i,2])), order)	
					
	return(q1, q2, p1, p2)#, H) 
'''
		
def kepler(q1, q2, p1, p2, D, N, Neff, H, m, dt, order, q_peri):
	
	q_rel, p_rel, q_cm, p_cm = CM_system(p1, p2, q1, q2, Neff, m[0], m[1])	
	
	#L1 = np.cross(q1, p1) 
	#L1 = np.linalg.norm(L1, axis=-1)
	#L2 = np.cross(q2, p2) 
	#L2 = np.linalg.norm(L2, axis=-1)
	
	#L_cm = np.cross(q_cm, p_cm)
	#L_cm = np.linalg.norm(L_cm, axis=-1)
	
	#L = np.linalg.norm(L1 + L2 - L_cm, axis=-1)
	#L = L1 + L2 - L_cm		
	
	L_rel =	np.cross(q_rel, p_rel)
	L = np.linalg.norm(L_rel, axis = -1)
	
	#print(L, L_cm, L_rel)	
	
	q_analit_rel = np.array([[0 for i in range(0, 3)] for Neff in range(0, Neff)], dtype='float64')

	#normal = np.array([[0 for i in range(0, 3)] for Neff in range(0, Neff)], dtype='float64')	
	p_cm_2 = np.array([[0 for i in range(0, 3)] for Neff in range(0, Neff)], dtype='float64')	
	r_dif = np.zeros(Neff, dtype='float64')
	r_rel = np.zeros(Neff, dtype='float64')
	
	for i in range(0, Neff):
		r_rel[i] = math.sqrt(q_rel[i,0]*q_rel[i,0] + q_rel[i,1]*q_rel[i,1] + q_rel[i,2]*q_rel[i,2])	
			   
	#Dinamica Kepleriana#-------------------------------------------------#
	
	M = m[0] + m[1]
	mu = (m[0]*m[1])/M
	
	#ricavo H_rel -----------------------
	p_cm_2 = (p_cm[:,0]*p_cm[:,0] + p_cm[:, 1]*p_cm[:, 1] + p_cm[:, 2]*p_cm[:, 2])
	#print(p_cm_2)
	H -= p_cm_2/(2*M)
	
	'''
	if (order >= 1):
	  
		for k in range(3):
			normal[:,k] = (q_cm[:, k]- q_rel[:, k])/r_rel
		
		H -= (-(1./8.)*(p_cm_2*p_cm_2)/(M*M*M))/C2
			
		for i in range(Neff):
				
			H[i] -= ((1./8.)*(G*mu*M/r_rel[i])*(-12.*p_cm_2[i]/(M*M) + 14.0*np.dot(p_cm[i,:], p_rel[i,:])/(M*mu) + 2.0*np.dot(normal[i,:], p_cm[i,:])*np.dot(normal[i,:], p_rel[i,:])/(M*mu)))/C2
		
		H -= (0.25*(G*mu*M/r_rel)*(G*M/r_rel))/C2
	'''	
				
	#------------------------------------		

	#print(p_cm, p_rel)
	
	H2 = H*H
	L2 = L*L	
		
	k = G*M*mu

	R = L2/(k*mu) # semi-latus rectum = a*(1 - e*e)
	alpha = H2/R 
	
	e = np.zeros(Neff, dtype='float64')	
	E = np.zeros(Neff, dtype='float64')	
	m_a = np.zeros(Neff, dtype='float64')	
	a_p = np.zeros(Neff, dtype='float64')	
	P_quad = np.zeros(Neff, dtype='float64')		
	t = np.zeros(Neff, dtype='float64')	
	r_kepler = np.zeros(Neff, dtype='float64')
	phi_orb = np.zeros(Neff, dtype='float64')
	#True_anom = np.zeros(Neff, dtype='float64')
	phi_shift = np.zeros(Neff, dtype='float64')
	
	for i in range(0, Neff):

		if (order == 0):
		
			e[i] = np.float(math.sqrt(1 + (2*H[i]*L2[i])/(k*k*mu)))		
			 
			'''
			routine per il calcolo dell'orbita kepleriana anche in caso di orbita iperbolica (https://kyleniemeyer.github.io/space-systems-notes/orbital-mechanics/two-body-problems.html)
			'''		
			
			phi_orb[i] = np.float(math.atan2(q_rel[i,1], q_rel[i,0]))
		
			if (1 > e[i]):
				a = np.float(R[i]/(1 - e[i]*e[i])) # semi-major axis
				b = np.float(R[i]/(math.sqrt(1 - e[i]*e[i]))) # semi-minor axis

			if (1 <= e[i]):
				a = np.float(R[i]/(e[i]*e[i] - 1)) # semi-major axis
				b = np.float(R[i]/(math.sqrt(e[i]*e[i] - 1))) # semi-minor axis

			c = np.sqrt(a*a - b*b)		
		
			E[i] = 2*math.atan2(math.sqrt(1.- e[i])*math.tan(phi_orb[i]/2.), math.sqrt(1.+ e[i])) # eccentric anomaly
			
			#True_anom[i] = 2*math.atan2(math.sqrt(1.+ e)*math.tan(E/2.), math.sqrt(1.- e))
			'''
			m = math.atan2(math.sqrt(1 - e*e)*math.sin(phi_orb)/(1+e*math.cos(phi_orb)), (e + math.cos(phi_orb))/(1 + e*math.cos(phi_orb))) - e*math.sqrt(1 - e*e)*math.sin(phi_orb)/(1 + e*math.cos(phi_orb))
			
			E = m + (e - (1./8.)*e*e*e )*math.sin(m) + (1./2.)*e*e*math.sin(2.*m) + (3./8.)*math.sin(3.*m)
			'''
			
			
			#another routine to calculate the right E (Newton-Raphson iterarion). "ON THE COMPUTATION OF THE ECCENTRIC ANOMALY FROM THE MEAN ANOMALY OF A PLANET"
			
			j_iter = 0
			
			for j in range(0, j_iter):
				m_a[i] = E[i] - e[i]*math.sin(E[i])

				theta = math.atan2(e[i]*math.sqrt(2.)*math.sin(m_a[i]), (1. - e[i]*math.cos(m_a[i])))
				
				root = math.sqrt(2.)*math.tan(0.5*theta)
				
				E_temp = m_a[i] + root
				m_temp = E_temp - e[i]*math.sin(E_temp)
				
				E[i] = E_temp + (m_a[i] - m_temp)/(1. - e[i]*math.cos(E_temp))
				
			m_a[i] = E[i] - e[i]*math.sin(E[i])

			#Development Of Closed-Form Approximation Of The Eccentric Anomaly For Circular And Elliptical Keplerian Orbit 

			#m = math.atan2(math.sqrt(1 - e*e)*math.sin(phi_orb)/(1+e*math.cos(phi_orb)), (e + math.cos(phi_orb))/(1 + e*math.cos(phi_orb))) - e*math.sqrt(1 - e*e)*math.sin(phi_orb)/(1 + e*math.cos(phi_orb))
			
			'''
			if (0.5 <= e[i] <= 1):
				A = -0.584013113 
				B = 1.173439404
				F = 0.809460441
				D = 0.077357763
				
			if (0.01 <= e[i] <= 5):
				A = -0.248393819  
				B = 1.019165175 
				F = 0.961260155
				D = 0.004043021		
				
			theta = (B*math.sin(m_a) + D*math.cos(m_a))/(1./e[i] - A*math.sin(m_a) - F*math.cos(m_a))
			
			print(theta, m_a)
			
			E = m_a + e[i]*(math.sin(m_a + theta))
			'''
					
			if (e[i]<1):
				x_kepler = a*math.cos(E[i]) - c #a*math.cos(E[i])  # 
				y_kepler = b*math.sin(E[i]) 
				
				t[i] = a*math.sqrt(a/alpha[i])*(E[i] - e[i]*math.sin(E[i]))
				r_kepler[i] = a*(1 - e[i]*math.cos(E[i]))

			if (e[i]>1):
				x_kepler = c - a*math.cosh(E[i]) #( a*math.cosh(E[i]) 
				y_kepler = b*(math.sinh(E[i]))

				t[i] = a*math.sqrt(a/alpha[i])*(e[i]*math.sinh(E[i]) - E)
				r_kepler[i] = a*(e[i]*math.cosh(E[i]) - 1)
				
			if (e[i]==1):
				x_kepler = a*math.cosh(E[i]) + c #a*math.cosh(E[i]) 
				y_kepler = b*(math.sinh(E[i]))

				t[i] = a*math.sqrt(a/alpha[i])*(e[i]*math.sinh(E[i]) + E)
				r_kepler[i] = a*(e[i]*math.cosh(E[i]) + 1) 
			
			'''
			# angular displacement of the ellipse (https://mathworld.wolfram.com/Ellipse.html)
			
			if ((b==0) & (a < c)):
				theta_ellipse = 0  
				
			if ((b==0) & (a > c)):
				theta_ellipse = 0.5*math.pi

			if ((b!=0) & (a < c)):
				theta_ellipse = 0.5*(1/(math.atan((a - c)/(2*b))))

			if ((b!=0) & (a > c)):
				theta_ellipse = 0.5*(math.pi + 1/(math.atan((a - c)/(2*b))))

			#Equations to center a generical ellipse (https://math.stackexchange.com/questions/2645689/what-is-the-parametric-equation-of-a-rotated-ellipse-given-the-angle-of-rotatio/2647450#2647450)
			
			if (i == 0):
				disp_x = q_rel[i,0] - a*math.cos(phi_orb)*math.cos(theta_ellipse) + b*math.sin(phi_orb)*math.sin(theta_ellipse) 
				disp_y = q_rel[i,1] - a*math.cos(phi_orb)*math.sin(theta_ellipse) - b*math.sin(phi_orb)*math.cos(theta_ellipse)

			#cartesian coordinates 

			x_kepler += disp_x
			y_kepler += disp_y
			'''

			q_analit_rel[i,0] = x_kepler #+ tmp_x
			q_analit_rel[i,1] = y_kepler #+ tmp_y
			q_analit_rel[i,2] = q_rel[i,2] #+ q_cm[i,2]

		if (order >= 1):
			a = (np.max(r_rel) + np.min(r_rel))/2.
			b = np.sqrt(np.max(r_rel)*np.min(r_rel))
			e[i] = np.sqrt(1 - (b*b)/(a*a))
			
			q_analit_rel[i,0] = q_rel[i,0] #+ tmp_x
			q_analit_rel[i,1] = q_rel[i,1]#+ tmp_y
			q_analit_rel[i,2] = q_rel[i,2] #+ q_cm[i,2]

		r_dif[i] = abs(math.sqrt(q_analit_rel[i,0]*q_analit_rel[i,0] + q_analit_rel[i,1]*q_analit_rel[i,1] + q_analit_rel[i,2]*q_analit_rel[i,2]) - r_rel[i]) 

		# Dediu, Adrian-Horia; Magdalena, Luis; Martín-Vide, Carlos (2015). Theory and Practice of Natural Computing: Fourth International Conference, TPNC 2015, Mieres, Spain, December 15–16, 2015. Proceedings (illustrated ed.); Springer. [p. 141]

		a_p[i] = 6*math.pi*G*M/(C*C*a*(1 - e[i]*e[i])) #apsidial precession [#rad per revolution]
		
		#t[i] = math.sqrt((4*math.pi*math.pi*a*a*a)/(G*M))
		
		#dal maggiore ---------------------
		f_e = (1./((1. - e[i]*e[i])**(7./2.)))*(1. + (73./24.)*(e[i]*e[i]) + (37./96.)*(e[i]*e[i]*e[i]*e[i]))
		
		P_quad[i] = -(((32./5.)*(G*G*G*G)*(mu*mu)*(M*M*M))/((a*a*a*a*a)*(C*C*C*C*C)))*f_e   	
          	#----------------------------------
        
        #numerical shift
	for i in range(0, Neff):
        
        	shift = math.sqrt(G*M*R[i])/(r_rel[i]*r_rel[i]) #in keplerian motion this is constant (2nd law)
        	phi_shift[i] = shift

	'''
	peri_indexes = argrelextrema(r_rel, np.less)
	peri_indexes = np.transpose(peri_indexes)
		
	n_peri = len(peri_indexes)
	
	q_peri = np.array([[0 for i in range(0, 3)] for n_peri in range(0, n_peri)], dtype='float64')

	v_peri = np.array([[0 for i in range(0, 3)] for n_peri in range(0, n_peri)], dtype='float64')
        '''

	n_peri = len(q_peri)
	
	#Dq_shift_tmp2 = np.zeros(3, dtype='float64')
	#Dq_shift_tmp1 = np.array([[0 for i in range(0, 3)] for n_peri in range(0, n_peri-1)], dtype='float64')	
	
	#diff = np.zeros(n_peri-1)
	
	phi_shift_test = 0 
	Dq_shift = 0


	test = np.ones((n_peri), dtype='float64')
	acos_test = np.zeros((n_peri), dtype='float64')


	if (n_peri != 0):
		for i in range(0, n_peri):
	
			#q_peri[i,:] = q2[peri_indexes[i], :] #- q_cm[peri_indexes[i], :]
			#v_peri[i,:] = p2[peri_indexes[i], :]/m[1] #- q_cm[peri_indexes[i], :]
			
			if (i != 0):		
				Dx = D[peri_indexes[i], 1, 0:3] 
				Dy = D[peri_indexes[i-1], 1, 0:3]
			
				#Dx = D[peri_indexes[i], 1, 3:6] 
				#Dy = D[peri_indexes[i-1], 1, 3:6]
			
				print(math.acos(np.dot(q_peri[i, :], q_peri[i-1, :])/(np.linalg.norm(q_peri[i, :])*np.linalg.norm(q_peri[i-1, :]))), np.dot(q_peri[i, :], q_peri[i-1, :])/(np.linalg.norm(q_peri[i, :])*np.linalg.norm(q_peri[i-1, :])))	
	
				test[i] = (np.dot(q_peri[i, :], q_peri[i-1, :])/(np.linalg.norm(q_peri[i, :])*np.linalg.norm(q_peri[i-1, :])))	                        

				phi_shift_test += np.float(math.acos(np.dot(q_peri[i, :], q_peri[i-1, :])/(np.linalg.norm(q_peri[i, :])*np.linalg.norm(q_peri[i-1, :]))))

				acos_test[i] = phi_shift_test

				#phi_shift_test += math.acos(np.dot(v_peri[i, :], v_peri[i-1, :])/(np.linalg.norm(v_peri[i, :])*np.linalg.norm(v_peri[i-1, :])))

				Dq_shift_tmp1[i-1, :] = (Dx[:]*np.abs(- (math.tan(np.dot(q_peri[i, :], q_peri[i-1, :]))*np.abs(q_peri[i, :]))/(q_peri[i, :]*q_peri[i, :]*np.abs(q_peri[i-1, :])))) + (Dy[:]*np.abs(- (math.tan(np.dot(q_peri[i, :], q_peri[i-1, :]))*np.abs(q_peri[i-1, :]))/(q_peri[i-1, :]*q_peri[i-1, :]*np.abs(q_peri[i, :]))))

				#Dq_shift_tmp1[i-1, :] = (Dx[:]*np.abs(- (math.tan(np.dot(v_peri[i, :], v_peri[i-1, :]))*np.abs(v_peri[i, :]))/(v_peri[i, :]*v_peri[i, :]*np.abs(v_peri[i-1, :])))) + (Dy[:]*np.abs(- (math.tan(np.dot(v_peri[i, :], v_peri[i-1, :]))*np.abs(v_peri[i-1, :]))/(v_peri[i-1, :]*v_peri[i-1, :]*np.abs(v_peri[i, :]))))
			

		Dq_shift_tmp2[:] = np.sum(Dq_shift_tmp1[:])
		Dq_shift = np.linalg.norm(Dq_shift_tmp2[:])

		 #print(Dq_shift, Dq_shift_sigma)

	'''
	N_arr = np.linspace(0, N - N/Neff, Neff)
	        
	f = plt.figure(figsize=(6,4))
	
	ax = f.add_subplot(121)
	ax.plot(N_arr, phi_shift)
	ax.set_xlabel('iteration')
	ax.set_ylabel('angular velocity')
	ax.grid()

	ax1 = f.add_subplot(122)
	ax1.scatter(m_a, E)
	ax1.set_xlabel('Mean anomaly')
	ax1.set_ylabel('Eccentric anomaly')
	ax1.grid()
	
	plt.show()         
	'''

	N_arr = np.linspace(0, N, n_peri)
	        
	f = plt.figure(figsize=(6,4))
	ax = f.add_subplot(111)
	ax.plot(N_arr, test, label ='argument')
	ax.plot(N_arr, acos_test, label ='acos(argument)')
	ax.set_xlabel('iteration')
	ax.set_ylabel('shift')
	plt.grid()
	plt.legend()
	plt.show()   

	return (r_dif, q_analit_rel, r_kepler, L, a_p, P_quad, phi_shift, phi_shift_test, Dq_shift)

	#return (r_dif, q_analit_rel, r_kepler, L, a_p, P_quad, q_peri, phi_shift, peri_indexes, phi_shift_test, Dq_shift)


	
def kepler_sol_sys(p, q, D, Neff, H, m, dt, order):

	L_arr = np.array([[0 for i in range(0, Neff)] for n_peri in range(0, len(m))], dtype='float64')
	normal = np.array([[0 for i in range(0, 3)] for Neff in range(0, Neff)], dtype='float64')		
	d_rel = np.zeros(Neff, dtype='float64')	
	d_cm = np.zeros(Neff, dtype='float64')	
	
	q_rel, p_rel, q_cm, p_cm = CM_system(p[0], p[1], q[0], q[1], Neff, m[0], m[1])	

	for i in range(0,Neff):
		d_rel[i] = math.sqrt(q_rel[i,0]*q_rel[i,0] + q_rel[i,1]*q_rel[i,1] + q_rel[i,2]*q_rel[i,2])
		d_cm[i] = math.sqrt(q_cm[i,0]*q_cm[i,0] + q_cm[i,1]*q_cm[i,1] + q_cm[i,2]*q_cm[i,2])

	for i in range(len(m)):
		L_temp = np.cross(q[i, :], p[i, :])
		L_arr[i] = np.linalg.norm(L_temp, axis=-1) 	

	L_tot = np.sum(L_arr, axis=0)
				   
	#Dinamica Kepleriana#------------------#

	M = m[0] + m[1]
	mu = (m[0]*m[1])/M	

	p_cm_2 = (p_cm[:,0]*p_cm[:,0] + p_cm[:, 1]*p_cm[:, 1] + p_cm[:, 2]*p_cm[:, 2])
	
	H -= p_cm_2/(2*M)

	'''
	if (order >= 1):
	  
		for k in range(3):
			normal[:,k] = q_cm[:, k]/d_cm
			#normal[:,k] = (q_cm[:, k]- q_rel[:, k])/d_rel
		
		H -= (-(1./8.)*(p_cm_2*p_cm_2)/(M*M*M))/C2
			
		for i in range(Neff):
				
			H[i] -= ((1./8.)*(G*mu*M/d_rel[i])*(-12.*p_cm_2[i]/(M*M) + 14.0*np.dot(p_cm[i,:], p_rel[i,:])/(M*mu) + 2.0*np.dot(normal[i,:], p_cm[i,:])*np.dot(normal[i,:], p_rel[i,:])/(M*mu)))/C2
		
		H -= (0.25*(G*mu*M/d_rel)*(G*M/d_rel))/C2  
	'''
					
	L_rel =	np.cross(q_rel, p_rel)
	L = np.linalg.norm(L_rel, axis=-1)

	#H = np.array(H)	
	H2 = H*H
	L2 = L*L	
		
	k = G*M*mu

	R = L2/(k*mu)
	
	P_quad = np.zeros(Neff, dtype='float64')	
	a_p1 = np.zeros(Neff, dtype='float64')	
	a_p2_arr = np.zeros(len(m), dtype='float64')
	a_p2 = np.zeros(Neff, dtype='float64')
	a_p3 = np.zeros(Neff, dtype='float64')	
	a_p4_arr = np.zeros(len(m), dtype='float64')
	a_p4 = np.zeros(Neff, dtype='float64')
	#T = np.zeros(Neff, dtype='float64')	
	t = np.zeros(Neff, dtype='float64')	
	r_kepler = np.zeros(Neff, dtype='float64')	
	phi_shift = np.zeros(Neff, dtype='float64')
	
	#a = (np.max(d_rel) + np.min(d_rel))/2.
	#b = np.sqrt(np.max(d_rel)*np.min(d_rel))
	#e = np.sqrt(1 - (b*b)/(a*a))

	#theoretical shift
	for i in range(0, Neff):		

		e = np.float(math.sqrt(1 + (2*H[i]*L2[i])/(k*k*mu)))	

		if (1 > e):		
			a = np.float(R[i]/(1 - e*e)) # semi-major axis
			b = np.float(R[i]/(math.sqrt(1 - e*e))) # semi-minor axis
		   
		if (1 <= e):
			a = (R[i]/(e*e - 1.)) # semi-major axis
			b = (R[i]/(math.sqrt(e*e - 1.))) # semi-minor axis

		#apsidial precession given by GR [#rad per revolution]
		a_p1[i] = 6.*math.pi*G*M/(C*C*a*(1. - e*e))
		
		#apsidial precession given by coupling of the planet with the other bodies  (newtonian) [#rad per revolution]

		for j in range(2, len(m)):	
		
			'''
			distance from sun-mercury system or distance from mercury?
			'''		
					
			#r_m = math.sqrt((q[1,i,0]-q[j,i,0])*(q[1,i,0]-q[j,i,0]) + (q[1,i,1]-q[j,i,1])*(q[1,i,1]-q[j,i,1]) + (q[1,i,2]-q[j,i,2])*(q[1,i,2]-q[j,i,2])) #distance from Mercury
			
			r_m = math.sqrt((q_rel[i,0]-q[j,i,0])*(q_rel[i,0]-q[j,i,0]) + (q_rel[i,1]-q[j,i,1])*(q_rel[i,1]-q[j,i,1]) + (q_rel[i,2]-q[j,i,2])*(q_rel[i,2]-q[j,i,2])) #distance from Sun-Mercury
			
			a_p2_arr[j] = (3.*math.pi/2.)*(m[j]/M)*(a/r_m)*(a/r_m)*(a/r_m)*(math.sqrt(1.- e*e)) #apsidial precession given by coupling of the planet with the other bodies  (newtonian) [#rad per revolution]

			a_p4_arr[j] = 4.*math.pi*(G*m[j]/(C*C*a))*(a/r_m)*(a/r_m)*math.sqrt(a/r_m) #apsidial precession given by gravitomagnetic effect [#rad per revolution]		
			
		a_p2[i] = np.sum(a_p2_arr) 
		
		a_p3[i] = (1. + 0.5*(G*M*(28. + 47.*e*e))/(C*C*a*(1. - e*e)*(1. - e*e)))
		
		a_p4[i] = np.sum(a_p4_arr)
		
		#t[i] = math.sqrt((4*math.pi*math.pi*a*a*a)/(G*M)) 
		#dal maggiore
		
		f_e = (1./((1. - e*e)**(7./2.)))*(1. + (73./24.)*(e*e) + (37./96.)*(e*e*e*e))
		
		P_quad[i] = - (((32./5.)*(G*G*G*G)*(mu*mu)*(M*M*M))/((a*a*a*a*a)*(C*C*C*C*C)))*f_e   

        #numerical shift
	
	for i in range(0, Neff):
        
        		shift = math.sqrt(G*M*R[i])/(d_rel[i]*d_rel[i]) #in keplerian motion this is constant (2nd law)
        		phi_shift[i] = shift
	
	peri_indexes = argrelextrema(d_rel, np.less)
	peri_indexes = np.transpose(peri_indexes)
		
	n_peri = len(peri_indexes)
	q_peri = np.array([[0 for i in range(0, 3)] for n_peri in range(0, n_peri)], dtype='float64')
	v_peri = np.array([[0 for i in range(0, 3)] for n_peri in range(0, n_peri)], dtype='float64')
	Dq_shift_tmp2 = np.zeros(3, dtype='float64')
	Dq_shift_tmp1 = np.array([[0 for i in range(0, 3)] for n_peri in range(0, n_peri-1)], dtype='float64')	
	
	#diff = np.zeros(n_peri-1)
	
	phi_shift_test = 0 
	Dq_shift = 0	

	if (n_peri != 0):
		for i in range(0, n_peri):

			#q_peri[i,:] = q_rel[peri_indexes[i], :] #- q_cm[peri_indexes[i], :]

			q_peri[i,:] = q[1, peri_indexes[i], :]

			#v_peri[i,:] = p[1, peri_indexes[i], :]/m[1] #- q_cm[peri_indexes[i], :]
			
			if (i != 0):					
				Dx = D[peri_indexes[i], 1, 0:3] 
				Dy = D[peri_indexes[i-1], 1, 0:3]			
			
				phi_shift_test += np.float(math.acos(np.dot(q_peri[i, :], q_peri[i-1, :])/(np.linalg.norm(q_peri[i, :])*np.linalg.norm(q_peri[i-1, :]))))

				print(math.acos(np.dot(q_peri[i, :], q_peri[i-1, :])/(np.linalg.norm(q_peri[i, :])*np.linalg.norm(q_peri[i-1, :]))), np.dot(q_peri[i, :], q_peri[i-1, :])/(np.linalg.norm(q_peri[i, :])*np.linalg.norm(q_peri[i-1, :])))

				#phi_shift_test += math.acos(np.dot(v_peri[i, :], v_peri[i-1, :])/(np.linalg.norm(v_peri[i, :])*np.linalg.norm(v_peri[i-1, :])))

				Dq_shift_tmp1[i-1, :] = (Dx[:]*np.abs(- (math.tan(np.dot(q_peri[i, :], q_peri[i-1, :]))*np.abs(q_peri[i, :]))/(q_peri[i, :]*q_peri[i, :]*np.abs(q_peri[i-1, :])))) + (Dy[:]*np.abs(- (math.tan(np.dot(q_peri[i, :], q_peri[i-1, :]))*np.abs(q_peri[i-1, :]))/(q_peri[i-1, :]*q_peri[i-1, :]*np.abs(q_peri[i, :]))))

				#Dq_shift_tmp1[i-1, :] = (Dx[:]*np.abs(- (math.tan(np.dot(v_peri[i, :], v_peri[i-1, :]))*np.abs(v_peri[i, :]))/(v_peri[i, :]*v_peri[i, :]*np.abs(v_peri[i-1, :])))) + (Dy[:]*np.abs(- (math.tan(np.dot(v_peri[i, :], v_peri[i-1, :]))*np.abs(v_peri[i-1, :]))/(v_peri[i-1, :]*v_peri[i-1, :]*np.abs(v_peri[i, :]))))
			
		phi_shift_test = phi_shift_test/n_peri

		'''
		for i in range(1, n_peri):		
			diff[i-1] = (np.float(math.acos(np.dot(q_peri[i, :], q_peri[i-1, :])/(np.linalg.norm(q_peri[i, :])*np.linalg.norm(q_peri[i-1, :])))) - phi_shift_test)*(np.float(math.acos(np.dot(q_peri[i, :], q_peri[i-1, :])/(np.linalg.norm(q_peri[i, :])*np.linalg.norm(q_peri[i-1, :])))) - phi_shift_test)
	
		Dq_shift_sigma = (1/(np.sqrt(n_peri - 1)))*(np.sqrt(np.sum(diff)/n_peri))
		'''

		Dq_shift_tmp2[:] = np.sum(Dq_shift_tmp1[:])
		Dq_shift = np.linalg.norm(Dq_shift_tmp2[:])

		#print(Dq_shift, Dq_shift_sigma)
	        
	return (L_tot, P_quad, a_p1, a_p2, a_p3, a_p4, q_peri, phi_shift, phi_shift_test, peri_indexes, Dq_shift)
	

def KepElemToCart(a, T, e, Omega, i, w, N, Neff):
	
	ke = pyasl.KeplerEllipse(a, T, e, Omega, i, w)
	
	#t = np.linspace(0, N, Neff)
	# Calculate the orbit position at the given points
	# in a Cartesian coordinate system.
	pos = ke.xyzPos(0)
	
	#print("Shape of output array: ", pos.shape)

	# Calculate orbit radius as a function of the
	#radius = ke.radius(1)

	# Calculate velocity on orbit
	vel = ke.xyzVel(0)
	
	return (pos[:][0], pos[:][1], pos[:][2], vel[:][0], vel[:][1], vel[:][2])#, radius)
	
