import numpy as np
import math
from nbody.CM_coord_system import CM_system
from scipy.special import lambertw

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
	
'''
WORK IN PROGRESS
'''

'''
def kep_integrator(p1, p2, q1, q2, m, dt, ICN_it, H, Neff):

	return    
         
'''

n_branch = 1

def f(t, n):
	n = n*math.sin(t)
	
	return t*math.exp(math.log(1 - n/t))

def g(k, n):
	r = k/n
	
	return -k/(lambertw(-r*math.exp(-r), n_branch) + r)
	
	
	
def gen_3d_frame(p1, p2, q1, q2, Neff, m):

	q_rel, p_rel, q_cm, p_cm = CM_system(p1, p2, q1, q2, Neff, m[0], m[1])	
	
	#H_2body = []
	#V_2body = []
	#T_2body = [] 	
	
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
		
def kepler(q1, q2, p1, p2, Neff, H, m, dt):

	L1 = q1[:,1]*p1[:,2] - q1[:,2]*p1[:,1] - q1[:,0]*p1[:,2] + q1[:,2]*p1[:,0] + q1[:,0]*p1[:,1] - q1[:,1]*p1[:,0]
		
	L2 = q2[:,1]*p2[:,2] - q2[:,2]*p2[:,1] - q2[:,0]*p2[:,2] + q2[:,2]*p2[:,0] + q2[:,0]*p2[:,1] - q2[:,1]*p2[:,0]
		
	L = L1 + L2

	q_rel, p_rel, q_cm, p_cm = CM_system(p1, p2, q1, q2, Neff, m[0], m[1])
				
	#L = q_rel[:,1]*p_rel[:,2] - q_rel[:,2]*p_rel[:,1] - q_rel[:,0]*p_rel[:,2] + q_rel[:,2]*p_rel[:,0] + q_rel[:,0]*p_rel[:,1] - q_rel[:,1]*p_rel[:,0]
	
	q_analit_rel = np.array([[0 for i in range(0, 3)] for Neff in range(0, Neff)], dtype='float64')

	#q2_analit = np.zeros(Neff)          
	#q1_analit = np.zeros(Neff)
	d_dif = np.zeros(Neff, dtype='float64')
	#q1_dif = np.zeros(Neff)         
	#q2_dif = np.zeros(Neff) 
			   
	#Dinamica Kepleriana#------------------#
	H2 = H*H
	
	L2 = L*L
	M = m[0] + m[1]
	mu = (m[0]*m[1])/M

	#P = p1 + p2
	#P_2 = np.sqrt(P[:,0]*P[:,0] + P[:,1]*P[:,1] + P[:,2]*P[:,2])
		
	k = -G*M*mu
	#k = G*(mu*mu*mu)*(M*M)
	R = L2/(-k*mu) # semi-latus rectum = a*(1 - e*e)
	alpha = H2/R 

	#e = np.zeros(Neff)
	#a = np.zeros(Neff, dtype='float64')	
	a_p = np.zeros(Neff, dtype='float64')	
	P_quad = np.zeros(Neff, dtype='float64')	
	#T = np.zeros(Neff, dtype='float64')	
	t = np.zeros(Neff, dtype='float64')	
	r_kepler = np.zeros(Neff, dtype='float64')	
	
	for i in range(0, Neff):

		d_rel = math.sqrt(q_rel[i,0]*q_rel[i,0] + q_rel[i,1]*q_rel[i,1] + q_rel[i,2]*q_rel[i,2])	

		'''
		q2_analit[i,:] = -q_rel[i,:]/2. 
		q1_analit[i,:] = q_rel[i,:]/2. 

		#q1_dif[i,:] = abs(q1_analit[i,:] - q1[i,:])
		#q2_dif[i,:] = abs(q2_analit[i,:] - q2[i,:])

		q_analit_rel[i,:] =  q1_analit[i,:] - q2_analit[i,:]
		
		d_analit_rel = math.sqrt(q_analit_rel[i,0]*q_analit_rel[i,0] + q_analit_rel[i,1]*q_analit_rel[i,1] + q_analit_rel[i,2]*q_analit_rel[i,2])

		r_dif[i] = d_rel - d_analit_rel
        '''
        
		#c = e*a
		#c = np.float(math.sqrt(abs(a*a - b*b))) 		  	
		#print(1 + (2*H[i]*L2[i])/(k*k*mu))
		
		#e = np.float(math.sqrt(1 + (2*H[i]*L2[i])/(mu*k*k*mu*mu))) # eccentricity
		e = np.float(math.sqrt(1 + (2*H[i]*L2[i])/(k*k*mu)))
			
		#print(e, q_cm, p_cm)	
		
		#e = c/a
		#a = np.float(R[i]/(1 - e*e)) # semi-major axis
		#b = np.float(R[i]/(math.sqrt(1 - e*e))) # semi-minor axis  	
		 
		'''
		routine per il calcolo dell'orbita kepleriana anche in caso di orbita iperbolica (https://kyleniemeyer.github.io/space-systems-notes/orbital-mechanics/two-body-problems.html)
		'''		

		#psi_orb = np.float(math.atan2(np.sqrt(q_rel[i,1]*q_rel[i,1] + q_rel[i,2]*q_rel[i,2]), q_rel[i,0]))
		#theta_orb = np.float(math.atan2(np.sqrt(q_rel[i,0]*q_rel[i,0] + q_rel[i,1]*q_rel[i,1]), q_rel[i,2])) # orbital angle
		phi_orb = np.float(math.atan2(q_rel[i,1], q_rel[i,0]))
	
		if (1 > e):
			a = np.float(R[i]/(1 - e*e)) # semi-major axis
			b = np.float(R[i]/(math.sqrt(1 - e*e))) # semi-minor axis

		if (1 <= e):
			a = np.float(R[i]/(e*e - 1)) # semi-major axis
			b = np.float(R[i]/(math.sqrt(e*e - 1))) # semi-minor axis

		c = e*a		
		#phi_orb = math.acos(q_rel[i,0]/d_rel)

		#r_kepler = a*(1 - e*e)/(1 + e*math.cos(phi_orb))
		#r_kepler[i] = np.float(R[i]/(1 + e*math.cos(phi_orb))) # r_phi keplerian equation
		
		#E = math.acos((e + math.cos(psi_orb))/(1 + e*math.cos(psi_orb)))		
		#print(math.tan(phi_orb/2))
		E = 2*math.atan2(math.sqrt(1. - e)*math.tan(phi_orb/2.), math.sqrt(1. + e)) # eccentric anomaly
			
		'''
		m = math.atan2(math.sqrt(1 - e*e)*math.sin(phi_orb)/(1+e*math.cos(phi_orb)), (e + math.cos(phi_orb))/(1 + e*math.cos(phi_orb))) - e*math.sqrt(1 - e*e)*math.sin(phi_orb)/(1 + e*math.cos(phi_orb))
		
		E = m + (e - (1./8.)*e*e*e )*math.sin(m) + (1./2.)*e*e*math.sin(2.*m) + (3./8.)*math.sin(3.*m)
		'''
		
		'''
		# "ON THE COMPUTATION OF THE ECCENTRIC ANOMALY FROM THE MEAN ANOMALY OF A PLANET"

		m = E - e*math.sin(E)
			
		theta = math.atan2(e*math.sqrt(2.)*math.sin(m), (1. - e*math.cos(m)))
		
		#root = - (1. - e*math.cos(m))/(e*math.sin(m)) - math.sqrt((1. - e*math.cos(m))/(e*math.sin(m))*(1. - e*math.cos(m))/(e*math.sin(m)) + 2.)
		root = math.sqrt(2.)*math.tan(0.5*theta)
		
		E_temp = m + root
		m_temp = E_temp - e*math.sin(E_temp)
		
		E = E_temp + (m - m_temp)/(1. - e*math.cos(E_temp))
		'''
		
		'''
		#Development Of Closed-Form Approximation Of The Eccentric Anomaly For Circular And Elliptical Keplerian Orbit 

		m = math.atan2(math.sqrt(1 - e*e)*math.sin(phi_orb)/(1+e*math.cos(phi_orb)), (e + math.cos(phi_orb))/(1 + e*math.cos(phi_orb))) - e*math.sqrt(1 - e*e)*math.sin(phi_orb)/(1 + e*math.cos(phi_orb))
		
		#m = E - e*math.sin(E)

		if (0.5 <= e <= 1):
			A = -0.584013113 
			B = 1.173439404
			F = 0.809460441
			D = 0.077357763
			
		if (0.01 <= e <= 5):
			A = -0.248393819  
			B = 1.019165175 
			F = 0.961260155
			D = 0.004043021		
			
		theta = (B*math.sin(m) + D*math.cos(m))/(1./e - A*math.sin(m) - F*math.cos(m))

		E = m + e*(math.sin(m + theta))
		'''
		
		#par2, par1 = e*math.sin(E), f(E, e)
		#E = np.real(g(par1, par2))

		#r_kepler = a*(1 - e*math.cos(E)) 
 
		#E = np.float(math.acos(r_kepler/c - 1/e)) 	
		#E = math.acos(q_rel[i,0]/a) 

		if (e<1):
			x_kepler = (a*math.cos(E) - c)
			y_kepler = b*math.sin(E)
			
			t[i] = a*math.sqrt(a/alpha[i])*(E - e*math.sin(E))
			r_kepler[i] = np.float(a*(1 - e*math.cos(E))) 

		if (e>1):
			x_kepler = (c - a*math.cosh(E))
			y_kepler = b*(math.sinh(E))

			t[i] = a*math.sqrt(a/alpha[i])*(e*math.sinh(E) - E)
			r_kepler[i] = np.float(a*(e*math.cosh(E) - 1))
			
		if (e==1):
			x_kepler = (a*math.cosh(E) + c)
			y_kepler = b*(math.sinh(E))

			t[i] = a*math.sqrt(a/alpha[i])*(e*math.sinh(E) + E)
			r_kepler[i] = np.float(a*(e*math.cosh(E) + 1))  

		
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
		
		'''
		if (a*a >= b*b):
			x_kepler = a*math.cos(E) - c
			y_kepler = b*math.sin(E) 
		 
		if (a*a < b*b):
			x_kepler = a*math.cos(E) + c  
			y_kepler = b*math.sin(E) 	
		'''
		
		'''		
		#Calculate the initial displacement due to the keplerian geometry. The same displacement is applied for all points

		if i == 0 :    
		 x_disp = -x_kepler[i] + q_rel[i,0] #
		 y_disp = -y_kepler[i] + q_rel[i,1]

		x_kepler[i] += x_disp
		y_kepler[i] += y_disp           

		#print(i, E[i], math.sin(E[i]))#r_dif[i], c, x_kepler[i] - q_rel[i,0], y_kepler[i] - q_rel[i,1], e[i])     
		'''
		
		q_analit_rel[i,0] = x_kepler
		q_analit_rel[i,1] = y_kepler
		q_analit_rel[i,2] = q_rel[i,2]	
		
		d_dif[i] = abs(math.sqrt(q_analit_rel[i,0]*q_analit_rel[i,0] + q_analit_rel[i,1]*q_analit_rel[i,1] + q_analit_rel[i,2]*q_analit_rel[i,2]) - d_rel) 
		
		#d_dif[i] = abs(r_kepler - d_rel) 

		'''
		# Dediu, Adrian-Horia; Magdalena, Luis; Martín-Vide, Carlos (2015). Theory and Practice of Natural Computing: Fourth International Conference, TPNC 2015, Mieres, Spain, December 15–16, 2015. Proceedings (illustrated ed.); Springer. [p. 141]

		#perihelion shift [#rad per revolution] 

		p_s[i] = 24*math.pi*math.pi*math.pi*a*a
		'''
		
		#apsidial precession [#rad per revolution]
		a_p[i] = 6*math.pi*G*M/(C*C*a*(1 - e*e))
		
		#t[i] = math.sqrt((4*math.pi*math.pi*a*a*a)/(G*M))
		
		#dal maggiore
		f_e = (1./((1. - e*e)**(7./2.)))*(1. + (73./24.)*(e*e) + (37./96.)*(e*e*e*e))
		
		P_quad[i] = -(((32./5.)*(G*G*G*G)*(mu*mu)*(M*M*M))/((a*a*a*a*a)*(C*C*C*C*C)))*f_e   
          
	q_rel_diff = q_rel - q_analit_rel
		
	#print (q_analit_rel, q_rel[0,:], q_analit_rel[0,:])

	return (d_dif, q_analit_rel, q_rel_diff, L, a_p, t, P_quad)

	
def kepler_sol_sys(p, q, Neff, H, m, dt):

	L_arr = np.zeros((len(m),Neff))


	d_rel = np.zeros(Neff, dtype='float64')	
	
	q_rel, p_rel, q_cm, p_cm = CM_system(p[0], p[1], q[0], q[1], Neff, m[0], m[1])
		
	for i in range(Neff):
		d_rel[i] = math.sqrt(q_rel[i,0]*q_rel[i,0] + q_rel[i,1]*q_rel[i,1] + q_rel[i,2]*q_rel[i,2])
	
	aphe_indexes = [index for index, item in enumerate(d_rel) if item == max(d_rel)]
	peri_indexes = [index for index, item in enumerate(d_rel) if item == min(d_rel)]
	
	e = (d_rel[aphe_indexes] - d_rel[peri_indexes])/(d_rel[aphe_indexes] + d_rel[peri_indexes])
	a = (d_rel[aphe_indexes] + d_rel[peri_indexes])/2.
	b = math.sqrt(1. - e*e)*a 
	
	print(e, a ,b)
	
	for i in range(len(m)):
	
		L_arr[i] = q[i,:,1]*p[i,:,2] - q[i,:,2]*p[i,:,1] - q[i,:,0]*p[i,:,2] + q[i,:,2]*p[i,:,0] + q[i,:,0]*p[i,:,1] - q[i,:,1]*p[i,:,0]		

	L_tot = np.sum(L_arr, axis=0)
	
	L = L_arr[0] + L_arr[1]

	q_analit_rel = np.array([[0 for i in range(0, 3)] for Neff in range(0, Neff)], dtype='float64')	
				   
	#Dinamica Kepleriana#------------------#
	M = m[0] + m[1]
	mu = (m[0]*m[1])/M

	'''
	H2 = np.array(H)*np.array(H)
	
	L2 = L*L

	k = -G*M*mu
	#k = G*(mu*mu*mu)*(M*M)
	R = L2/(-k*mu) # semi-latus rectum = a*(1 - e*e)
  
	alpha = H2/R
	'''
	 
	P_quad = np.zeros(Neff, dtype='float64')
	#e = np.zeros(Neff)
	#a = np.zeros(Neff, dtype='float64')
	#a_p = np.zeros(Neff, dtype='float64')		
	a_p1 = np.zeros(Neff, dtype='float64')	
	a_p2_arr = np.zeros(len(m), dtype='float64')
	a_p2 = np.zeros(Neff, dtype='float64')
	a_p3 = np.zeros(Neff, dtype='float64')	
	a_p4_arr = np.zeros(len(m), dtype='float64')
	a_p4 = np.zeros(Neff, dtype='float64')
	#T = np.zeros(Neff, dtype='float64')	
	t = np.zeros(Neff, dtype='float64')	
	r_kepler = np.zeros(Neff, dtype='float64')	
	
	for i in range(0, Neff):
		#t1 = np.float(k*k*mu)
		#t2 = np.float(H[i]*L2[i])	
		#a = 1. + (2.*t2)/(t1)
		
		#print(H[i]*L2[i], H[i], L2[i], (2.*H[i]*L2[i])/(k*k*mu))		
		
		#e = math.sqrt(1 + (2.*H[i]*L2[i])/(k*k*mu))# eccentricity
	
		#print(e)
		
		#a = R[i]/(1. - e*e) # semi-major axis
		#b = R[i]/(math.sqrt(1. - e*e)) # semi-minor axis
		
		'''	   
		if (1 <= e):
			a = (R[i]/(e*e - 1.)) # semi-major axis
			b = (R[i]/(math.sqrt(e*e - 1.))) # semi-minor axis   	
 		'''
 		
		#apsidial precession given by GR [#rad per revolution]
		a_p1[i] = 6.*math.pi*G*M/(C*C*a*(1. - e*e))
		
		#apsidial precession given by coupling of the planet with the other bodies  (newtonian) [#rad per revolution]

		for j in range(2, len(m)):	
							
			r_m = math.sqrt((q[1,i,0]-q[j,i,0])*(q[1,i,0]-q[j,i,0]) + (q[1,i,1]-q[j,i,1])*(q[1,i,1]-q[j,i,1]) + (q[1,i,2]-q[j,i,2])*(q[1,i,2]-q[j,i,2]))
			
			a_p2_arr[j] = (3.*math.pi/2.)*(m[j]/M)*(a/r_m)*(a/r_m)*(a/r_m)*(math.sqrt(1.- e*e))
		
		a_p2[i] = np.sum(a_p2_arr)
		
		#apsidial precession given by coupling the attractor and the other bodies [#rad per revolution]
		a_p3[i] = (1. + 0.5*(G*M*28. + 47.*e*e)/(C*C*a*(1. - e*e)*(1. - e*e)))
		
		#apsidial precession given by gravitomagnetic effect [#rad per revolution]
		for j in range(2, len(m)):	
		
			r_m = math.sqrt((q[1,i,0]-q[j,i,0])*(q[1,i,0]-q[j,i,0]) + (q[1,i,1]-q[j,i,1])*(q[1,i,1]-q[j,i,1]) + (q[1,i,2]-q[j,i,2])*(q[1,i,2]-q[j,i,2]))
			
			a_p4_arr[j] = 4.*math.pi*(G*m[j]/(C*C*a))*(a/r_m)*(a/r_m)*math.sqrt(a/r_m)
		
		a_p4[i] = np.sum(a_p4_arr)
		
		#t[i] = math.sqrt((4*math.pi*math.pi*a*a*a)/(G*M)) 
		#dal maggiore
		f_e = (1./((1. - e*e)**(7./2.)))*(1. + (73./24.)*(e*e) + (37./96.)*(e*e*e*e))
		
		P_quad[i] = -(((32./5.)*(G*G*G*G)*(mu*mu)*(M*M*M))/((a*a*a*a*a)*(C*C*C*C*C)))*f_e   
 
	return (L, P_quad, a_p1, a_p2, a_p3, a_p4)
	
