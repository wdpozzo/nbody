import numpy as np
import math
from nbody.CM_coord_system import CM_system

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
def kep_integrator(p1, p2, q1, q2, m, dt, ICN_it, H, Neff):

	nbodies = len(m)
	
	dt2 = 0.5*dt

	q_analit_rel = np.array([0 for i in range(0, 3)])
	 
	g = np.array([[0 for i in range(0, 3)] for Neff in range(0, Neff)], dtype='float64')

	_gradients_kep(g, p1, p2, q1, q2, nbodies, order)

	for i in range(ICN_it):   
		# FIXME: spins are not evolving!

		for k in range(nbodies):
			mass = m[k]
			mid_point[k].mass = mass 

			for j in range(3):

				tmp_b.q[j] = bodies[k].q[j] + dt2*g[k][3+j]
				mid_point[k].q[j] = 0.5*(tmp_b.q[j] + bodies[k].q[j])

				tmp_b.p[j] = bodies[k].p[j] - dt2*g[k][j]
				mid_point[k].p[j] = 0.5*(tmp_b.p[j] + bodies[k].p[j])

				tmp_b.s[j] = bodies[k].s[j]
				mid_point[k].s[j] = 0.5*(tmp_b.s[j] + bodies[k].s[j])

			# update the gradient
			for k in range(nbodies):
				memset(g[k], 0, 6*sizeof(long double))
			_gradients(g, mid_point, nbodies, order)

		#print(g[0][0])

	#calculate the final forward coordinates
	for i in range(nbodies):
		mass = bodies[i].mass

		for j in range(3):
			bodies[i].q[j] += dt2*g[i][3+j]
			bodies[i].p[j] -= dt2*g[i][j]
			#            bodies[i].s[j] =  dtsquare*g[i,j] #FIXME: spin evolution

	_free(mid_point)

	for i in range(nbodies):
		free(g[i])

	free(g);
	
	return
	
    def one_step_icn_cart(self,i):
    
    	  #iteration 0
    	  
        dhdp = dH_dposition(self.mu, self.p[i-1,:], self.q[i-1,:])
        dhdq = dH_dmomentum(self.m1, self.m2, self.p[i-1,:], self.q[i-1,:])
        
        q_1 = self.q[i-1,:] + self.dt2*dhdp
        q_1h = 0.5*(q_1 + self.q[i-1,:])
        p_1 = self.p[i-1,:] + self.dt2*dhdq
        p_1h = 0.5*(p_1 + self.p[i-1,:])
        
        #iteration 1
        dhdp = dH_dposition(self.mu, p_1h, q_1h)
        dhdq = dH_dmomentum(self.m1, self.m2, p_1h, q_1h)
                
        q_2 = self.q[i-1,:] + self.dt2*dhdp
        q_2h = 0.5*(q_2 + self.q[i-1,:]) 
        p_2 = self.p[i-1,:] + self.dt2*dhdq
        p_2h = 0.5*(p_2 + self.p[i-1,:])
    	  
        #solutions (2 iteration is the max number in order to have unconditional stability)
        dhdp = dH_dposition(self.mu, p_2h, q_2h)
        dhdq = dH_dmomentum(self.m1, self.m2, p_2h, q_2h)
        
        self.q[i,:] = self.q[i-1,:] + self.dt2*dhdp
        self.p[i,:] = self.p[i-1,:] + self.dt2*dhdq
   
		  # energy calculations
		  
        self.H[i] = self.H[i-1] + self.dt2*dHdt(self.m1, self.m2, self.p[i,:] self.q[i,:])
        self.energy_loss[i] = abs(self.H[0] - Energy_cart(self.m1, self.m2, self.p[i,:], self.q[i,:]))       
         
'''

def kepler(p1, p2, q1, q2, Neff, H, m, dt, ICN_it):

	L1 = q1[:,1]*p1[:,2] - q1[:,2]*p1[:,1] - q1[:,0]*p1[:,2] + q1[:,2]*p1[:,0] + q1[:,0]*p1[:,1] - q1[:,1]*p1[:,0]
		
	L2 = q2[:,1]*p2[:,2] - q2[:,2]*p2[:,1] - q2[:,0]*p2[:,2] + q2[:,2]*p2[:,0] + q2[:,0]*p2[:,1] - q2[:,1]*p2[:,0]
		
	L = L1 + L2

	
	q_rel, p_rel, q_cm, p_cm = CM_system(p1, p2, q1, q2, Neff, m[0], m[1])
		
	#L = q_rel[:,1]*p_rel[:,2] - q_rel[:,2]*p_rel[:,1] - q_rel[:,0]*p_rel[:,2] + q_rel[:,2]*p_rel[:,0] + q_rel[:,0]*p_rel[:,1] - q_rel[:,1]*p_rel[:,0]
	
	
	'''
	WORK IN PROGRESS
	'''

	#q1_analit = np.array([[0 for i in range(0, 3)] for Neff in range(0, Neff)], dtype='float64')
	#q2_analit = np.array([[0 for i in range(0, 3)] for Neff in range(0, Neff)], dtype='float64')

	#q1_dif = np.array([[0 for i in range(0, 3)] for Neff in range(0, Neff)], dtype='float64')
	#q2_dif = np.array([[0 for i in range(0, 3)] for Neff in range(0, Neff)], dtype='float64')
	
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

		#b = np.zeros(Neff, dtype='float64')
		#r_kepler = np.zeros(Neff, dtype='float64')
		#E = np.zeros(Neff)
		#phi_orb = np.zeros(Neff)
		#x_kepler = np.zeros(Neff, dtype='float64')
		#y_kepler = np.zeros(Neff, dtype='float64')
		#r_rel = np.zeros(Neff, dtype='float64')
		#r_dif = np.zeros(Neff, dtype='float64')   

		#q_analit_rel = np.array([[0 for i in range(0, 3)] for Neff in range(0, Neff)], dtype='float64')
		#q_rel_diff = np.array([[0 for i in range(0, 3)] for Neff in range(0, Neff)], dtype='float64')
		#Aggiungere routine per il calcolo dell'orbita kepleriana anche in caso di orbita iperbolica (https://kyleniemeyer.github.io/space-systems-notes/orbital-mechanics/two-body-problems.html)
		  
		#e = np.float(math.sqrt(1 + (2*H[i]*L2[i])/(mu*k*k*mu*mu))) # eccentricity
		e = np.float(math.sqrt(1 + (2*H[i]*L2[i])/(k*k*mu))) # eccentricity
		#a = np.float(R[i]/(1 - e*e)) # semi-major axis
		#b = np.float(R[i]/(math.sqrt(1 - e*e))) # semi-minor axis  	 
		
		if (1 > e):
			a = np.float(R[i]/(1 - e*e)) # semi-major axis
			b = np.float(R[i]/(math.sqrt(1 - e*e))) # semi-minor axis
			   
		if (1 <= e):
			a = np.float(R[i]/(e*e - 1)) # semi-major axis
			b = np.float(R[i]/(math.sqrt(e*e - 1))) # semi-minor axis   	
		
		c = e*a
		#c = np.float(math.sqrt(abs(a*a - b*b))) 
		
		#phi_orb = math.acos(q_rel[i,0]/d_rel) 		
		phi_orb = np.float(math.atan2(q_rel[i,1], q_rel[i,0])) # orbital angle		
		
		#r_kepler = a*(1 - e*e)/(1 + e*math.cos(phi_orb))
		#r_kepler[i] = np.float(R[i]/(1 + e*math.cos(phi_orb))) # r_phi keplerian equation
		
		E = 2*math.atan2(math.sqrt(1 - e)*math.tan(phi_orb/2), math.sqrt(1 + e)) # eccentric anomaly

		#r_kepler = a*(1 - e*math.cos(E)) 
 
		#E = np.float(math.acos(1/e - r_kepler[i]/(e*a))) 	
		#E = math.acos(q_rel[i,0]/a) 

		if (e<1):
			x_kepler = a*math.cos(E) - c
			y_kepler = b*math.sin(E)
			
			t[i] = a*math.sqrt(a/alpha[i])*(E - e*math.sin(E))
			r_kepler[i] = np.float(a*(1 - e*math.cos(E))) 

		if (e>1):
			x_kepler = -a*math.cosh(E) + c
			y_kepler = a*(math.sqrt(e*e -1))*(math.sinh(E))

			t[i] = a*math.sqrt(a/alpha[i])*(e*math.sinh(E) - E)
			r_kepler[i] = np.float(a*(e*math.cosh(E) - 1))
			
		if (e==1):
			x_kepler = +a*math.cosh(E) + c
			y_kepler = a*(math.sqrt(e*e -1))*(math.sinh(E))

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
        
	q_rel_diff = q_rel - q_analit_rel
	
	#print (q_analit_rel, q_rel[0,:], q_analit_rel[0,:])

	return (d_dif, q_analit_rel, q_rel_diff, L, a_p, r_kepler, t)

