import numpy as np
import math
#from nbody.body import body
#from nbody.hamiltonian import hamiltonian, gradients, kinetic_energy, potential
from nbody.engine import run, _H_2body
from Kep_dynamic import kepler, kepler_shift, kepler_sol_sys, KepElemToCart
from collections import deque 
from optparse import OptionParser
from nbody.CM_coord_system import CM_system, CartToSpher
import pickle
from scipy.signal import argrelextrema

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits import mplot3d
            
import random

import astropy.units as u
from datetime import datetime
from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, get_body_barycentric_posvel

'''
Precession of the perihelion of mercury = 43 seconds of arc per century (1 second of arch = 1/3600 degrees) 
Number of seconds in a century = 3153600000
More information at "https://math.ucr.edu/home/baez/physics/Relativity/GR/mercury_orbit.html"
'''

#python nbodysimulation.py -n 5 --steps 1280000000 --dt 2.5 --ICN_order 2 --PN_order 1

day = 86400. #*u.second
year = day*365

G = 6.67e-11 #*(u.meter**3)/(u.kilogram*u.second**2) # 6.67e-11 #

# AU**3/((d**2)*solMass) = (86400 * 86400) /( 2e30 * 1.5e11 * 1.5e11)

C = 299792458. #*(u.meter/u.second) #299792458. #
Ms = 1.988e30 #*(u.kilogram) # 1.988e30 #

Mmerc = 0.3301e24
Mearth = 5.97216787e24 
AU = 149597870700. #*u.meter
Ms = 1.98840987e30

def gaussian_random_sphere(x, y, z, r, num, bulge):

    x_a = np.zeros(num)
    y_a = np.zeros(num)
    z_a = np.zeros(num)
        
    for i in range(0, num):
        
        if (bulge == 0):
            factor = random.random()
        	    
        if (bulge == 1): 
            factor = min(1, max(0, abs(random.gauss(0.0, 0.5)))) #gaussian distribution
            #factor = random.random()
            print(factor)
               
        ir = r * factor
        itheta = np.arccos(np.random.uniform(-1, 1))
        iphi = np.random.uniform(0, 2 * math.pi)
        ix = x + ir * math.sin(itheta) * math.cos(iphi)
        iy = y + ir * math.sin(itheta) * math.sin(iphi)
        iz = z + ir * math.cos(itheta)
        
        x_a[i] = ix
        y_a[i] = iy
        z_a[i] = iz
                        
    return ((x_a).astype(np.longdouble), (y_a).astype(np.longdouble), (z_a).astype(np.longdouble))


if __name__=="__main__":

    parser = OptionParser()
    parser.add_option('-n', default=2, type='int', help='n bodies')
    parser.add_option('--steps', default= 5000000, type='long', help='n steps (must be a multiple of  1e7, which is the # of datas in a file solution fragment)') #, type='longint'
    parser.add_option('--PN_order', default=0, type='int', help='Post Newtonian approximation order')
    parser.add_option('--dt', default=1, type='float', help='dt')
    parser.add_option('-p', default = False, action = 'store_true', help='post process')
    parser.add_option('--animate', default=0, type='int', help='animate')
    parser.add_option('--plot', default=1, type='int', help='simulations plots')
    parser.add_option('--cm', default=1, type='int', help='orbit plot in CM system and angular momentum; requires n=2 !')
    parser.add_option('--seed', default=1, type='int', help='seed')
    parser.add_option('--ICN_order', default=2, type='int', help='ICN iteration number')
    (opts,args) = parser.parse_args()

    N = opts.steps
    nbodies = opts.n	
    ICN_it = opts.ICN_order
    order = opts.PN_order
    np.random.seed(opts.seed)        
    
    '''
    #points in generated randomly inside a sphere (bulge == 0) or with gaussian density distribution centered in the origin + a massive body there (bulge == 1)

    bulge = 1 
    
    m = np.random.uniform(5e-1*Ms, 1.e0*Ms, size = nbodies).astype(np.longdouble)
    
    x, y, z = gaussian_random_sphere(0, 0, 0, 25.0*AU, nbodies, bulge)

    vx = np.random.uniform(-5e-7, 5e-7, size = nbodies).astype(np.longdouble)
    vy = np.random.uniform(-5e-7, 5e-7, size = nbodies).astype(np.longdouble)
    vz = np.random.uniform(-5e-7, 5e-7, size = nbodies).astype(np.longdouble)
    
    sx = np.random.uniform(-1.0, 1.0, size = nbodies).astype(np.longdouble)
    sy = np.random.uniform(-1.0, 1.0, size = nbodies).astype(np.longdouble)
    sz = np.random.uniform(-1.0, 1.0, size = nbodies).astype(np.longdouble)
    
    if (bulge == 1): #aggiungo una massa "grande" al centro del cluster
        m = np.append(m, 5.e1*Ms)
        
        x = np.append(x, 0.)
        y = np.append(y, 0.)
        z = np.append(z, 0.)
        
        vx = np.append(vx, 0.)
        vy = np.append(vy, 0.)
        vz = np.append(vz, 0.) 
        
        sx = np.append(sx, 0.)
        sy = np.append(sy, 0.)
        sz = np.append(sz, 0.)    
                  
        nbodies = opts.n + 1 
    #print(x,y,z,vx,vy,vz,sx,sy,sz)  	
    '''

    '''
    #actual natural initial coordinates    
    t = Time(datetime.now()) #Time("2021-05-21 12:05:50", scale="tdb") #Time(datetime.now())    
    
    masses = {
    'sun'     : Ms, #1st planet has to be the central attractor
    'mercury' : Mmerc, #2nd planet has to be the one which we want to test the GR dynamics effects on 
    #'earth'   : Mearth,
    #'mars'    : 0.1075*Mearth,
    #'venus'   : 0.815*Mearth,
    #'jupiter' : 317.8*Mearth,
    #'saturn'  : 95.2*Mearth,
    #'uranus'  : 14.6*Mearth,
    #'neptune' : 17.2*Mearth,
    #'pluto'   : 0.00218*Mearth,
}

    planet_names = [
    'sun',
    'mercury',
    #'earth',
    #'mars',
    #'venus',
    #'jupiter',
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

    print('m=',m)
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

    #print(x,y,z,vx,vy,vz,sx,sy,sz)
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
    
    #'''
    m[0], m[1] = 1.e0*Ms, 1.e0*Mmerc
    
    x[0], x[1] =  0., 69.818e9 #Aphelion
    #x[0], x[1] =  0., 46.e9 #Perihelion
    y[0], y[1] = 0., 0.
    z[0], z[1] = 0., 0.

    vx[0], vx[1] = 0., 0.
    vy[0], vy[1] = 0., 38.86e3 #Aphelion
    #vy[0], vy[1] = 0., 58.98e3 #Perihelion
    vz[0], vz[1] = 0., 0.
    
    sx[0], sx[1] = 0., 0.
    sy[0], sy[1] = 0., 0.
    sz[0], sz[1] = 0., 0.
    #'''

    '''
    m[0], m[1] = 1.e0*Ms, 1.e0*Mearth
    
    x[0], x[1] =  0., 152.100e9
    y[0], y[1] = 0., 0.
    z[0], z[1] = 0., 0.

    vx[0], vx[1] = 0., 0.
    vy[0], vy[1] = 0., 29.29e3
    vz[0], vz[1] = 0., 0.
    
    sx[0], sx[1] = 0., 0.
    sy[0], sy[1] = 0., 0.
    sz[0], sz[1] = 0., 0.
    '''

    '''
    #random initial coordinates
    
    m = np.random.uniform(1e0, 1e0,size = nbodies).astype(np.longdouble)

    x = np.random.uniform(- 2.0, 2.0,size = nbodies).astype(np.longdouble)
    y = np.random.uniform(- 2.0, 2.0,size = nbodies).astype(np.longdouble)
    z = np.random.uniform(- 2.0, 2.0,size = nbodies).astype(np.longdouble)

    vx = np.random.uniform(-5e-7, 5e-7,size = nbodies).astype(np.longdouble)
    vy = np.random.uniform(-5e-7, 5e-7,size = nbodies).astype(np.longdouble)
    vz = np.random.uniform(-5e-7, 5e-7,size = nbodies).astype(np.longdouble)
    
    sx = np.random.uniform(-1.0,1.0,size = nbodies).astype(np.longdouble)
    sy = np.random.uniform(-1.0,1.0,size = nbodies).astype(np.longdouble)
    sz = np.random.uniform(-1.0,1.0,size = nbodies).astype(np.longdouble)
    #print(m,x,y,z,vx,vy,vz,sx,sy,sz)
    '''
    
    #parameters for solution files management 
  
    plot_step = 2000
    buffer_lenght = 2000000
    data_thin = 500

    #------------------------------------------------------#  
 
    dt = opts.dt
    N  = opts.steps
    Neff = int(N/(data_thin*plot_step))
    nout = int(N/buffer_lenght)    

    N_arr = np.linspace(0, N, Neff)

    #------------------------------------------------------#
    
    if not opts.p:
        run(N, np.longdouble(dt), opts.PN_order, m, x, y, z, m*vx, m*vy, m*vz, sx, sy, sz, ICN_it, data_thin, buffer_lenght, plot_step)
    
    s, H, T, V, D, t_sim = [], [], [], [], [], []
    s_long, D_long = [], [] #,  T_long, V_long, t_sim_long = [], [], [], []
    
    for i in range(nout):  
        s_tot, H_tot, T_tot, V_tot, t_tot, D_tot = [], [], [], [], [], []

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

        s_long.append(s_tot[0]) #[::plot_step])
        #H_long.append(H_tot[0]) #[::plot_step])
        #T_long.append(T_tot[0]) #[::plot_step])
        #V_long.append(V_tot[0]) #[::plot_step])
        D_long.append(D_tot[0]) #[::plot_step])
        #t_sim_long.append(t_tot) #[::plot_step])    
        
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
    D = np.array(D, dtype=object)
    t_sim = np.array(t_sim, dtype=object)
    
    s = np.concatenate((s[:]))
    H = np.concatenate((H[:]))
    T = np.concatenate((T[:]))
    V = np.concatenate((V[:]))
    D = np.concatenate((D[:]), axis=0)
    t_sim = np.concatenate((t_sim[:]), axis=0) 

    #
    s_long = np.array(s_long, dtype=object)#.flatten()
    #H_long = np.array(H_long, dtype=object)#.flatten()
    #T_long = np.array(T_long, dtype=object)#.flatten()
    #V_long = np.array(V_long, dtype=object)#.flatten()
    D_long = np.array(D_long, dtype=object)
    #t_sim_long = np.array(t_sim_long, dtype=object)
    
    s_long = np.concatenate((s_long[:]))
    #H_long = np.concatenate((H_long[:]))
    #T_long = np.concatenate((T_long[:]))
    #V_long = np.concatenate((V_long[:]))
    D_long = np.concatenate((D_long[:]), axis=0)
    #t_sim_long = np.concatenate((t_sim_long[:]), axis=0) 
    #
    
    if opts.animate == 1:
        
        f = plt.figure(figsize=(6,4))
        ax = f.add_subplot(111, projection = '3d')
        colors = cm.rainbow(np.linspace(0, 1, nbodies))
  
        trails = {}
        lines  = []
        symbols = []
        
        for b in range(nbodies):
            q = s[0][b]['q']
            trails[b] = deque(maxlen=200)
            trails[b].append(q)
            q_trail = np.array(trails[b])
            l, = ax.plot(q_trail[:,0],q_trail[:,1],q_trail[:,2], color=colors[b], alpha=0.5)
            lines.append(l)
            sym = ax.scatter(q[0],q[1],q[2], color=colors[b], s=10*s[0][b]['mass'])
            symbols.append(sym)
        
        ax.view_init(25, 10)
        ax.set(xlim=(-100, 100), ylim=(-100, 100), zlim=(-100,100))
        
        def animate_bodies(i,q,symbol):
            symbol._offsets3d = (np.atleast_1d(q[0]),
                                 np.atleast_1d(q[1]),
                                 np.atleast_1d(q[2]))
        
        def animate_trails(i,line,q_trail):
            line.set_data(q_trail[:,0],q_trail[:,1])
            line.set_3d_properties(q_trail[:,2])
        
        import sys
        def animate(i, s, lines, symbols):
            sys.stderr.write('{0}'.format(i))
            for b in range(nbodies):
                q = s[i][b]['q']
                trails[b].append(q)
                q_trail = np.array(trails[b])
                animate_trails(i,lines[b],q_trail)
                animate_bodies(i,q,symbols[b])
            return []
        
        anim = FuncAnimation(f, animate, Neff, fargs=(s, lines, symbols),
                             interval = 0.1, blit = True)
        Writer = writers['ffmpeg']
        writer = Writer(fps=120, metadata=dict(artist='Me'), bitrate=900, extra_args=['-vcodec', 'libx264'])
        anim.save('nbody.mp4', writer=writer)
           
    if opts.plot==1:
        
        f = plt.figure(figsize=(8,6))
        ax1 = f.add_subplot(111)
        #ax1.plot(N_arr, T, label = 'Kinetic energy')
        #ax1.plot(N_arr, V, label = 'Potential energy')
        ax1.plot(N_arr, H, label = 'Hamiltonian')
        ax1.set_xlabel('iteration')
        ax1.set_ylabel('Energy')
        ax1.legend()
        ax1.grid()
        plt.show()
                   
        colors = iter(cm.rainbow(np.linspace(0, 1, nbodies)))      
               
        qs = [[] for x in range(nbodies)]
            
        # this is the number of bodies active in each step of the solution
        
        nbodies = [len(si) for si in s]

        f = plt.figure(figsize=(6,4))
        ax = f.add_subplot(111)
        ax.plot(N_arr, nbodies)
        ax.set_xlabel('iteration')
        ax.set_ylabel('nbodies')
    
        f = plt.figure(figsize=(6,4))
        ax = f.add_subplot(111, projection = '3d')
        
        for i in range(0, Neff): #,plotting_step):
            for j in range(nbodies[i]):
                qs[j].append(s[i][j]['q'])
         
        for q in qs:
            q = np.array(q)# , dtype = 'object')
            #print(np.shape(q))
            c = next(colors)
            ax.plot(q[:,0], q[:,1], q[:,2], color=c, lw=0.5)
            ax.plot(q[:,0], q[:,1], q[:,2], color='w', alpha=0.5, lw=2, zorder=0)
                
        f.set_facecolor('black')
        ax.set_facecolor('black')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Now set color to white (or whatever is "invisible")
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
#        ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
#        ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
#        ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.zaxis.label.set_color('white')

        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.tick_params(axis='z', colors='white')
        # Bonus: To get rid of the grid as well:
        ax.grid(False)

        plt.show()
        
        if 0:

            f = plt.figure(figsize=(6,4))
            ax = f.add_subplot(111, projection = '3d')
            f.set_facecolor('black')
            ax.set_facecolor('black')
            colors = cm.rainbow(np.linspace(0, 1, nbodies[0]))
            trails = {}
            
            for b in range(nbodies[0]):
                trails[b] = deque(maxlen=500)

            for i in range(0, Neff): #,plotting_step):
                plt.cla()
                ax.set_title('H = {}'.format(H[i]), fontdict={'color':'w'}, loc='center')
                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False

                # Now set color to white (or whatever is "invisible")
                ax.xaxis.pane.set_edgecolor('w')
                ax.yaxis.pane.set_edgecolor('w')
                ax.zaxis.pane.set_edgecolor('w')
                #        ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
                #        ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
                #        ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.zaxis.label.set_color('white')

                ax.tick_params(axis='x', colors='white')
                ax.tick_params(axis='y', colors='white')
                ax.tick_params(axis='z', colors='white')
                # Bonus: To get  rid of the grid as well:
                ax.grid(False)
                
                for b in range(nbodies[0]):
                    q = s[i][b]['q']
                    trails[b].append(q)
                    q_trail = np.array(trails[b])
                    ax.scatter(q[0],q[1],q[2],color=colors[b],s=1e4*s[i][b]['mass'])
                    ax.plot(q_trail[:,0],q_trail[:,1],q_trail[:,2],color=colors[b],lw=0.5)
                    ax.plot(q_trail[:,0],q_trail[:,1],q_trail[:,2],color='w',alpha=0.5,lw=2,zorder=0)
    #            ax.set(xlim=(-50, 50), ylim=(-50, 50), zlim=(-50,50))
                plt.pause(0.00001)
                
            plt.show()
  
    if opts.cm == 1:
    
        if opts.n == 2:
            
            q1 = np.array([[0 for i in range(0, 3)] for Neff in range(0, Neff)], dtype='longdouble')
            p1 = np.array([[0 for i in range(0, 3)] for Neff in range(0, Neff)], dtype='longdouble')
            s1 = np.array([[0 for i in range(0, 3)] for Neff in range(0, Neff)], dtype='longdouble')
            q2 = np.array([[0 for i in range(0, 3)] for Neff in range(0, Neff)], dtype='longdouble')
            p2 = np.array([[0 for i in range(0, 3)] for Neff in range(0, Neff)], dtype='longdouble')       
            s2 = np.array([[0 for i in range(0, 3)] for Neff in range(0, Neff)], dtype='longdouble')   

            q1_long = np.array([[0 for i in range(0, 3)] for k in range(0, int(N/data_thin))], dtype='longdouble')
            p1_long = np.array([[0 for i in range(0, 3)] for k in range(0, int(N/data_thin))], dtype='longdouble')
            q2_long = np.array([[0 for i in range(0, 3)] for k in range(0, int(N/data_thin))], dtype='longdouble')
            p2_long = np.array([[0 for i in range(0, 3)] for k in range(0, int(N/data_thin))], dtype='longdouble')           
                      
            for i in range(0, Neff):

                q1[i,:] = s[i][0]['q']
                p1[i,:] = s[i][0]['p']
                s1[i,:] = s[i][0]['s']

                q2[i,:] = s[i][1]['q']
                p2[i,:] = s[i][1]['p']
                s2[i,:] = s[i][1]['s']                      

            #print(len(s[:]), len(s[:]), N, np.shape(q1_long), np.shape(q1))

            for i in range(0, int(N/data_thin)):

                q1_long[i,:] = s_long[i][0]['q']
                p1_long[i,:] = s_long[i][0]['p']

                q2_long[i,:] = s_long[i][1]['q']
                p2_long[i,:] = s_long[i][1]['p']


            q_rel, p_rel, q_cm, p_cm = CM_system(p1, p2, q1, q2, Neff, m[0], m[1])        
            r_sim = np.sqrt(q_rel[:,0]*q_rel[:,0] + q_rel[:,1]*q_rel[:,1] + q_rel[:,2]*q_rel[:,2])

            N_shift = 6.488703338e-5

            r_dif, q_an_rel, r_kepler, L, a_p, P_quad = kepler(q1, q2, p1, p2, N, Neff, H, m, dt, order)

            #r_dif, q_an_rel, r_kepler, L, a_p, P_quad, phi_shift, phi_shift_test, Dq_shift, q_peri = kepler_shift(q1, q2, p1, p2, D_long, N, Neff, int(N/data_thin), H, m, dt, order, q1_long, q2_long, p1_long, p2_long) #peri_indexes, Dq_shift
            
                   
            #p_s = np.sum(phi_shift)
            #p_s = p_s/Neff 

            a_shift = np.sum(a_p)
            a_shift = a_shift/Neff 
            
            #-------------------  Plots  ------------------------#   
            '''
            f = plt.figure(figsize=(16,6))
            
            ax = f.add_subplot(131, projection = '3d')  
            ax.plot(q_rel[:,0], q_rel[:,1], q_rel[:,2], label = 'Numerical solution', alpha=0.9)
            ax.plot(q_rel[0,0], q_rel[0,1], q_rel[0,2], 'o', label = 'Num. starting point', alpha=0.9)     
            ax.plot(q_an_rel[:,0], q_an_rel[:,1], q_an_rel[:,2], label = 'Analitical solution')
            ax.plot(q_an_rel[0,0], q_an_rel[0,1], q_an_rel[0,2], 'o', label = 'Analit. starting point')
            ax.plot(q_cm[:,0], q_cm[:,1], q_cm[:,2], 'o', label = 'CM')
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            ax.set_zlabel('z [m]')
            plt.legend()
            
            ax1 = f.add_subplot(132)
            ax1.plot(N_arr, r_dif, label = 'Analitycal vs. Numerical', alpha=0.9)
            ax1.set_xlabel('iterations')
            ax1.set_ylabel('Orbital radial difference [m]')
            plt.grid()
            plt.legend()
            
            ax2 = f.add_subplot(133)
            ax2.plot(N_arr, P_quad, label = 'Quadrupole power loss', alpha=0.9)
            ax2.set_xlabel('iterations')
            ax2.set_ylabel('Power [J/s]')
            plt.grid()
            plt.legend()
            
            plt.show()
            '''

            #N_arr_long = np.linspace(0,N,int(N/data_thin))

            f = plt.figure(figsize=(16,8))
            
            ax1 = f.add_subplot(121)
            ax1.plot(N_arr, r_kepler - r_sim, label = 'Analitycal vs. Numerical', alpha=0.9)
            ax1.set_xlabel('Iteration', fontsize = 'x-large')
            ax1.set_ylabel('Orbital radial difference [m]', fontsize = 'x-large')
            plt.legend(fontsize = 'large')
            plt.grid()
            
            ax2 = f.add_subplot(122)
            #ax1.plot(N_arr, T, label = 'Kinetic energy')
            #ax1.plot(N_arr, V, label = 'Potential energy')
            ax2.plot(N_arr, H, label = 'Hamiltonian')
            ax2.set_xlabel('Iteration', fontsize = 'x-large')
            ax2.set_ylabel('Energy [J]', fontsize = 'x-large')
            plt.legend(fontsize = 'large')
            plt.grid()

            plt.show()

            f = plt.figure(figsize=(8,6))
            ax1 = f.add_subplot(111)
            #ax1.plot(N_arr, T, label = 'Kinetic energy')
            #ax1.plot(N_arr, V, label = 'Potential energy')
            ax1.plot(N_arr, (H[0] - H)/H, label = 'Hamiltonian variation')
            ax1.set_xlabel('iteration')
            ax1.set_ylabel(r'$\Delta H / H$')
            ax1.legend()
            ax1.grid()
            plt.show()

            f = plt.figure(figsize=(16,8))
            
            ax = f.add_subplot(131) 
            ax.plot(N_arr, r_dif, label = 'r_rel vs. r_kepler', alpha=0.9)
            ax.set_xlabel('iterations', fontsize="x-large")
            ax.set_ylabel('Orbital displacement [m]', fontsize="x-large")
            plt.grid()
            
            ax1 = f.add_subplot(132)
            ax1.plot(N_arr, L)
            ax1.set_xlabel('iteration', fontsize="x-large")
            ax1.set_ylabel('Angolar Momentum', fontsize="x-large")
            plt.grid()

            ax2 = f.add_subplot(133)
            ax2.plot(N_arr, P_quad, label = 'Quadrupole power loss', alpha=0.9)
            ax2.set_xlabel('iterations', fontsize="x-large")
            ax2.set_ylabel('Power [J/s]', fontsize="x-large")
            plt.grid()
            plt.legend(fontsize="large")
            
            plt.show()
            
            '''
            f = plt.figure(figsize=(6,4))
            ax = f.add_subplot(111)
            ax.plot(N_arr, t)
            ax.set_xlabel('iteration')
            ax.set_ylabel('Orbital period')
            ax.grid()
            
            plt.show()
            '''

            #col_rainbow = cm.rainbow(np.linspace(0, 1, len(q_peri)))   
            
            '''
            f = plt.figure(figsize=(16,6))
            
            ax = f.add_subplot(111, projection = '3d')
            #ax.plot(q_rel[:,0], q_rel[:,1], q_rel[:,2], label = 'Numerical solution', alpha=0.5)  
            for i in range(0, len(q_peri)): 
                ax.plot(q_peri[i,0], q_peri[i,1], q_peri[i,2], 'o', label = 'Perihelion orbit {}'.format(i), color = col_rainbow[i])
            ax.set_xlabel('x [m]', fontsize="x-large")
            ax.set_ylabel('y [m]', fontsize="x-large")
            ax.set_zlabel('z [m]', fontsize="x-large")        
            plt.legend(fontsize="large")
            
            plt.show()                  
            '''
            
            r_spher_rel, theta_spher_rel, phi_spher_rel= CartToSpher(np.ascontiguousarray(q_rel[:,0]), np.ascontiguousarray(q_rel[:,1]), np.ascontiguousarray(q_rel[:,2]))
                
            f = plt.figure(figsize=(16,14))
            ax1 = f.add_subplot(131)
            ax1.plot(N_arr, r_spher_rel)
            ax1.set_xlabel('iteration', fontsize="x-large")
            ax1.set_ylabel('Relative distance [m]', fontsize="x-large")
            plt.grid()

            ax2 = f.add_subplot(132)
            ax2.plot(N_arr, theta_spher_rel, label = '')
            ax2.set_xlabel('iteration', fontsize="x-large")
            ax2.set_ylabel('Righ ascension [rad]', fontsize="x-large")
            plt.grid()

            ax3 = f.add_subplot(133)
            ax3.plot(N_arr, phi_spher_rel, label = '')
            ax3.set_xlabel('iteration', fontsize="x-large")
            ax3.set_ylabel('Declination [rad]', fontsize="x-large")
            plt.grid()
            plt.show()
 

            f = plt.figure(figsize=(16,16))

            ax1 = f.add_subplot(121)
            ax1.plot(N_arr, D[:,1, 0], label= "Numerical Error")
            ax1.set_xlabel('iteration', fontsize="x-large")
            ax1.set_ylabel(r'$\Delta x$ [m]', fontsize="x-large")
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.grid() 
            plt.legend(fontsize="large")

            ax2 = f.add_subplot(122)
            ax2.plot(N_arr, D[:,1,1], label= "Numerical Error")
            ax2.set_xlabel('iteration', fontsize="x-large")
            ax2.set_ylabel(r'$\Delta y$ [m]', fontsize="x-large")
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.grid()
            plt.legend(fontsize="large")

            plt.show()       


            q2_an = q1 - q_an_rel     
            
            D_q2_an = np.array([[0 for i in range(0, 3)] for Neff in range(0, Neff)], dtype='longdouble')#.astype(np.longdouble) 

            for i in range(Neff):
       
                D_q2_an[i, 0] = np.sqrt(D[i,0, 0]*D[i,0, 0] + (np.sqrt(D[i,0, 0]*D[i,0, 0] + D[i,1, 0]*D[i,1, 0])))
                D_q2_an[i, 1] = np.sqrt(D[i,0, 1]*D[i,0, 1] + (np.sqrt(D[i,0, 1]*D[i,0, 1] + D[i,1, 1]*D[i,1, 1])))
                D_q2_an[i, 2] = np.sqrt(D[i,0, 2]*D[i,0, 2] + (np.sqrt(D[i,0, 2]*D[i,0, 2] + D[i,1, 2]*D[i,1, 2])))
            
            f = plt.figure(figsize=(16,6))
            ax = f.add_subplot(111)
            ax.errorbar(q2[:,0], q2[:,1], yerr = D[:,1, 1], xerr = D[:,1, 0], label = 'Numerical solution', alpha=0.7)  
            ax.errorbar(q2_an[:,0], q2_an[:,1], yerr = D_q2_an[:, 1], xerr = D_q2_an[:,0], label = 'Analytical solution')
            ax.set_xlabel('x [m]', fontsize="x-large")
            ax.set_ylabel('y [m]', fontsize="x-large")      
            plt.legend(fontsize="large")
            
    
            print('Newtonian shift {} [rad/rev]'.format(N_shift)) 
            print('Standard GR shift = {} [rad/revolution]'.format(a_shift))
            #print('Perihelion shift = {}'.format(a_p[-1]*415.2))
            
            print('Numerical perihelion shift: {} [rad/revolution]'.format(p_s))
            print('Numerical perihelion shift (test): {} +- {} [rad/revolution]'.format(phi_shift_test, Dq_shift)) #Dq_shift))
            
        else :

            H_2body = []
            V_2body = []
            T_2body = []   
            
            mass = np.array([m[0], m[1]]).astype(np.longdouble)
                    
            q = np.array([[[0 for i in range(0, 3)] for Neff in range(0, Neff)] for m in range(0,len(m))], dtype='longdouble')#.astype(np.longdouble)
            p = np.array([[[0 for i in range(0, 3)] for Neff in range(0, Neff)] for m in range(0,len(m))], dtype='longdouble')#.astype(np.longdouble)
            spn = np.array([[[0 for i in range(0, 3)] for Neff in range(0, Neff)] for m in range(0,len(m))], dtype='longdouble')#.astype(np.longdouble)
     
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


            for i in range(0, Neff):
            
                x = np.array([q[0,i,0],q[1,i,0]]).astype(np.longdouble)
                y = np.array([q[0,i,1],q[1,i,1]]).astype(np.longdouble)
                z = np.array([q[0,i,2],q[1,i,2]]).astype(np.longdouble)
            
                px = np.array([p[0,i,0],p[1,i,0]]).astype(np.longdouble)
                py = np.array([p[0,i,1],p[1,i,1]]).astype(np.longdouble)
                pz = np.array([p[0,i,2],p[1,i,2]]).astype(np.longdouble)
            
                sx = np.array([spn[0,i,0],spn[1,i,0]]).astype(np.longdouble)
                sy = np.array([spn[0,i,1],spn[1,i,1]]).astype(np.longdouble)
                sz = np.array([spn[0,i,2],spn[1,i,2]]).astype(np.longdouble)
                
                #print(mass,x,y,z,px,py,pz,sx,sy,sz)
                               
                h, t, v = _H_2body(mass, x, y, z, px, py, pz, sx, sy, sz, order)
                
                H_2body.append(h)
                T_2body.append(t)
                V_2body.append(v)
                               
            L, P_quad, a_p1, a_p2, a_p3, a_p4, q_peri, phi_shift, phi_shift_test = kepler_sol_sys(p, q, Neff, H_2body, m, dt, order)
            
            #r = np.sqrt(q_rel[:,0]*q_rel[:,0] + q_rel[:,1]*q_rel[:,1] + q_rel[:,2]*q_rel[:,2])
            q_rel, p_rel, q_cm, p_cm = CM_system(p[0], p[1], q[0], q[1], Neff, m[0], m[1])
            
            r_sim = np.sqrt(q_rel[:,0]*q_rel[:,0] + q_rel[:,1]*q_rel[:,1] + q_rel[:,2]*q_rel[:,2])
            
           	#perihelion total shift
            
            N_shift = 6.488703338e-5
            p_s = a_p1 + a_p2*a_p3 + a_p4
            p_s_t = np.sum(p_s)
            p_s_t = p_s_t/Neff
            
            p_shift = np.sum(phi_shift)
            p_shift = p_shift/Neff 

            #-----------Plots-----------------#
            
            #print(a_p1, a_p2, a_p3, a_p4)
            
            f = plt.figure(figsize=(16,6))
            
            ax = f.add_subplot(121, projection = '3d')  
            #ax.title(r"$m_{1} = {}$, $m_{2} = {}$".format(m[0], m[1]))
            ax.plot(q_rel[:,0], q_rel[:,1], q_rel[:,2], label = 'Numerical solution', alpha=0.5)
            ax.plot(q_rel[0,0], q_rel[0,1], q_rel[0,2], 'o', label = 'Num. starting point', alpha=0.9)
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            ax.set_zlabel('z [m]')        
            plt.legend()

            ax1 = f.add_subplot(122)
            ax1.plot(N_arr, p_s, label = 'Total', alpha=0.9)
            ax1.plot(N_arr, a_p1, label = 'GR standard precession', alpha=0.9)
            ax1.plot(N_arr, a_p2*a_p3, label = 'Coupling of Sun-Mercury system with other planets', alpha=0.9)
            #ax3.plot(N_arr, a_p3, label = 'Coupling of Sun and other planets', alpha=0.9)
            ax1.plot(N_arr, a_p4, label = 'Gravitomagnetic effect', alpha=0.9)
            ax1.set_xlabel('iterations')
            ax1.set_xscale('log')
            ax1.set_ylabel('Perihelion shift [rad/revolution]')
            ax1.set_yscale('log')
            plt.grid()
            plt.legend()      

            plt.show()

            f = plt.figure(figsize=(16,6))
            ax1 = f.add_subplot(121)
            ax1.plot(N_arr, L)
            ax1.set_xlabel('iterations')
            ax1.set_ylabel('Total angolar momentum')
            plt.grid()
            
            ax2 = f.add_subplot(122)
            ax2.plot(N_arr, P_quad, label = 'Quadrupole power loss', alpha=0.9)
            ax2.set_xlabel('iterations')
            ax2.set_ylabel('Power [J/s]')
            plt.grid()
            plt.legend()
            
            plt.show()
            
            col_rainbow = cm.rainbow(np.linspace(0, 1, len(q_peri)))   
         
            f = plt.figure(figsize=(16,6))
            
            ax = f.add_subplot(111, projection = '3d')
            ax.scatter(q[1,:,0], q[1,:,1], q[1,:,2], label = 'Numerical solution', alpha=0.5)
            for i in range(0, len(q_peri)): 
            	ax.plot(q_peri[i,0], q_peri[i,1], q_peri[i,2], 'o', label = 'Perihelion orbit {}'.format(i), color = col_rainbow[i])
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            ax.set_zlabel('z [m]')        
            
            plt.legend()
            plt.show()         
            
            print('GR standard shift = {} [rad/rev];\nCoupling with other planets shift = {} [rad/rev];\nGravitomagnetic shift = {} [rad/rev].'.format(np.sum(a_p1)/Neff, (np.sum(a_p2)/Neff)*(np.sum(a_p3)/Neff), np.sum(a_p4)/Neff))
            print('Total theorethical shift = {} [rad/rev];\nNumerical shift = {} [rad/rev];\nNumerical shift (test) = {} [rad/rev].'.format(p_s_t, p_shift, phi_shift_test))
            
            print('Newtonian shift {} [rad/rev]'.format(N_shift))
            #print('Numerical shift: {} [rad]'.format(phi_shift))
        
        #if (opts.n!= 2):
        #    print("n do not equal 2: no CM plot")
