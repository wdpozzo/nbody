import numpy as np
import math
#from nbody.body import body
#from nbody.hamiltonian import hamiltonian, gradients, kinetic_energy, potential
from nbody.engine import run
from Kep_dynamic import kepler
from collections import deque
from optparse import OptionParser
from nbody.CM_coord_system import CM_system
import pickle

import astropy.units as u
from datetime import datetime
from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, get_body_barycentric_posvel


'''
Precession of the perihelion of mercury = 43 seconds of arc per century (1 second of arch = 1/3600 degrees) 
Number of seconds in a century = 3153600000
More information at "https://math.ucr.edu/home/baez/physics/Relativity/GR/mercury_orbit.html"
'''

day = 86400. #*u.second

G = 6.67e-11 #*(u.meter**3)/(u.kilogram*u.second**2) # 6.67e-11 #

# AU**3/((d**2)*solMass) = (86400 * 86400) /( 2e30 * 1.5e11 * 1.5e11)

C = 299792458. #*(u.meter/u.second) #299792458. #
Ms = 1.988e30 #*(u.kilogram) # 1.988e30 #

Mmerc = 0.3301e24
Mearth = 5.9722e24 
AU = 149597870700. #*u.meter

#G = (6.67e-11*u.m**3/(u.kg*u.s**2)).to(u.AU**3/(u.d**2*u.solMass)).value #G = 6.67e-11 

# AU**3/((d**2)*solMass) = (86400 * 86400) /( 2e30 * 1.5e11 * 1.5e11)

#C = (299792458.*(u.m/u.s)).to(u.AU/u.d).value #299792458. 
#Msun = (1.988e30*u.kg).to(u.solMass).value 

if __name__=="__main__":
    parser = OptionParser()
    parser.add_option('-n', default=2, type='int', help='n bodies')
    parser.add_option('--steps', default=5000000, type='long', help='n steps (must be >= 1e7, which is the n. of datas in a file solution fragment)') #, type='longint'
    parser.add_option('--PN_order', default=0, type='int', help='Post Newtonian approximation order')
    parser.add_option('--dt', default=1, type='float', help='dt')
    parser.add_option('-p', default = False, action = 'store_true', help='post process')
    parser.add_option('--animate', default=0, type='int', help='animate')
    parser.add_option('--plot', default=1, type='int', help='simulations plots')
    parser.add_option('--cm', default=1, type='int', help='orbit plot in CM system and angular momentum; requires n=2 !')
    parser.add_option('--seed', default=1, type='int', help='seed')
    parser.add_option('--ICN_order', default=2, type='int', help='ICN iteration number')
    (opts,args) = parser.parse_args()

    nbodies = opts.n	
    ICN_it = opts.ICN_order
    np.random.seed(opts.seed)    
    
    '''
    #actual natural initial coordinates    
    t = Time(datetime.now())
    
    masses = {
    'sun'     : Ms,
    #'earth'   : Mearth,
    'mercury' : Mmerc,
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
    #'earth',
    'mercury',
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

    print(x,y,z,vx,vy,vz,sx,sy,sz)
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

    
    m[0], m[1] = Ms, Mmerc
    
    x[0], x[1] = -0.*AU, 69.818e9
    y[0], y[1] = 0.*AU, 0.*AU
    z[0], z[1] = 0.*AU, 0.*AU

    vx[0], vx[1] = -0., -0.
    vy[0], vy[1] = 0., -38.86e3
    vz[0], vz[1] = 0., 0.
    
    sx[0], sx[1] = 0., 0.
    sy[0], sy[1] = 0., 0.
    sz[0], sz[1] = 0., 0.

    #print(x,y,z,vx,vy,vz,sx,sy,sz)

    
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
    plot_step = 500
    buffer_lenght = 10000000
    data_thin = 10
    #---------------------------------------#
    
    dt = opts.dt
    N  = opts.steps
    Neff = int(N/(data_thin*plot_step))
    nout = int(N/buffer_lenght)    
    
    if not opts.p:
        run(N, np.longdouble(dt), opts.PN_order, m, x, y, z, m*vx, m*vy, m*vz, sx, sy, sz, ICN_it, data_thin, buffer_lenght)
    
    s, H, T, V = [], [], [], []
    
    for i in range(nout):  
        s_tot, H_tot, T_tot, V_tot = [], [], [], []
        
        s_tot.append(pickle.load(open('solution_{}.pkl'.format(i),'rb')))
        H_tot.append(pickle.load(open('hamiltonian_{}.pkl'.format(i),'rb')))
        T_tot.append(pickle.load(open('kinetic_{}.pkl'.format(i),'rb')))
        V_tot.append(pickle.load(open('potential_{}.pkl'.format(i),'rb')))       
        
        #print(np.shape(s_tot), np.shape(H_tot), np.shape(T_tot), np.shape(V_tot))
        
        s.append(s_tot[0][::plot_step])
        H.append(H_tot[0][::plot_step])
        T.append(T_tot[0][::plot_step])
        V.append(V_tot[0][::plot_step])
       
        #print(np.shape(s), np.shape(H), np.shape(T), np.shape(V))
        
        del s_tot
        del H_tot
        del T_tot
        del V_tot
        
        if (1+i) % (10*nout)//100 == 0 :
            print("Data deframmentation: {}%".format((100*i)/nout))
    
    #print(np.shape(s), np.shape(H), np.shape(T), np.shape(V))
    
    s = np.array(s, dtype=object)#.flatten()
    H = np.array(H, dtype=object)#.flatten()
    T = np.array(T, dtype=object)#.flatten()
    V = np.array(V, dtype=object)#.flatten()
    
    #print(np.shape(s), np.shape(H), np.shape(T), np.shape(V))
    
    s = np.concatenate((s[:]))
    H = np.concatenate((H[:]))
    T = np.concatenate((T[:]))
    V = np.concatenate((V[:])) 
    
    #print(np.shape(s), np.shape(H), np.shape(T), np.shape(V))      
    #print(N, Neff, nout)
    
    if opts.animate == 1:
    
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from mpl_toolkits import mplot3d
        from matplotlib.animation import FuncAnimation, writers
        
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
    
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from mpl_toolkits import mplot3d
        
        N_arr = np.linspace(0, N, Neff)
        #plotting_step = np.maximum(64, Neff//int(0.1*Neff))
        
        f = plt.figure(figsize=(6,4))

        ax1 = f.add_subplot(121)
        ax1.plot(N_arr, T, label = 'Kinetic energy')
        ax1.plot(N_arr, V, label = 'Potential energy')
        ax1.set_xlabel('iteration')
        ax1.set_ylabel('Energy')
        ax1.legend()
        ax1.grid()
                
        ax2 = f.add_subplot(122)
        ax2.plot(N_arr, H, label = 'Hamiltonian')
        ax2.set_xlabel('iteration')
        ax2.grid()
           
        colors = iter(cm.rainbow(np.linspace(0, 1, nbodies)))      
                
        qs = [[] for x in range(nbodies)]
            
        # this is the number of bodies active in each step of the solution
        
        nbodies = [len(si) for si in s]

        f = plt.figure(figsize=(6,4))
        ax = f.add_subplot(111)
        ax.plot(range(Neff), nbodies)
        ax.set_xlabel('iteration')
        ax.set_ylabel('nbodies')
    
        f = plt.figure(figsize=(6,4))
        ax = f.add_subplot(111, projection = '3d')
        
        for i in range(0,Neff): #,plotting_step):
            for j in range(nbodies[i]):
                qs[j].append(s[i][j]['q'])
        
        #print(np.shape(qs))
         
        for q in qs:
            q = np.array(q) #/AU
            c = next(colors)
            ax.plot(q[:,0],q[:,1],q[:,2],color=c,lw=0.5)
            ax.plot(q[:,0],q[:,1],q[:,2], color='w', alpha=0.5, lw=2, zorder=0)
                

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
                # Bonus: To get rid of the grid as well:
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
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from mpl_toolkits import mplot3d
        
        N_arr = np.linspace(0, N, Neff)
        
        q_rel = np.array([[0 for i in range(0, 3)] for Neff in range(0, Neff)], dtype='float64')
        p_rel = np.array([[0 for i in range(0, 3)] for Neff in range(0, Neff)], dtype='float64')
        
        q1 = np.array([[0 for i in range(0, 3)] for Neff in range(0, Neff)], dtype='float64')
        p1 = np.array([[0 for i in range(0, 3)] for Neff in range(0, Neff)], dtype='float64')
        q2 = np.array([[0 for i in range(0, 3)] for Neff in range(0, Neff)], dtype='float64')
        p2 = np.array([[0 for i in range(0, 3)] for Neff in range(0, Neff)], dtype='float64')        
      
        for i in range(0, Neff):
        
            q1[i,:] = s[i][0]['q']
            p1[i,:] = s[i][0]['p']
            q2[i,:] = s[i][1]['q']
            p2[i,:] = s[i][1]['p']
        
        q_rel, p_rel = CM_system(p1, p2, q1, q2, Neff)          

        r_dif, q_an_rel, q_rel_diff, L, a_p = kepler(p1, p2, q1, q2, Neff, H, m)
        
        #perihelion total shift
        p_s = np.sum(a_p)
        
	     #-----------Plots-----------------#
	     
        f = plt.figure(figsize=(16,6))
        ax = f.add_subplot(121, projection = '3d')  
        #ax.title(r"$m_{1} = {}$, $m_{2} = {}$".format(m[0], m[1]))
        ax.plot(q_rel[:,0], q_rel[:,1], q_rel[:,2], label = 'Numerical solution', alpha=0.9)
        ax.plot(q_rel[0,0], q_rel[0,1], q_rel[0,2], 'o', label = 'Num starting point', alpha=0.9)     
        ax.plot(q_an_rel[:,0], q_an_rel[:,1], q_an_rel[:,2], label = 'Analitical solution')
        ax.plot(q_an_rel[0,0], q_an_rel[0,1], q_an_rel[0,2], 'o', label = 'Analit starting point')
        #ax.plot(q_rel[-1,0], q_rel[-1,1], q_rel[-1,2], 'o', label = 'Num ending point', alpha=0.9)   
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        plt.legend()

        #ax.set_xlim(min(q_rel[:,0]), max(q_rel[:,0]))
        #ax.set_ylim(min(q_rel[:,0]) - max(q_rel[:,0])/2, max(q_rel[:,0]))
        #ax.set_ylim(min(q_rel[:,2]), max(q_rel[:,2]))
        #plt.axis('auto')
        
        ax1 = f.add_subplot(122)
        ax1.plot(N_arr, r_dif, label = 'Analitycal vs. Numerical', alpha=0.9)
        ax1.set_xlabel('iterations')
        ax1.set_ylabel('Orbital radius difference [m]')
        plt.grid()
        plt.legend()                         
        
        plt.show()
        
        f = plt.figure(figsize=(16,6))
        
        ax2 = f.add_subplot(121) 
        ax2.plot(N_arr, q_rel_diff[:,0], label = 'Simul vs. Analit y coordinate', alpha=0.9)
        ax2.set_xlabel('iterations')
        ax2.set_ylabel('Displacement [m]')
        plt.grid()
        plt.legend()
        
        ax3 = f.add_subplot(122)
        ax3.plot(N_arr, q_rel_diff[:,1], label = 'Simul vs. Analit x coordinate', alpha=0.9)
        ax3.set_xlabel('iterations')
        ax3.set_ylabel('Displacement [m]')
        plt.grid()
        plt.legend()                    
        plt.show()
       
        f = plt.figure(figsize=(6,4))
        ax = f.add_subplot(111)
        ax.plot(N_arr, a_p)
        ax.set_xlabel('iteration')
        ax.set_ylabel('Apsidial precession [rad x revolution]')
        ax.grid()
        plt.show()
        
        f = plt.figure(figsize=(6,4))
        ax = f.add_subplot(111)
        ax.plot(N_arr, L)
        ax.set_xlabel('iteration')
        ax.set_ylabel('Angolar Momentum')
        ax.grid()
        plt.show()
        
        print('Perihelion shift = {}'.format(p_s))
        #print('Perihelion shift = {}'.format(a_p[-1]*415.2))
