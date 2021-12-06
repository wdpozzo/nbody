import numpy as np
#from nbody.body import body
#from nbody.hamiltonian import hamiltonian, gradients, kinetic_energy, potential
from nbody.engine import run
from collections import deque
from optparse import OptionParser
from nbody.CM_coord_system import CM_system
import pickle

#import astropy.units as u

G = 6.67e-11 #(6.67e-11*u.m**3/(u.kg*u.s**2)).to(u.AU**3/(u.d**2*u.solMass)).value #* (86400 * 86400) /( 2e30 * 1.5e11 * 1.5e11)
C = 3.0e8 #(3.0e8*(u.m/u.s)).to(u.AU/u.d).value
Ms = 2.0e30 #(2e30*u.kg).to(u.solMass).value
#GM = 1.32712440018e20

au = 149597870700. 

if __name__=="__main__":
    parser = OptionParser()
    parser.add_option('-n', default=2, type='int', help='n bodies')
    parser.add_option('--steps', default=5000000, type='int', help='n steps (must be equal or greater than 5e6')
    parser.add_option('--PN_order', default=0, type='int', help='PN order')
    parser.add_option('--dt', default=1, type='float', help='dt')
    parser.add_option('-p', default = False, action = 'store_true', help='post process')
    parser.add_option('--animate', default=0, type='int', help='animate')
    parser.add_option('--plot', default=1, type='int', help='simulations plots')
    parser.add_option('--cm', default=1, type='int', help='orbit plot in CM system and angular momentum')
    parser.add_option('--seed', default=1, type='int', help='seed')
    parser.add_option('--ICN_order', default=7, type='int', help='ICN iteration number')
    (opts,args) = parser.parse_args()

    nbodies = opts.n
    ICN_it = opts.ICN_order
    np.random.seed(opts.seed)    
    
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
    
    m[0], m[1] = 1.*Ms, 1.*Ms
    
    x[0], x[1] = -0.1*au, 0.1*au
    y[0], y[1] = 0.*au, 0.*au
    z[0], z[1] = 0.*au, 0.*au

    vx[0], vx[1] = 2.e3, -2.e3
    vy[0], vy[1] = -1.e4, 1.e4
    vz[0], vz[1] = 0., 0.
    
    sx[0], sx[1] = 0., 0.
    sy[0], sy[1] = 0., 0.
    sz[0], sz[1] = 0., 0.
    
    #print(x,y,z,vx,vy,vz,sx,sy,sz)
    
    '''
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
    
    plot_step = 50000
    
    dt = opts.dt
    N  = opts.steps
    Neff = int((N//10)//plot_step)
    s_f_step = 5000000 #solution_file_step
    
    nout = int(N/s_f_step)
    
    
    if not opts.p:
        run(N, np.longdouble(dt), opts.PN_order, m, x, y, z, m*vx, m*vy, m*vz, sx, sy, sz, ICN_it)
    
    s, H, T, V = [], [], [], []
    
    for i in range(nout):  
        s_tot, H_tot, T_tot, V_tot = [], [], [], []
        
        s_tot.append(pickle.load(open('solution_{}.pkl'.format(i),'rb')))
        H_tot.append(pickle.load(open('hamiltonian_{}.pkl'.format(i),'rb')))
        T_tot.append(pickle.load(open('kinetic_{}.pkl'.format(i),'rb')))
        V_tot.append(pickle.load(open('potential_{}.pkl'.format(i),'rb')))       
        
        #print(np.shape(s_tot), np.shape(H_tot), np.shape(T_tot), np.shape(V_tot))
        
        s.append(s_tot[0][::plot_step][:])
        H.append(H_tot[0][::plot_step])
        T.append(T_tot[0][::plot_step])
        V.append(V_tot[0][::plot_step])
       
        #print(np.shape(s), np.shape(H), np.shape(T), np.shape(V))
        
        del s_tot
        del H_tot
        del T_tot
        del V_tot
        
        if i%10 == 0 :
            print("Data deframmentation: {}%".format((100*i)/nout))
    
    s = np.array(s, dtype=object)#.flatten()
    H = np.array(H, dtype=object)#.flatten()
    T = np.array(T, dtype=object)#.flatten()
    V = np.array(V, dtype=object)#.flatten()

    s = np.concatenate((s[:]))
    H = np.concatenate((H[:]))
    T = np.concatenate((T[:]))
    V = np.concatenate((V[:])) 
       
    
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
        ax = f.add_subplot(111)
        ax.plot(N_arr, H)
        ax.set_xlabel('iteration')
        ax.set_ylabel('Hamiltonian')
        ax.grid()
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
         
        for q in qs:
            q = np.array(q)/au
            c = next(colors)
            ax.plot(q[:,0],q[:,1],q[:,2],color=c,lw=0.5)
            ax.plot(q[:,0],q[:,1],q[:,2],color='w',alpha=0.5,lw=2,zorder=0)
                

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
        
        q_rel = np.array([[0 for i in range(0, 3)] for Neff in range(0, Neff)])
        p_rel = np.array([[0 for i in range(0, 3)] for Neff in range(0, Neff)])
        
        #print("p1 = {} \np2 = {} \nq1 = {} \nq2 = {}".format(s[i][0]['p'], s[i][1]['p'], s[i][0]['q'], s[i][1]['q']))
        
        L1 = np.zeros(Neff)          
        L2 = np.zeros(Neff)
        L = np.zeros(Neff)
         
        #print(type(L), L)
                  
        for i in range(0, Neff):
            q_rel[i,:], p_rel[i,:] = CM_system(s[i][0]['p'], s[i][1]['p'], s[i][0]['q'], s[i][1]['q'])
            L1[i] = s[i][0]['q'][1]*s[i][0]['p'][2] - s[i][0]['q'][2]*s[i][0]['p'][1] - s[i][0]['q'][0]*s[i][0]['p'][2] + s[i][0]['q'][2]*s[i][0]['p'][0] + s[i][0]['q'][0]*s[i][0]['p'][1] - s[i][0]['q'][1]*s[i][0]['p'][0]
            
            L2[i] = s[i][1]['q'][1]*s[i][1]['p'][2] - s[i][1]['q'][2]*s[i][1]['p'][1] - s[i][1]['q'][0]*s[i][1]['p'][2] + s[i][1]['q'][2]*s[i][1]['p'][0] + s[i][1]['q'][0]*s[i][1]['p'][1] - s[i][1]['q'][1]*s[i][1]['p'][0]
            
            L[i] = L1[i] + L2[i]

        
        f = plt.figure(figsize=(6,4))
        ax = f.add_subplot(111, projection = '3d')
        colors = cm.rainbow(np.linspace(0, 1, nbodies[0]))    
        ax.plot(q_rel[:,0], q_rel[:,1], q_rel[:,2], alpha=0.9)
        ax.plot(q_rel[0,0], q_rel[0,1], q_rel[0,2], 'o-', alpha=0.9)       
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()

        f = plt.figure(figsize=(6,4))
        ax = f.add_subplot(111)
        ax.plot(N_arr, L)
        ax.set_xlabel('iteration')
        ax.set_ylabel('Angolar Momentum')
        ax.grid()
        plt.show()
