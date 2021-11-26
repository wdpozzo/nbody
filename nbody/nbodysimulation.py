import numpy as np
#from nbody.body import body
#from nbody.hamiltonian import hamiltonian, gradients, kinetic_energy, potential
from nbody.engine import run
from collections import deque
from optparse import OptionParser
from nbody.CM_coord_system import CM_system
import pickle

G = 1  #6.67e-11
C = 1 #3.0e8 
Msun = 2e30
GM = 1.32712440018e20

if __name__=="__main__":
    parser = OptionParser()
    parser.add_option('-n', default=9, type='int', help='n bodies')
    parser.add_option('--steps', default=1000, type='int', help='n steps')
    parser.add_option('--order', default=0, type='int', help='PN order')
    parser.add_option('--dt', default=1, type='float', help='dt')
    parser.add_option('-p', default = False, action = 'store_true', help='post process')
    parser.add_option('--animate', default=0, type='int', help='animate')
    parser.add_option('--plot', default=1, type='int', help='plot')
    parser.add_option('--cm', default=1, type='int', help='plot')
    parser.add_option('--seed', default = False, action = 'store_true', help='seed')
    (opts,args) = parser.parse_args()

    nbodies = opts.n
    if opts.seed:
        np.random.seed(1)
    
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
    m[0], m[1] = 1e-1, 1e-3 #Msun*(10e6), 10*Msun
    
    x[0], x[1] = 50., -50.
    y[0], y[1] = 50., 50.
    z[0], z[1] = 0., 0.

    vx[0], vx[1] = 1e-4, 1e-3
    vy[0], vy[1] = -1e-4, 1e-3
    vz[0], vz[1] = 0., 0.
    
    sx[0], sx[1] = 0., 0.
    sy[0], sy[1] = 0., 0.
    sz[0], sz[1] = 0., 0.
    
    print(x,y,z,vx,vy,vz,sx,sy,sz)
    '''

    m = np.random.uniform(1e0, 1e1,size = nbodies).astype(np.longdouble)

    x = np.random.uniform(- 1000.0, 1000.0,size = nbodies).astype(np.longdouble)
    y = np.random.uniform(- 1000.0, 1000.0,size = nbodies).astype(np.longdouble)
    z = np.random.uniform(- 1000.0, 1000.0,size = nbodies).astype(np.longdouble)

    vx = np.random.uniform(-0.01, 0.01,size = nbodies).astype(np.longdouble)
    vy = np.random.uniform(-0.01, 0.01,size = nbodies).astype(np.longdouble)
    vz = np.random.uniform(-0.01, 0.01,size = nbodies).astype(np.longdouble)
    
    sx = np.random.uniform(-1.0,1.0,size = nbodies).astype(np.longdouble)
    sy = np.random.uniform(-1.0,1.0,size = nbodies).astype(np.longdouble)
    sz = np.random.uniform(-1.0,1.0,size = nbodies).astype(np.longdouble)

    #print(m,x,y,z,vx,vy,vz,sx,sy,sz)

    dt = opts.dt
    N  = opts.steps
    Neff = N//10
    
    if not opts.p:
        s,H = run(N, np.longdouble(dt), opts.order, m, x, y, z, m*vx, m*vy, m*vz, sx, sy, sz)
        s   = np.array(s, dtype=object)
        pickle.dump(s, open('solution.p','wb'))
        pickle.dump(H, open('hamiltonian.p','wb'))
        
    else:
        s = pickle.load(open('solution.p','rb'))
        H = pickle.load(open('hamiltonian.p','rb'))
    
    #print("p1 = {} \np2 = {} \nq1 = {} \nq2 = {}".format(s[1][0]['p'], s[1][1]['p'], s[1][0]['q'], s[1][1]['q']))
        
    
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
        
        plotting_step = np.maximum(64,Neff//int(0.1*Neff))
        
        f = plt.figure(figsize=(6,4))
        ax = f.add_subplot(111)
        ax.plot(range(Neff), H)
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
        
        for i in range(0,Neff,plotting_step):
            for j in range(nbodies[i]):
                qs[j].append(s[i][j]['q'])

        for q in qs:
            q = np.array(q)
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

            for i in range(0,Neff,plotting_step):
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
            

    if opts.cm == 1 and nbodies == 2:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from mpl_toolkits import mplot3d
        
        k = int(opts.steps/10)
        
        q_rel = np.array([[0 for i in range(0, 3)] for k in range(0, k)])
        p_rel = np.array([[0 for i in range(0, 3)] for k in range(0, k)])

    
        for i in range(k):
            try:
                q_rel[i,:], p_rel[i,:] = CM_system(s[i][0]['p'], s[i][1]['p'], s[i][0]['q'], s[i][1]['q'])
            except:
                pass
        
        f = plt.figure(figsize=(6,4))
        
        ax = f.add_subplot(111, projection = '3d')
        colors = cm.rainbow(np.linspace(0, 1, nbodies[0]))    
        ax.plot(q_rel[:,0], q_rel[:,1], q_rel[:,2], alpha=0.9)
        ax.plot(q_rel[0,0], q_rel[0,1], q_rel[0,2], 'o-', alpha=0.9)       
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_ylabel('z')

        plt.show()
        
