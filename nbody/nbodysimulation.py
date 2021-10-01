import numpy as np
#from nbody.body import body
#from nbody.hamiltonian import hamiltonian, gradients, kinetic_energy, potential
from nbody.engine import run
from collections import deque
from optparse import OptionParser
import pickle

if __name__=="__main__":
    parser = OptionParser()
    parser.add_option('-n', default=9, type='int', help='n bodies')
    parser.add_option('--steps', default=1000, type='int', help='n steps')
    parser.add_option('--order', default=0, type='int', help='PN order')
    parser.add_option('--dt', default=1, type='float', help='dt')
    parser.add_option('-p', default=0, type='int', help='post process')
    parser.add_option('--animate', default=0, type='int', help='animate')
    parser.add_option('--plot', default=1, type='int', help='plot')
    parser.add_option('--seed', default=1, type='int', help='seed')
    (opts,args)=parser.parse_args()

    nbodies = opts.n
    np.random.seed(opts.seed)
#    m = np.random.uniform(1.0,10.0,size = nbodies)
#    x = np.random.uniform(-200.0,200.0,size = nbodies)
#    y = np.random.uniform(-200.0,200.0,size = nbodies)
#    z = np.random.uniform(-200.0,200.0,size = nbodies)
    vx = np.random.uniform(0.0,0.1,size = nbodies).astype(np.longdouble)
    vy = np.random.uniform(-0.01,0.01,size = nbodies).astype(np.longdouble)
#    vz = np.random.uniform(-0.01,0.01,size = nbodies)
    m = np.random.uniform(1.0,10.0,size = nbodies).astype(np.longdouble)
    x = np.random.uniform(-400.0,400.0,size = nbodies).astype(np.longdouble)
    y = np.random.uniform(-400.0,400.0,size = nbodies).astype(np.longdouble)
    z = np.random.uniform(-400.0,400.0,size = nbodies).astype(np.longdouble)
#    vx = np.array((0.0,0.0))
#    vy = np.array((0.1,-0.1))
    vz = np.random.uniform(-0.01,0.01,size = nbodies).astype(np.longdouble)
    
    sx = np.random.uniform(-1.0,1.0,size = nbodies).astype(np.longdouble)
    sy = np.random.uniform(-1.0,1.0,size = nbodies).astype(np.longdouble)
    sz = np.random.uniform(-1.0,1.0,size = nbodies).astype(np.longdouble)
    
    dt = opts.dt
    N  = opts.steps
    
    if opts.p == 1:
        s,H = run(N, np.longdouble(dt), opts.order, m, x, y, z,
            m*vx, m*vy, m*vz, sx, sy, sz)
        s   = np.array(s)
        pickle.dump(s,open('solution.p','wb'))
        pickle.dump(H,open('hamiltonian.p','wb'))
    else:
        s = pickle.load(open('solution.p','rb'))
        H = pickle.load(open('hamiltonian.p','rb'))

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
        
        anim = FuncAnimation(f, animate, N, fargs=(s, lines, symbols),
                             interval = 0.1, blit = True)
        Writer = writers['ffmpeg']
        writer = Writer(fps=120, metadata=dict(artist='Me'), bitrate=900, extra_args=['-vcodec', 'libx264'])
        anim.save('nbody.mp4', writer=writer)
        
        
    if opts.plot==1:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from mpl_toolkits import mplot3d
        
        plotting_step = np.maximum(64,N//int(0.1*N))
        
        f = plt.figure(figsize=(6,4))
        ax = f.add_subplot(111, projection = '3d')
        colors = cm.rainbow(np.linspace(0, 1, nbodies))
        for b in range(nbodies):
            q = np.array([s[i][b]['q'] for i in range(0,N,plotting_step)])
            ax.plot(q[:,0],q[:,1],q[:,2],color=colors[b],alpha=0.5)
        
        f = plt.figure(figsize=(6,4))
        ax = f.add_subplot(111)
        ax.plot(range(N),H)
        ax.set_xlabel('iteration')
        ax.set_ylabel('Hamiltonian')

        plt.show()
        
        if 0:
            f = plt.figure(figsize=(6,4))
            ax = f.add_subplot(111, projection = '3d')
            colors = cm.rainbow(np.linspace(0, 1, nbodies))
      
            trails = {}
            for b in range(nbodies):
                trails[b] = deque(maxlen=500)

            for i in range(0,N,plotting_step):
                plt.cla()
                for b in range(nbodies):
                    q = s[i][b]['q']
                    trails[b].append(q)
                    q_trail = np.array(trails[b])
                    ax.scatter(q[0],q[1],q[2],color=colors[b],s=10*s[i][b]['mass'])
                    ax.plot(q_trail[:,0],q_trail[:,1],q_trail[:,2],color=colors[b],alpha=0.5)
    #            ax.set(xlim=(-50, 50), ylim=(-50, 50), zlim=(-50,50))
                plt.pause(0.00001)
            plt.show()
