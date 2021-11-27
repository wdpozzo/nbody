import numpy as np
#from nbody.body import body
#from nbody.hamiltonian import hamiltonian, gradients, kinetic_energy, potential
from nbody.engine import run
from collections import deque
from optparse import OptionParser
from nbody.CM_coord_system import CM_system
import pickle

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits import mplot3d

from datetime import datetime
from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, get_body_barycentric_posvel
from astropy.constants import M_earth, M_sun, au

G = 6.67e-11
c = 3.0e8
Msun = 2e30
AU = 1.5e11 #astronomical unit
GM = 1.32712440018e20 # G*Msun
day = 86400 # s

def make_plots(H, Neff, nbodies, s):
    plotting_step = 64#np.maximum(64,Neff//int(0.1*Neff))
    
    f = plt.figure(figsize=(6,4))
    ax = f.add_subplot(111)
    ax.plot(range(Neff), H)
    ax.set_xlabel('iteration')
    ax.set_ylabel('Hamiltonian')
    ax.grid()
    plt.savefig('./hamiltonian.pdf')
    colors = iter(cm.rainbow(np.linspace(0, 1, nbodies)))
    
    qs = [[] for x in range(nbodies)]

    # this is the number of bodies active in each step of the solution
    nbodies = [len(si) for si in s]
    f = plt.figure(figsize=(6,4))
    ax = f.add_subplot(111)
    ax.plot(range(Neff), nbodies)
    ax.set_xlabel('iteration')
    ax.set_ylabel('nbodies')
    plt.savefig('./nbodies.pdf', bbox_inches = 'tight')
    
    f = plt.figure(figsize=(6,4))
    ax = f.add_subplot(111, projection = '3d')
    
    for i in range(0,Neff,plotting_step):
        for j in range(nbodies[i]):
            qs[j].append(s[i][j]['q'])

    for q in qs:
        q = np.array(q)
        c = next(colors)
        ax.plot(q[:,0]/AU,q[:,1]/AU,q[:,2]/AU,color=c,lw=0.5)
#        ax.plot(q[:,0]/AU,q[:,1]/AU,q[:,2]/AU,color='w',alpha=0.5,lw=2,zorder=0)

    plt.savefig('./earth_sun.pdf', bbox_inches = 'tight')
    plt.show()


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option('--steps', default = 1000, type='int', help = "Number of steps to compute")
    parser.add_option('--order', default = 0, type='int', help = "Post Newtonian order")
    parser.add_option('--dt', default = 1, type='float', help = "Time interval (dt)")
    parser.add_option('-p', default = False, action = 'store_true', help = "Run postprocessing only")
    parser.add_option('--plot', default = True, action = 'store_true', help = "Make plots")
    parser.add_option('--cm', default = False, action = 'store_true', help = "Plot separation (requires nbodies = 2)")
    (opts,args) = parser.parse_args()
    
    
    t = Time('2021-06-21T00:00:00')#datetime.now())

    m = np.array([1.5*Msun, (M_earth/M_sun).value*Msun]).astype(np.longdouble)
    
    
    earth = get_body_barycentric_posvel('earth', t)
    sun   = get_body_barycentric_posvel('sun', t)
    print(earth)
    x = np.array([sun[0].x.value*AU, earth[0].x.value*AU]).astype(np.longdouble)
    y = np.array([sun[0].y.value*AU, earth[0].y.value*AU]).astype(np.longdouble)
    z = np.array([sun[0].z.value*AU, earth[0].z.value*AU]).astype(np.longdouble)

    vx = np.array([sun[1].x.value*AU/day, earth[1].x.value*AU/day]).astype(np.longdouble)
    vy = np.array([sun[1].y.value*AU/day, earth[1].y.value*AU/day]).astype(np.longdouble)
    vz = np.array([sun[1].z.value*AU/day, earth[1].z.value*AU/day]).astype(np.longdouble)
    
    sx = np.zeros(2).astype(np.longdouble)
    sy = np.zeros(2).astype(np.longdouble)
    sz = np.zeros(2).astype(np.longdouble)
    
    v0 = np.sqrt(vx[0]**2+vy[0]**2+vz[0]**2)
    v1 = np.sqrt(vx[1]**2+vy[1]**2+vz[1]**2)
    d  = np.sqrt((x[1]-x[0])**2+(y[1]-y[0])**2+(z[1]-z[0])**2)
    
    E = 0.5*(v0**2*m[0] + v1**2*m[1])- G*m[1]*m[0]/d
    print(E)

    dt = opts.dt
    N  = opts.steps
    Neff = N//10
    if not opts.p:
        s,H = run(N, np.longdouble(dt), opts.order, m, x, y, z, m*vx, m*vy, m*vz, sx, sy, sz)
        s   = np.array(s, dtype=object)
        pickle.dump(s, open('solution.pkl','wb'))
        pickle.dump(H, open('hamiltonian.pkl','wb'))
        
    else:
        s = pickle.load(open('solution.pkl','rb'))
        H = pickle.load(open('hamiltonian.pkl','rb'))

    make_plots(H, Neff, 2, s)
