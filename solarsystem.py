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
from astropy.constants import M_earth, M_sun, au, M_jup

import astropy.units as u

G = 6.67e-11  # m^3 kg^-1 s^-2
Msun = 2e30   # kg
AU = 1.5e11   # m
day = 86400   # s
Mearth = 6e24 # kg

# Solar system masses
masses = {
    'sun'     : Msun,
    'earth'   : Mearth,
    'moon'    : 0.0123*Mearth,
    'mercury' : 0.0553*Mearth,
    'mars'    : 0.1075*Mearth,
    'venus'   : 0.815*Mearth,
    'jupiter' : 317.8*Mearth,
    'saturn'  : 95.2*Mearth,
    'uranus'  : 14.6*Mearth,
    'neptune' : 17.2*Mearth,
    'pluto'   : 0.00218*Mearth,
}

def make_plots(H, Neff, nbodies, s):
    plotting_step =1# 64#np.maximum(64,Neff//int(0.1*Neff))
    
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

    plt.savefig('./solarsystem.pdf', bbox_inches = 'tight')
    plt.show()


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option('--steps', default = 1000, type='int', help = "Number of steps to compute")
    parser.add_option('--order', default = 0, type='int', help = "Post Newtonian order")
    parser.add_option('--cn_order', default = 7, type='int', help = "CN integrator order")
    parser.add_option('--dt', default = 1, type='float', help = "Time interval (dt)")
    parser.add_option('-p', default = False, action = 'store_true', help = "Run postprocessing only")
    parser.add_option('--plot', default = True, action = 'store_true', help = "Make plots")
    parser.add_option('--cm', default = False, action = 'store_true', help = "Plot separation (requires nbodies = 2)")
    (opts,args) = parser.parse_args()
    
    
    t = Time(datetime.now())#'2021-06-21T00:00:00')
    
    planet_names = ['sun',
    'mercury',
    'venus',
    'earth',
    'mars',
    'jupiter',
    'saturn',
    'uranus',
    'neptune',
    ]
    
    planets = []
    
    for planet in planet_names:
        planets.append(get_body_barycentric_posvel(planet,t))
        
    print(planets)
    m = np.array([masses[planet] for planet in planet_names]).astype(np.longdouble)

    print(m)
    Mtot = np.sum(m)
   
   
    x = np.array([planet[0].x.value*AU for planet in planets]).astype(np.longdouble)
    y = np.array([planet[0].y.value*AU for planet in planets]).astype(np.longdouble)
    z = np.array([planet[0].z.value*AU for planet in planets]).astype(np.longdouble)
    
    vx = np.array([planet[1].x.value*AU/day for planet in planets]).astype(np.longdouble)
    vy = np.array([planet[1].y.value*AU/day for planet in planets]).astype(np.longdouble)
    vz = np.array([planet[1].z.value*AU/day for planet in planets]).astype(np.longdouble)
    print(vx,vy,vz)
    exit(0)
    vcm = np.array([np.sum(vx*m/Mtot), np.sum(vy*m/Mtot), np.sum(vz*m/Mtot)])
    print(vcm, np.linalg.norm(vcm))
    sx = np.zeros(len(m)).astype(np.longdouble)
    sy = np.zeros(len(m)).astype(np.longdouble)
    sz = np.zeros(len(m)).astype(np.longdouble)

    v0 = np.sqrt(vx[0]**2+vy[0]**2+vz[0]**2)
    v1 = np.sqrt(vx[1]**2+vy[1]**2+vz[1]**2)
    d  = np.sqrt((x[1]-x[0])**2+(y[1]-y[0])**2+(z[1]-z[0])**2)
    print(d/AU)
    E = 0.5*(v0**2*m[0] + v1**2*m[1]) - G*m[1]*m[0]/d
    print(E)

    dt = opts.dt
    N  = opts.steps
    thin = 10
    Neff = N//thin
    n_buf = 100000
    if not opts.p:
        run(N,
            np.longdouble(dt),
            opts.order,
            m,
            x,
            y,
            z,
            m*vx,
            m*vy,
            m*vz,
            sx,
            sy,
            sz,
            opts.cn_order,
            nthin = thin,
            buffer_length = n_buf)
    n_buf = N//n_buf
    for i in range(n_buf):
        if i == 0:
            s = np.array(pickle.load(open('solution_{}.pkl'.format(i),'rb')), dtype=object)
            H = np.array(pickle.load(open('hamiltonian_{}.pkl'.format(i),'rb')))
        else:
            s = np.row_stack((s,np.array(pickle.load(open('solution_{}.pkl'.format(i),'rb')), dtype=object)))
            H = np.concatenate((H,np.array(pickle.load(open('hamiltonian_{}.pkl'.format(i),'rb')), dtype=object)))
    
    print(s.shape,H.shape)
    make_plots(H, Neff, len(m), s)
