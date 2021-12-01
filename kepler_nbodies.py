import numpy as np

from optparse import OptionParser

from numba import jit
from tqdm import tqdm

from datetime import datetime
from astropy.time import Time
from astropy.coordinates import get_body_barycentric_posvel
from astropy.constants import M_earth, M_sun, au, M_jup

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits import mplot3d
from matplotlib import rcParams
from distutils.spawn import find_executable

if find_executable('latex'):
    rcParams["text.usetex"] = True
rcParams["xtick.labelsize"]=14
rcParams["ytick.labelsize"]=14
rcParams["xtick.direction"]="in"
rcParams["ytick.direction"]="in"
rcParams["legend.fontsize"]=15
rcParams["axes.labelsize"]=16
rcParams["axes.grid"] = True
rcParams["grid.alpha"] = 0.6

G = 6.67e-11
Msun = 2e30
AU = 1.5e11
day = 86400

@jit
def angular_momentum(q, p):
    L = np.zeros(3)
    for i in range(len(q)//3):
        qi = q[3*i:3*(i+1)]
        pi = p[3*i:3*(i+1)]
        L += np.array([(qi[1]*pi[2] - qi[2]*pi[1]), -(qi[0]*pi[2] - qi[2]*pi[0]), (qi[0]*pi[1] - qi[1]*pi[0])])
    return np.sqrt(np.sum(L**2))
    

@jit
def hamiltonian(q, p, m):
    T = 0.
    V = 0.
    for i in range(len(m)):
        mi = m[i]
        qi = q[3*i:3*(i+1)]
        pi = p[3*i:3*(i+1)]
        T += np.sum(pi**2)/(2*mi)
        for j in range(i+1,len(m)):
            mj = m[j]
            qj = q[3*j:3*(j+1)]
            dr = qi - qj
            r  = np.sqrt(np.sum(dr**2))
            V  += -G*mi*mj/r
    return T + V, V, T

@jit
def gradient(q, p, m):
    g_q = np.zeros(len(q))
    g_p = np.zeros(len(p))
    for i in range(len(m)):
        mi = m[i]
        qi = q[3*i:3*(i+1)]
        pi = p[3*i:3*(i+1)]
        
        g_p[3*i:3*(i+1)] = pi/mi
        for j in range(i+1, len(m)):
            mj = m[j]
            qj = q[3*j:3*(j+1)]
            dr = qi - qj
            r  = np.sqrt(np.sum(dr**2))
            K  = G*mi*mj/(r*r*r)
            g_q[3*i:3*(i+1)] += K*dr
            g_q[3*j:3*(j+1)] -= K*dr
        
    return g_q, g_p

@jit
def one_step(q, p, dt, m, order):

    dt2 = dt/2.
    mid_q = q
    mid_p = p
    
    for _ in range(order):
        g_q, g_p = gradient(mid_q, mid_p, m)
        
        new_q = q + g_p*dt2
        new_p = p - g_q*dt2
    
        mid_q = (q + new_q)/2.
        mid_p = (p + new_p)/2.

    return new_q, new_p

def run(nsteps, dt, q0, p0, m, order):
    
    q = q0
    p = p0
    
    solution = np.empty(nsteps, dtype = np.ndarray)
    H        = np.empty(nsteps, dtype = np.ndarray)
    V        = np.empty(nsteps, dtype = np.ndarray)
    T        = np.empty(nsteps, dtype = np.ndarray)
    L        = np.empty(nsteps, dtype = np.ndarray)
    
    solution[0]      = q
    H[0], V[0], T[0] = hamiltonian(q, p, m)
    L[0]             = angular_momentum(q, p)
    
    for i in tqdm(range(1,nsteps)):
        q, p             = one_step(q, p, dt, m, order)
        solution[i]      = q
        L[i]             = angular_momentum(q, p)
        H[i], V[i], T[i] = hamiltonian(q, p, m)
    
    return solution, H, V, T, L

@jit
def distance(v1, v2):
    d = np.zeros(len(v1))
    for i, (a,b) in enumerate(zip(v1, v2)):
        d[i] = np.sqrt(np.sum((a-b)**2))
    return d

def plot_solutions(solutions):
    colors = iter(cm.rainbow(np.linspace(0, 1, len(solutions))))
    f = plt.figure(figsize=(6,4))
    ax = f.add_subplot(111, projection = '3d')

    for q in solutions:
        q = np.array(q)
        c = next(colors)
        ax.plot(q[:,0]/AU, q[:,1]/AU, q[:,2]/AU, color=c, lw=0.5)

    f.savefig('./n_bodies.pdf', bbox_inches = 'tight')
    
def plot_hamiltonian(t, H, V, T):

    fig, (ax, e) = plt.subplots(2,1, sharex = True)
    fig.subplots_adjust(hspace=.0)
    
    ax.plot(t, H, lw = 0.5, label = '$H$')
    e.plot(t, T - np.mean(T), lw = 0.5, color = 'g', label = '$T$')
    e.plot(t, V - np.mean(V), lw = 0.5, color = 'r', label = '$V$')
    ax.plot(t, np.ones(len(H))*H[0], lw = 0.5, ls = '--', color = 'k', label = '$H(0)$')
    
    e.set_ylabel('$E(t)$')
    e.set_xlabel('$t\ [yr]$')
    ax.set_ylabel('$H(t)$')
    
    ax.grid(True,dashes=(1,3))
    ax.legend(loc=0,frameon=False,fontsize=10)
    e.grid(True,dashes=(1,3))
    e.legend(loc=0,frameon=False,fontsize=10)
    
    fig.savefig('./kepler_hamiltonian_nbodies.pdf', bbox_inches = 'tight')

def plot_angular_momentum(t, L):
    
    fig, ax = plt.subplots()
    
    ax.plot(t, L, lw = 0.5)
    
    ax.set_ylabel('$L(t)$')
    ax.set_xlabel('$t\ [yr]$')
    ax.grid(True,dashes=(1,3))
    
    fig.savefig('./angular_momentum_nbodies.pdf', bbox_inches = 'tight')
if __name__ == '__main__':
    
    parser = OptionParser()
    parser.add_option('--years', default = 1, type = 'int', help = "Number of years")
    parser.add_option('--cm', default = False, action = 'store_true', help = "Set center of mass velocity to 0")
    parser.add_option('--cn_order', default = 7, type = 'int', help = "Crank-Nicolson integrator order")
    parser.add_option('--dt', default = 1, type = 'int', help = "Number of seconds for each dt")
    parser.add_option('-p', dest = "postprocessing", default = False, action = 'store_true', help = "Postprocessing")

    (opts,args) = parser.parse_args()
    
    t = Time(datetime.now())#'2021-06-21T00:00:00')

    m = np.array([1*Msun, (M_earth/M_sun).value*Msun, (M_jup/M_sun).value*Msun])
    planet_names = ['sun', 'earth', 'jupiter']
    
    planets = np.array([get_body_barycentric_posvel(planet, t) for planet in planet_names])
    
    # Initial conditions
    q0 = np.concatenate([np.array([float(planet[0].x.value*AU), float(planet[0].y.value*AU), float(planet[0].z.value*AU)]) for planet in planets])
    v0 = np.concatenate([np.array([float(planet[1].x.value*AU/day), float(planet[1].y.value*AU/day), float(planet[1].z.value*AU/day)]) for planet in planets])
    
    if opts.cm:
        v_cm = np.sum([v0[3*i:3*(i+1)]*m[i] for i in range(len(m))])/np.sum(m)
        for i in range(len(m)):
            v0[3*i:3*(i+1)] -= v_cm
        
    p0 = np.concatenate([v0[3*i:3*(i+1)]*m[i] for i in range(len(m))])

    # Integrator settings
    n_years = int(opts.years)
    nsteps = int(365*2*n_years*day/int(opts.dt))
    dt = opts.dt
    
    order = int(opts.cn_order)
    
    if not opts.postprocessing:
        s, H, V, T, L = run(nsteps, dt, q0, p0, m, order)

        x = np.array([[si[3*i:3*(i+1)] for si in s] for i in range(len(m))])

        t = np.arange(x.shape[1])*dt
        
        #np.savetxt('./orbit_nbodies.txt', np.array([t, x1[:,0], x1[:,1], x1[:,2], x2[:,0], x2[:,1], x2[:,2], H, V, T, L]).T, header = 't x1x x1y x1z x2x x2y x2z H V T, L')
    
    else:
        sol = np.genfromtxt('./orbit_nbodies.txt', names = True)
        
        t  = sol['t']
        
        x1 = np.array([sol['x1x'], sol['x1y'], sol['x1z']]).T
        x2 = np.array([sol['x2x'], sol['x2y'], sol['x2z']]).T
        
        H  = sol['H']
        V  = sol['V']
        T  = sol['T']
        L  = sol['L']
    
    
    plot_solutions(x)
    plot_hamiltonian(t/(2*365*day), H, V, T)
    plot_angular_momentum(t/(2*365*day), L)
    
    
