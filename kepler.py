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
    q1 = q[:3]
    q2 = q[3:]
    p1 = p[:3]
    p2 = p[3:]
    
    L1 = np.array([(q1[1]*p1[2] - q1[2]*p1[1]), -(q1[0]*p1[2] - q1[2]*p1[0]), (q1[0]*p1[1] - q1[1]*p1[0])])
    L2 = np.array([(q2[1]*p2[2] - q2[2]*p2[1]), -(q2[0]*p2[2] - q2[2]*p2[0]), (q2[0]*p2[1] - q2[1]*p2[0])])
    
    L = L1 + L2
    
    return np.sqrt(np.sum(L**2))
    

@jit
def hamiltonian(q, p, m):
    dx = q[0] - q[3]
    dy = q[1] - q[4]
    dz = q[2] - q[5]
    r  = np.sqrt(dx**2 + dy**2 + dz**2)
    T = np.sum(p[:3]**2/(2*m[0])) + np.sum(p[3:]**2/(2*m[1]))
    V = - G*np.prod(m)/r
    return T + V, V, T

@jit
def gradient(q, p, m):
    dx = q[0] - q[3]
    dy = q[1] - q[4]
    dz = q[2] - q[5]
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    M = np.prod(m)
    K = G*M/(r*r*r)
    return np.array([K*dx, K*dy, K*dz, -K*dx, -K*dy, -K*dz]), np.array([p[0]/m[0], p[1]/m[0], p[2]/m[0], p[3]/m[1], p[4]/m[1], p[5]/m[1]])

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

def distance(v1, v2):
    d = np.zeros(len(v1))
    for i, (a,b) in enumerate(zip(v1, v2)):
        d[i] = np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)
    return d

def plot_solutions(x1, x2):

    colors = iter(cm.rainbow(np.linspace(0, 1, 2)))
    f = plt.figure(figsize=(6,4))
    ax = f.add_subplot(111, projection = '3d')

    for q in [x1,x2]:
        q = np.array(q)
        c = next(colors)
        ax.plot(q[:,0]/AU, q[:,1]/AU, q[:,2]/AU, color=c, lw=0.5)

    f.savefig('./two_bodies.pdf', bbox_inches = 'tight')
    
def plot_hamiltonian(t, H, V, T, dist):

    fig, (ax, e, d) = plt.subplots(3,1, sharex = True, gridspec_kw={'height_ratios': [2, 2, 1]})
    fig.subplots_adjust(hspace=.0)
    
    ax.plot(t, H, lw = 0.1, color = 'b', label = '$H$')
    e.plot(t, T - np.mean(T), lw = 0.1, color = 'g', label = '$T$')
    e.plot(t, V - np.mean(V), lw = 0.1, color = 'r', label = '$V$')
    ax.plot(t, np.ones(len(H))*H[0], lw = 0.5, ls = '--', color = 'k', label = '$H(0)$')
    d.plot(t, dist/AU, lw = 0.5, color = 'g')
    
    e.set_ylabel('$E(t)$')
    d.set_xlabel('$t\ [yr]$')
    d.set_ylabel('$d(t)$')
    ax.set_ylabel('$H(t)$')
    
    ax.grid(True,dashes=(1,3))
    ax.legend(loc=0,frameon=False,fontsize=10)
    e.grid(True,dashes=(1,3))
    e.legend(loc=0,frameon=False,fontsize=10)
    d.grid(True,dashes=(1,3))
    
    fig.savefig('./kepler_hamiltonian.pdf', bbox_inches = 'tight')

def plot_angular_momentum(t, L):
    
    fig, ax = plt.subplots()
    
    ax.plot(t, L, lw = 0.5)
    
    ax.set_ylabel('$L(t)$')
    ax.set_xlabel('$t\ [yr]$')
    ax.grid(True,dashes=(1,3))
    
    fig.savefig('./angular_momentum.pdf', bbox_inches = 'tight')
if __name__ == '__main__':
    
    parser = OptionParser()
    parser.add_option('--years', default = 1, type = 'int', help = "Number of years")
    parser.add_option('--cm', default = False, action = 'store_true', help = "Set center of mass velocity to 0")
    parser.add_option('--cn_order', default = 7, type = 'int', help = "Crank-Nicolson integrator order")
    parser.add_option('--dt', default = 1, type = 'int', help = "Number of seconds for each dt")
    parser.add_option('-p', dest = "postprocessing", default = False, action = 'store_true', help = "Postprocessing")

    (opts,args) = parser.parse_args()
    
    t = Time(datetime.now())#'2021-06-21T00:00:00')

    m = np.array([1*Msun, (M_earth/M_sun).value*Msun])
    earth = get_body_barycentric_posvel('earth', t)
    sun   = get_body_barycentric_posvel('sun', t)

    # Initial conditions
    q0 = np.array([float(sun[0].x.value*AU), float(sun[0].y.value*AU), float(sun[0].z.value*AU), float(earth[0].x.value*AU), float(earth[0].y.value*AU), float(earth[0].z.value*AU)])
    v0 = np.array([float(sun[1].x.value*AU/day), float(sun[1].y.value*AU/day), float(sun[1].z.value*AU/day), float(earth[1].x.value*AU/day), float(earth[1].y.value*AU/day), float(earth[1].z.value*AU/day)])
    
    if opts.cm:
        v_cm = (v0[:3]*m[0] + v0[3:]*m[1])/np.sum(m)
        v0[:3] -= v_cm
        v0[3:] -= v_cm
        
    p0 = np.concatenate((v0[:3]*m[0], v0[3:]*m[1]))
    
    # Integrator settings
    n_years = int(opts.years)
    nsteps = int(365*2*n_years*day/int(opts.dt))
    dt = opts.dt
    
    order = int(opts.cn_order)
    
    if not opts.postprocessing:
        s, H, V, T, L = run(nsteps, dt, q0, p0, m, order)

        x1 = np.array([si[:3] for si in s])
        x2 = np.array([si[3:] for si in s])

        t = np.arange(len(x1))*dt
        
        np.savetxt('./orbit.txt', np.array([t, x1[:,0], x1[:,1], x1[:,2], x2[:,0], x2[:,1], x2[:,2], H, V, T, L]).T, header = 't x1x x1y x1z x2x x2y x2z H V T, L')
    
    else:
        sol = np.genfromtxt('./orbit.txt', names = True)
        
        t  = sol['t']
        
        x1 = np.array([sol['x1x'], sol['x1y'], sol['x1z']]).T
        x2 = np.array([sol['x2x'], sol['x2y'], sol['x2z']]).T
        
        H  = sol['H']
        V  = sol['V']
        T  = sol['T']
        L  = sol['L']
        
        
    
    d = distance(x1, x2)
    
    plot_solutions(x1, x2)
    plot_hamiltonian(t/(2*365*day), H, V, T, d)
    plot_angular_momentum(t/(2*365*day), L)
    
    
