import numpy as np

from optparse import OptionParser

from numba import jit
from tqdm import tqdm

from datetime import datetime
from astropy.time import Time
from astropy.coordinates import get_body_barycentric_posvel
from astropy.constants import M_earth, M_sun, au

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
day = 86400.

@jit
def hamiltonian(q, p, m):
    dx = q[0] - q[3]
    dy = q[1] - q[4]
    dz = q[2] - q[5]
    r  = np.sqrt(dx**2 + dy**2 + dz**2)
    return np.sum(p[:3]**2/(2*m[0])) + np.sum(p[3:]**2/(2*m[1])) - G*np.prod(m)/r

@jit
def gradient(q, p, m):
    dx = q[0] - q[3]
    dy = q[1] - q[4]
    dz = q[2] - q[5]
    r3 = np.sqrt(dx**2 + dy**2 + dz**2)**3
    M = np.prod(m)
    K = G*M/r3
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
    
    solution = np.empty(nsteps+1, dtype = np.ndarray)
    H        = np.empty(nsteps+1, dtype = np.ndarray)
    
    solution[0] = q
    H[0]        = hamiltonian(q, p, m)
    
    for i in tqdm(range(0,nsteps)):
        q, p = one_step(q, p, dt, m, order)
        solution[i+1] = q
        H[i+1]        = hamiltonian(q, p, m)
    
    return solution, np.array(H)

#def exact_solution(t, q0, p0):
#    omega = np.sqrt(k/(2*m))
#    x2 = (q0[1] - q0[0] - l)*np.cos(omega*t)/2. + (p0[1] - p0[0])*np.sin(omega*t)/(4*m*omega) + l/2.
#    return -x2, x2

def plot_solutions(x1, x2):

    colors = iter(cm.rainbow(np.linspace(0, 1, 2)))
    f = plt.figure(figsize=(6,4))
    ax = f.add_subplot(111, projection = '3d')

    for q in [x1,x2]:
        q = np.array(q)
        c = next(colors)
        ax.plot(q[:,0]/AU, q[:,1]/AU, q[:,2]/AU, color=c, lw=0.5)

    f.savefig('./two_bodies.pdf', bbox_inches = 'tight')
    
def plot_hamiltonian(t, H):

    fig, ax = plt.subplots()
    
    ax.plot(t, H, lw = 0.5)
    ax.plot(t, np.ones(len(H))*H[0], lw = 0.1, ls = '--', label = '$H(0)$')
    
    ax.set_xlabel('$t\ [yr]$')
    ax.set_ylabel('$H(t)$')
    
    ax.grid(True,dashes=(1,3))
    ax.legend(loc=0,frameon=False,fontsize=10)
    
    fig.savefig('./kepler_hamiltonian.pdf', bbox_inches = 'tight')

if __name__ == '__main__':
    
    parser = OptionParser()
    parser.add_option('--nyears', default = 5, type = 'int', help = "Number of years")
    parser.add_option('--cm', default = False, action = 'store_true', help = "Set center of mass velocity to 0")
    parser.add_option('--cn_order', default = 3, type = 'int', help = "Crank-Nicolson integrator order")

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
        v0[0] -= v_cm[0]
        v0[1] -= v_cm[1]
        v0[2] -= v_cm[2]
        v0[3] -= v_cm[0]
        v0[4] -= v_cm[1]
        v0[5] -= v_cm[2]
    
    p0 = np.concatenate((v0[:3]*m[0], v0[3:]*m[1]))
    
    # Integrator settings
    n_years = opts.nyears
    nsteps = 365*2*n_years
    dt = day
    
    order = int(opts.cn_order)
    
    s, H = run(nsteps, dt, q0, p0, m, order)

    x1 = np.array([si[:3] for si in s])
    x2 = np.array([si[3:] for si in s])
    
    t = np.arange(len(s))*dt

    plot_solutions(x1, x2)
    plot_hamiltonian(t/(2*365*day), H)
