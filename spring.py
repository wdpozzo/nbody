import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from tqdm import tqdm

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

@jit
def hamiltonian(q, p, m, k, l):
    return np.sum(p**2/(2*m)) + k*((q[1] - q[0]) - l)**2

@jit
def gradient(q, p, m, k, l):
    return np.array([-2*k*(q[1] - q[0] - l), 2*k*(q[1] - q[0] - l)]), p/m

@jit
def one_step(q, p, dt, m, k, l):

    dt2 = dt/2.
    mid_q = q
    mid_p = p
    
    for _ in range(3):
        g_q, g_p = gradient(mid_q, mid_p, m, k, l)
    
        new_q = q + g_p*dt2
        new_p = p - g_q*dt2
    
        mid_q = (q + new_q)/2.
        mid_p = (p + new_p)/2.
    
    return new_q, new_p

def run(nsteps, dt, q0, p0, m, k, l):
    
    q = q0
    p = p0
    
    solution = np.empty(nsteps+1, dtype = np.ndarray)
    H        = np.empty(nsteps+1, dtype = np.ndarray)
    
    solution[0] = q
    H[0]        = hamiltonian(q, p, m, k, l)
    
    for i in tqdm(range(0,nsteps)):
        q, p = one_step(q, p, dt, m, k, l)
        solution[i+1] = q
        H[i+1]        = hamiltonian(q, p, m, k, l)
    
    return np.array(solution), np.array(H)

def exact_solution(t, q0, p0, m, k, l):
    omega = np.sqrt(k/m)
    x2 = (q0[1] - q0[0] - l)*np.cos(omega*t)/2. + (p0[1] - p0[0])*np.sin(omega*t)/(2*omega) + l/2.
    return -x2, x2

def plot_solutions(t, x1, x2, x1_e, x2_e):
    
    fig, (ax, d) = plt.subplots(2,1, sharex = True, gridspec_kw={'height_ratios': [2, 1]})
    
    ax.plot(t, x1, color = 'r', lw = 0.5, label = '$m_{1,num}$')
    ax.plot(t, x2, color = 'g', lw = 0.5, label = '$m_{2,num}$')
    ax.plot(t, x1_e, color = 'r', lw = 0.5, ls = '--', label = '$m_{1,exact}$')
    ax.plot(t, x2_e, color = 'g', lw = 0.5, ls = '--', label = '$m_{2,exact}$')
    
    d.plot(t, x1-x1_e, color = 'r', ls = '--', lw = 0.5)
    d.plot(t, x2-x2_e, color = 'g', ls = '--', lw = 0.5)
    
    ax.set_ylabel('$x(t)$')
    d.set_ylabel('$x_{num}(t)-x_{exact}(t)$')
    d.set_xlabel('$t/T$')
    
    ax.grid(True,dashes=(1,3))
    ax.legend(loc=0,frameon=False,fontsize=10)
    d.grid(True,dashes=(1,3))
    
    fig.subplots_adjust(hspace=.0)
    
    fig.savefig('./harmonic_oscillator.pdf', bbox_inches = 'tight')
    
def plot_hamiltonian(t, H):

    fig, ax = plt.subplots()
    
    ax.plot(t, H, lw = 0.5)
    ax.plot(t, np.ones(len(H))*H[0], lw = 0.5, ls = '--', label = '$H(0)$')
    
    ax.set_xlabel('$t/T$')
    ax.set_ylabel('$H(t)$')
    
    ax.grid(True,dashes=(1,3))
    ax.legend(loc=0,frameon=False,fontsize=10)
    
    fig.savefig('./harm_osc_hamiltonian.pdf', bbox_inches = 'tight')

if __name__ == '__main__':
    
    k = 65
    m = 0.68
    l = 1

    nsteps = 1000000
    dt = (2*np.pi/(np.sqrt(k/m)))/(nsteps/4)
    T  = 2*np.pi/(np.sqrt(k/m))
    
    thinning = 10
    
    q0 = np.array([-0.7, 0.7])
    p0 = np.array([0., 0.])
    
    s, H = run(nsteps, dt, q0, p0, m, k, l)

    x1 = np.array([si[0] for si in s])
    x2 = np.array([si[1] for si in s])
    
    t = np.arange(len(s))*dt
    
    x1_exact, x2_exact = exact_solution(t, q0, p0, m, k, l)
    
    plot_solutions(t/T, x1, x2, x1_exact, x2_exact)
    plot_hamiltonian(t/T, H)
