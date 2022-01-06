#! /home/toni/.pyenv/shims/python3
"""
Created on Feb 2, 2021

@author:inot
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as scint

# This is the complete code for forward euler
# def forward_euler(f, time, y0):
#     """Integrate by using the forward Euler method"""
#     y = np.zeros((np.shape(y0)[0], len(time)))
#     y[:, 0] = y0
#     for i in range(len(time)-1):
#         h = time[i+1]-time[i]
#         y[:, i+1] = y[:, i] + h * f(y[0, i], y[1, i])
#     return y


def forward_euler_step(y, f, h):
    """The forward Euler method.
    
    y -- np.array, initial conditions
    f -- function of two variables
    h -- step size
    
    returns: evaluation of the function in the next step
    """
    return y + h * f(y[0], y[1])


def improved_euler_step(y, f, h):
    y_temp = forward_euler_step(y, f, h)  # why do we use temporary values?
    return y + h/2 * (f(y[0], y[1]) + f(y_temp[0], y_temp[1]))


def midpoint_rule(y, f, h):
    y_new = y  # also, why do we use temporary values?
    J = lambda y_new: np.array([[1, h/2 * np.cos((y_new[1] + y[1]) / 2)],
                                [-h/2, 1]])
    
    F = lambda y_new: y_new - y - h * f(*((y_new + y)/2))
    while np.linalg.norm(F(y_new-y)) > 1e-10:
        delta_y = -np.linalg.solve(J(y_new), F(y_new-y))
        y_new = y_new + delta_y
        
    return y_new
    
    
def ODE(f, time, y0, method):
    """Integrate by using the forward Euler method."""
    y = np.zeros((np.shape(y0)[0], len(time)))
    y[:, 0] = y0
    for i in range(len(time)-1):
        h = time[i+1]-time[i]
        y[:, i+1] = method(y[:, i], f, h)
    return y


def main():
    f = lambda p, q: np.array([-np.sin(q), p])

    y0 = np.array([0, 2])
    results = {}
    for n in range(5):
        N = 5 * 2**n
        time = np.linspace(0, 1, N+1)
        y0 = np.array([0, 2])  # we define the initial conditions
        result = ODE(f, time, y0, midpoint_rule)
        results[n] = (time, result)
        
    true_time = np.linspace(0, 1, 1e5)
    g = lambda x, time: np.array([-np.sin(x[1]), x[0]])
    true_solution = scint.odeint(g, y0, true_time)
        
    fig, ax = plt.subplots(2, 1, sharex='all', figsize=(10, 10))
    for (time, result) in results.values():
        ax[0].plot(time, result[0, :].T)
        ax[0].plot(time, result[1, :].T)
    ax[0].plot(true_time, true_solution[0, :], '--')
    ax[0].plot(true_time, true_solution[1, :], '--')
    plt.show()
        
    # loglog is standardly used for error plots
    plt.figure(figsize=(10, 10))
    y0 = np.array([0, 2])
    for method in [forward_euler_step, improved_euler_step, midpoint_rule]:
        e = []
        for n in range(5):
            N = 5 * 2**n
            time = np.linspace(0, 1, N+1)
            result = ODE(f, time, y0, method)
            e.append((1/(5 * 2**n),
                      np.linalg.norm(result[:, -1] - true_solution[:, -1])))
        e = np.array(e)
        plt.loglog(e[:, 0], e[:, 1], '-*')
        
    plt.loglog(e[:, 0], e[:, 0])
    plt.loglog(e[:, 0], e[:, 0]**2/10)
    plt.legend(['Forward', 'Improved', 'Midpoint', '$O(h)$', '$O(h^2)$'])
    plt.show()
    
    true_time = np.linspace(0, 4*np.pi, 1e6)
    g = lambda x, time: np.array([-np.sin(x[1]), x[0]])
    true_solution = scint.odeint(g, y0, true_time).T
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    y0 = np.array([0, 2])
    for i, Ni in enumerate([20, 200]):
        for method in ([improved_euler_step, midpoint_rule]):
            time = np.linspace(0, 4*np.pi, Ni+1)
            result = ODE(f, time, y0, method)
            ax[i].plot(result[1, :], result[0, :])
        ax[i].plot(true_solution[1, :], true_solution[0, :])
    ax[0].set_title('N = 20')
    ax[1].set_title('N = 200')
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    y0 = np.array([0, 2])
    for method in ([midpoint_rule]):
        time = np.linspace(0, 4000, 8000+1)
        result = ODE(f, time, y0, method)
        ax.plot(result[1, :], result[0, :])
    plt.show()


if __name__ == "__main__":
    main()
