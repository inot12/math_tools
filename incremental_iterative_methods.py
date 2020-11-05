#! /home/toni/.pyenv/shims/python3
"""
Created on Nov 2, 2020

@author:toni

A collection of mathematical tools used to obtain solutions by using the
incremental-iterative approach.
The function 'func' has to be modified to return your mathematical function
every time you want to use Newton-Raphson to obtain its solution/roots.
"""

import cProfile
import math
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from expmap import PrintTimeit


def func(x):
    """Return a mathematical function.
    
    returns: function object
    
    Functions composed of special functions like trigonometric functions etc.
    must be defined with definitions of those functions from the sympy module!
    
    Examples:
    math.e**(-x) - x  # math module can be used for constants
    sp.cos(x) - 2*x  # use sympy (sp) for sin(), cos(), sqrt() etc.
    sp.sqrt(x)  # WARNING: issues arise because the solution may be negative
    math.sin(x**2) - x**3 - 1
    """
    return math.sin(x**2) - x**3 - 1


def derivative(f, x, h=1e-7):
    """Return approximate derivative by using symmetric difference quotient.
    
    f -- function object
    x -- float
    h -- float, small change in x
    
    returns: float
    """
    return (f(x+h) - f(x-h)) / (2*h)


def derive_func(f):
    """Return lambdified mathematical function and it's lambdified derivative.
    
    f -- function object
    
    returns: tuple of two function objects
    """
    x = sp.Symbol('x')
    symbolic_f = f(x)
    derivative_f = symbolic_f.diff(x)
    return sp.lambdify(x, symbolic_f), sp.lambdify(x, derivative_f)


def plot_func(f, a, b, nsteps=100):
    """Plot a function.
    
    f -- function object
    a, b -- float, plot limits
    nsteps -- int, number of steps for the plot
    """
    x = tuple((a+i*(b-a)/nsteps for i in range(nsteps+1)))
    # x = np.linspace(a, b, num=100)
    y = tuple((f(val) for val in x))
    # y = f(x)  # change math.sin -> np.sin
    fig = plt.figure(1)
    plt.plot(x, y, linewidth=2, label='f(x)')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.legend(loc='lower right')
    plt.grid(linestyle='--', linewidth=0.5)
    plt.ion()
    plt.show()
    fig.savefig('function' + '.png')
    plt.close(fig)


def error(a, b):
    """Return the relative error of two values.
    
    a -- float, reference value
    b -- float, compared value
    
    returns: float, relative error
    """
    return abs(a-b) / abs(a)
    

def newton_raphson(f, xn):
    """
    Return the solution of a function by using Newton-Raphson method.
    
    f -- function object, mathematical function of ONE argument
    df -- function object, derivative of mathematical function
    xn -- float, current initial guess
    
    returns: float
    
    x_n+1 = x_n - f(x_n)/df(x_n)
    """
    return xn - f(xn) / derivative(f, xn)


def riks(f, df, xn):
    """
    Return the solution of a function by using Riks (Arc Length) method.
    
    f -- mathematical function of ONE argument
    df -- function object, derivative of mathematical function
    xn -- float, current initial guess
    
    returns: float
    """
    pass


@PrintTimeit
def iterate(f, x0, method=newton_raphson, tol=1e-7, imax=50):
    """Return the solution of iteration procedure for the chosen method.
    
    f -- function object mathematical function of ONE argument
    x0 -- float, initial guess
    method -- function object
    tol -- float
    imax -- integer, maximum number of iterations
    
    returns: float
    
    Numerical differentiation is more efficient than symbolic differentiation.
    # NUMERIC: iterate ran in 2e-05s
    # SYMBOLIC: iterate ran in 0.06404s
    # cProfile.run() number of calls reduced from 24k to 2.2k
    """
    # f, df = derive_func(f)
    i = 0
    while error(method(f, x0), x0) > tol and i < imax:
        x0 = method(f, x0)
        i += 1
        
    if i >= imax:
        raise MaxNumberOfIterationsWarning(
            'WARNING: Exceeded maximum number of iterations.')
        
    return method(f, x0)


def increment_it(increment=0.1):
    """Return the solution to a mathematical problem by using increments."""
    pass


class MaxNumberOfIterationsWarning(RuntimeWarning):
    pass
    
    
def main():
    # print(iterate(func, 0))
    # cProfile.run('iterate(func, 0)')
    print(iterate(func, -0.8))
    cProfile.run('iterate(func, 0)')
    # print(iterate(func, 0, method=riks))
    
    plot_func(func, -1.5, 1.5)


if __name__ == "__main__":
    main()
