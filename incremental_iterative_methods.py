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
import numpy as np
import sympy as sp

from expmap import PrintTimeit


def func(x):
    """Return a mathematical function."""
    return math.e**(-x) - x


# derive the function only once instead each time in newton_raphson()
# during the while loop execution when iterate() is called
# OLD CODE: iterate ran in 0.01505s
# NEW CODE: iterate ran in 0.00162s
# cProfile.run() number of calls reduced from 24k to 2.2k
def derive_func(f):
    """Return the lambdified derivative of mathematical function.
    
    f -- function object
    
    returns: function object
    """
    x = sp.Symbol('x')
    symbolic_f = f(x)
    derivative_f = symbolic_f.diff(x)
    return sp.lambdify(x, derivative_f)


def error(a, b):
    """Return the relative error of two values.
    
    a -- float, reference value
    b -- float, compared value
    
    returns: float, relative error
    """
    return abs(a-b)/abs(a)
    

def newton_raphson(f, df, xn):
    """
    Return the solution of a function by using Newton-Raphson method.
    
    f -- function object, mathematical function of ONE argument
    df -- function object, derivative of mathematical function
    xn -- float, current initial guess
    
    returns: float
    
    x_n+1 = x_n - f(x_n)/df(x_n)
    """
    return xn - f(xn)/df(xn)


def riks(f, df, xn):
    """
    Return the solution of a function by using Riks (Arc Length) method.
    
    f -- mathematical function of ONE argument
    xn -- float, current initial guess
    
    returns: float
    """
    pass


def iterate(f, x0, method=newton_raphson, tol=1e-7):
    """Return the solution of iteration procedure for the chosen method."""
    df = derive_func(f)
    while error(method(f, df, x0), x0) > tol:
        x0 = method(f, df, x0)
    return method(f, df, x0)


def incremental(increment=0.1):
    """Return the solution to a mathematical problem by using increments."""
    pass
    
    
def main():
    print(newton_raphson(func, derive_func(func), 0))
    # cProfile.run('newton_raphson(func, 0)')
    print(iterate(func, 0))
    cProfile.run('iterate(func, 0)')
    # print(iterate(func, 0, method=riks))


if __name__ == "__main__":
    main()
