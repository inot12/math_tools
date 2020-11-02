#! /home/toni/.pyenv/shims/python3
"""
Created on Nov 2, 2020

@author:toni

A collection of mathematical tools used to obtain solutions by using the
incremental-iterative approach.
The function 'func' has to be modified to return your mathematical function
every time you want to use Newton-Raphson to obtain its solution/roots.
"""

import math
import numpy as np
import sympy as sp


def func(x):
    """Return a mathematical function."""
    return math.e**(-x) - x


def error(a, b):
    """Return the relative error of two values.
    
    a -- float, reference value
    b -- float, compared value
    
    returns: float, relative error
    """
    return abs(a-b)/abs(a)
    

def newton_raphson(f, xn):
    """
    Return the solution of a function by using Newton-Raphson method.
    
    f -- function object, mathematical function of ONE argument
    xn -- float, current initial guess
    
    returns: float
    
    x_n+1 = x_n - f(x_n)/df(x_n)
    """
    x = sp.Symbol('x')
    symbolic_f = f(x)
    derivative_f = symbolic_f.diff(x)
    df = sp.lambdify(x, derivative_f)
    return xn - f(xn)/df(xn)


def riks(f, xn):
    """
    Return the solution of a function by using Riks (Arc Length) method.
    
    f -- mathematical function of ONE argument
    xn -- float, current initial guess
    
    returns: float
    """
    pass


def iterate(f, x0, method=newton_raphson, tol=1e-7):
    """Return the solution of iteration procedure for the chosen method."""
    while error(method(f, x0), x0) > tol:
        x0 = method(f, x0)
    return method(f, x0)


def incremental(increment=0.1):
    """Return the solution to a mathematical problem by using increments."""
    pass
    
    
def main():
    print(newton_raphson(func, 0))
    print(iterate(func, 0))
    print(iterate(func, 0, method=riks))


if __name__ == "__main__":
    main()
