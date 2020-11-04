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
    """
    return math.e**(-x) - x


def derivative(f, x, h=1e-7):
    """Return approximate derivative by using symmetric difference quotient.
    
    f -- function object
    x -- float
    h -- float, small change in x
    
    returns: float
    """
    return (f(x+h) - f(x-h)) / (2*h)


@PrintTimeit
def derive_func(f):
    """Return lambdified mathematical function and it's lambdified derivative.
    
    f -- function object
    
    returns: tuple of two function objects
    """
    x = sp.Symbol('x')
    symbolic_f = f(x)
    derivative_f = symbolic_f.diff(x)
    return sp.lambdify(x, symbolic_f), sp.lambdify(x, derivative_f)


def error(a, b):
    """Return the relative error of two values.
    
    a -- float, reference value
    b -- float, compared value
    
    returns: float, relative error
    """
    return abs(a-b) / abs(a)
    

def newton_raphson(f, df, xn):
    """
    Return the solution of a function by using Newton-Raphson method.
    
    f -- function object, mathematical function of ONE argument
    df -- function object, derivative of mathematical function
    xn -- float, current initial guess
    
    returns: float
    
    x_n+1 = x_n - f(x_n)/df(x_n)
    """
    return xn - f(xn) / df(xn)


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
def iterate(f, x0, method=newton_raphson, tol=1e-7, imax=10):
    """Return the solution of iteration procedure for the chosen method.
    
    f -- function object mathematical function of ONE argument
    x0 -- float, initial guess
    method -- function object
    tol -- float
    imax -- integer, maximum number of iterations
    
    returns: float
    
    Derive the function only once instead each time in newton_raphson()
    # during the while loop execution when newton_raphson() is called
    # OLD CODE: iterate ran in 0.01505s
    # NEW CODE: iterate ran in 0.00162s
    # AFTER LAMBDIFYING f and df: iterate ran in 0.06404s
    # cProfile.run() number of calls reduced from 24k to 2.2k
    """
    f, df = derive_func(f)
    i = 0
    while error(method(f, df, x0), x0) > tol and i < imax:
        x0 = method(f, df, x0)
        i += 1
    return method(f, df, x0)


def increment_it(increment=0.1):
    """Return the solution to a mathematical problem by using increments."""
    pass
     

@PrintTimeit
def nr(f, x_guess=None, max_num_iter=100, tolerance=1e-4, alpha=1.0,
       print_info=True):
    """
    Author: Christian Howard
    Function for representing a Newton-Raphson iteration for multidimensional
    systems of equations.
    :param f: function class that must define the following methods:
        - numDims(): Method that returns an integer number of variables in the
        system of equations
        - getJacobian(np.ndarray): Method to compute the Jacobian of the
        system of equations at the current root estimate.The output is an
        n by n matrix where n is the number of variables in the system of
        equations
        - __call__(np.ndarray): Method to make this class act like a function
        operating on some input x
    :param x_guess: an initial guess for the Newton-Raphson iteration
    :param max_num_iter: a maximum number of iterations that will be taken
    :param tolerance: a tolerance that will stop the sequence once the error
    drops below it
    :param alpha: A coefficient that can tune the Newton-Raphson stepsize.
    Recommend setting alpha <= 1.
    :return: A tuple with the root estimate, final error for the root, and the
    number of iterations it took
    """
    # set the initial guess
    if x_guess is None:
        x_guess = np.random.rand(f.numDims())
    x = x_guess
 
    # compute function value at initial guess
    fx = f(x)
 
    # define the initial value for the error and the starting iteration count
    err = np.linalg.norm(fx)
    iter_ = 0
 
    if print_info:
        print(f"Iteration {iter_}: Error of {err} with an estimate of {x}")
 
    # perform the Newton-Raphson iteration algo
    while err > tolerance and iter_ < max_num_iter:
 
        # perform newton step
        x = x - alpha * np.linalg.solve(f.getJacobian(x), fx)
 
        # update the function value at the new root estimate
        fx = f(x)
 
        # compute the current root error
        err = np.linalg.norm(fx)
 
        # update the iteration counter
        iter_ = iter_ + 1
 
        # print useful message
        if print_info:
            print(f"Iteration {iter_}: Error of {err} with an estimate of {x}")
 
    # return the root estimate, 2-norm error of the estimate, and iteration
    # count we ended at
    return (x, err, iter_)


class HuggingEllipses:
    """Defines the equations to be solved by nr().
    f1(x) = (x1-1)**2 + 4*x2**2 -1
    f2(x) = (x1-2)**2 + 4*x2**2 -4
    """
    def __init__(self):
        self.name = "HuggingEllipses"
 
    def numDims(self):
        return 2
 
    def getJacobian(self, x):
        return np.array([[2*(x[0]-1), 8*x[1]], [2*(x[0]-2), 8*x[1]]])
 
    def __call__(self, x):
        f = np.zeros((2, ))
        f[0] = (x[0]-1)**2 + 4*x[1]**2 - 1
        f[1] = (x[0]-2)**2 + 4*x[1]**2 - 4
        return f
    
    
def main():
    # print(newton_raphson(func, derive_func(func)[-1], 0))
    # cProfile.run('newton_raphson(func, derive_func(func)[-1], 0)')
    print(iterate(func, 0))
    # cProfile.run('iterate(func, 0)')
    # print(iterate(func, 0, method=riks))
    # f = HuggingEllipses()
    # xn = nr(f, tolerance=1e-7)


if __name__ == "__main__":
    main()
