#! /home/toni/.pyenv/shims/python3
"""
Created on Oct 12, 2020

@author:toni
"""

import numpy as np
import math
import scipy.integrate as scint


def mid_rect(f, x, h):
    """Integrate a function by using the midpoint rule."""
    return f(x + h/2)


def trapezium(f, x, h):
    """Integrate a function by using the trapezium rule."""
    return (f(x) + f(x+h))/2.0
 

def simpson(f, x, h):
    """Integrate a function by using the Simpson rule.
    
    Simpson rule is one of the Newton-Cotes formulas of the closed type.
    If your reference is Wikipedia we have Dx=h/2 and Simpson formula has
    Dx/3. Here we don't multiply by h, this is done in integrate, but we get
    (1/2)/3 = 1/6 and f(xi) = f(x+i*Dx) i=0,1,2."""
    return (f(x) + 4*f(x + h/2) + f(x+h))/6.0


def faster_simpson(f, a, b, steps):
    """Integrate a function with a faster implementation of Simpson rule."""
    h = (b-a)/float(steps)
    a1 = a+h/2
    s1 = sum(f(a1+i*h) for i in range(0, steps))
    s2 = sum(f(a+i*h) for i in range(1, steps))
    return (h/6.0)*(f(a)+f(b)+4.0*s1+2.0*s2)


def func(x):
    """Define the function for integration."""
    return x*x*x
    # return math.sin(x**2)*math.exp(x) + math.cos(x)**2
    # pass


def integrate(f, a, b, steps, meth):
    """Integrate a function by using the chosen method.
    
    f -- function
    a -- float, lower bound
    b -- float, upper bound
    steps -- integer, number of steps
    meth -- function, integration method
    
    Example use:
    integrate(func, 0, 10, 100, simpson)
    """
    h = (b-a)/steps
    # evaluate the function at each increment by using the specified method
    # where x=a+i*h, and return the sum of each evaluation multiplied by h
    ival = h * sum(meth(f, a+i*h, h) for i in range(steps))
    return ival


def main():
    a = 0
    b = 1
    steps = 1000
    print(f'{func.__name__} integrated from {a} to {b} in {steps} steps'
          f' by faster Simpson rule: {faster_simpson(func, a, b, steps)}')
    print(f'With integrate function: {integrate(func, a, b, steps, simpson)}')
    print(f'Trapezoidal rule: {integrate(func, a, b, steps, trapezium)}')
    print(scint.quad(func, a, b, full_output=0))
    y = np.arange(0, 10)
    print(scint.simps(y))
    
    print(f'Gauss quadrature: {scint.quadrature(func, a, b)}')


if __name__ == "__main__":
    main()
