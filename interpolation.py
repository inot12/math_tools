#! /home/toni/.pyenv/shims/python3
"""
Created on Sep 18, 2020

@author:toni

A collection of interpolation algorithms.
"""

import numpy as np
import math
import sympy
from math_functions.algebra_functions import factorial

   
def lagrange_polynomial(points):
    """Return the Lagrange polynomial of a data set.
    
    points - np.array of dim nx2, where n is the number of points
    
    returns: expression for the interpolating function
    """
    x, k = sympy.symbols('x k')
    n = len(points)
    L = []
    
    for i in range(n):
        Li = 1
        xi = points[i, 0]
        
        for j in range(n):
            if j != i:
                Li *= (x-points[j, 0]) / (xi-points[j, 0])
            
        L.append(Li)
        
    interpolation_polynomial = 0
    
    for k in range(n):
        interpolation_polynomial += L[k] * points[k, 1]
    
    return sympy.simplify(interpolation_polynomial)


def newton_polynomial(data):
    """Return the Newton polynomial of order n-1 for n data points.
    
    data -- array like, set of n points dim(2,n)
    
    returns: expression for the interpolating function
    """
    x = sympy.Symbol('x')
    xData = data[0, :]  # first column
    yData = data[1, :]  # second column
    a = newton_coefficients(xData, yData)
    res = a[0]
    
    for i in range(1, len(a)):
        c = a[i]
        for k in range(0, i):
            c *= (x-xData[k])
        res += c
        
    return sympy.simplify(res)
        
        
def evalNewtonPoly(a, xData, x):
    """Evaluate the Newton polynomial"""
    n = len(xData)-1  # order of polynomial
    p = a[n]
    for i in range(1, n+1):
        p = a[n-i] + (x-xData[n-i])*p
    return p


def newton_coefficients(xData, yData):
    """Return coefficients of the of Newton polynomial.
    
    xData -- array like of numeric values, dim (1,n), n-number of points
    yData -- array like of numeric values, dim (1,n), n-number of points
    
    returns: array like of numeric values
    """
    n = len(xData)
    a = np.zeros((n, n))
    for i in range(0, n):
        a[i, 0] = yData[i]  # fill the first column of a with yData values
    for j in range(1, n):
        for k in range(0, n-j):
            a[k, j] = (a[k+1, j-1] - a[k, j-1])/(xData[k+j]-xData[k])
    return a[0, :]  # return the zeroth row
    
    
def hermite_polynomial(n, method='physics'):
    """Return the Hermite polynomials as defined by physicists or probabilists.
    
    n -- integer, order of the polynomial
    method -- string, either 'physics' (default) or 'probability'
    
    returns: expression for the interpolating function
    
    Example use:
    >>> hermite_polynomial(5)
    32*x**5 - 160*x**3 + 120*x
    >>> em.hermite_polynomial(5, method='physics')
    32*x**5 - 160*x**3 + 120*x
    >>> hermite_polynomial(5, method='probability')
    1.0*x**5 - 10.0*x**3 + 15.0*x
    """
    x, k = sympy.symbols('x k')
    
    n_even = int(n/2)
    n_odd = int((n-1)/2)
    
    k_even = 2*k
    k_odd = 2*k + 1
    
    base = 2*x
    factor = 1
    
    if method == 'probability':
        base = math.sqrt(2)*x
        factor = 2**(-n/2)
    
    if n % 2 == 0:
        return hermite_summand(n, k, n_even, k_even, base, factor)
    else:
        return hermite_summand(n, k, n_odd, k_odd, base, factor)
    
    
def hermite_summand(n, k, m, p, b, multiplier):
    """Determine the summand part of Hermite polynomial.
    
    n -- integer, polynomial order
    k -- sympy.Symbol
    m -- integer, limit index of the sum
    p -- expression for the power of polynomial
    b -- integer*sympy.Symbol or float*sympy.Symbol, polynomial base
    multiplier -- a coefficient multiplying the sum
    
    returns: expression for the interpolating function
    
    Parameters m and p of this function take different arguments depending on
    whether the order n of the polynomial is even or odd.
    m = int(n/2) or m = int((n-1)/2)
    p = 2*k or p = 2*k + 1
    Parameters b and multiplier depend on the definition of Hermite polynomial:
    physicists' definition or probabilists' definition.
    b = 2*x or b = math.sqrt(2)*x
    multiplier = 1 or multiplier = 2**(-n/2)
    """
    sum_multiplier = factorial(n)
    
    numerator = (-1)**(m-k)
    denominator = sympy.factorial(p) * sympy.factorial(m-k)
    summand_factor = numerator / denominator
    
    summand = summand_factor * (b)**(p)
    hermite_sum = multiplier * sum_multiplier * sympy.Sum(summand, (k, 0, m))

    return hermite_sum.doit()


def main():
    # Ok, you schmuck. Convert those silly print statements to unit tests.
    print(hermite_polynomial(10))
    print(hermite_polynomial(5))
    print(hermite_polynomial(1))
    print(hermite_polynomial(0))
    print(hermite_polynomial(0, 'probability'))
    print(hermite_polynomial(1, 'probability'))
    print(hermite_polynomial(5, 'probability'))
    print(hermite_polynomial(10, 'probability'))
    
    data = np.array([[0.534, 1.523], [0.353, 0.874], [1.321, 8.431]])
    print(lagrange_polynomial(data))
    pts = np.array([[1, 1], [2, 8], [3, 27], [4, 64]])
    print(lagrange_polynomial(pts))
    pt = np.array([[1, 1], [2, 4], [3, 9]])
    print(lagrange_polynomial(pt))
    print(hermite_polynomial(20))
    xData = [1, 2, 3]
    yData = [6, 11, 18]
    print(newton_coefficients(xData, yData))
    data = np.array([[1, 2, 3], [6, 11, 18]])
    print(newton_polynomial(data))
    data1 = np.array([[0, 10, 15, 20, 22.5, 30],
                      [0, 227.04, 362.78, 517.35, 602.97, 901.67]])
    print(newton_polynomial(data1))


if __name__ == "__main__":
    main()
