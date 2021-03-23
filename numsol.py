#! /home/toni/.pyenv/shims/python3
"""
Created on Nov 2, 2020

@author:toni

Implementation of incremental-iterative methods: Newton-Raphson & Riks method.
One variable Newton-Raphson and multivariate Newton-Raphson is supported.

If some algebraic equations are local at a specific element, we can solve them
at the element level which reduces the number of unknowns at the global level
by employing the nested iterative sub-iterative procedure. This improves
numerical efficiency, but the complexity of the algorithm is increased.

The function 'func' has to be modified to return your mathematical function
every time you want to use Newton-Raphson or Riks to obtain its solution/roots.
"""

# import cProfile
import math
import warnings

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

from scipy.linalg import lu_factor, lu_solve, lstsq


def func(x):
    """Return a mathematical function.
    
    returns: function object
    
    Special functions can be from any module: e.g. math, numpy, sympy
    
    Examples:
    math.e**(-x) - x  # math module can be used for constants
    sp.cos(x) - 2*x  # sympy (sp) can be used for sin(), cos(), sqrt() etc.
    math.sin(x**2) - x**3 - 1
    np.sin(x**2) - x**3 - 1
    """
    return math.e**(-x) - x


def derivative(f, x, h=1e-7):
    """Return approximate derivative by using symmetric difference quotient.
    
    f -- function object
    x -- float, point for which the derivative is calculated
    h -- float, small change in x
    
    returns: float
    """
    return (f(x+h) - f(x-h)) / (2*h)


def multivariate_derivative(f, x, varindex, h=1e-7):
    """Numerical derivative of a function of multiple variables.
    
    f -- function object
    x -- vector of floats, np.array, initial guess
    varindex -- integer, index of vector x denoting the variable we
                differentiate with respect to
                
    returns: vector of floats
    
    pd(u)/pd(x) = (u(x+h, y, z) - u(x-h, y, z)) / 2h
    
    By default np.array assigns dtype to the minimum type required to hold the
    objects in the sequence. If only integers are in the array x, dtype will be
    dtype=np.int64 and numerical derivation fails. x.astype(np.float64) ensures
    that the type stores float numbers.
    """
    x = x.astype(np.float64)
    x_plus = np.copy(x)
    x_minus = np.copy(x)
    x_plus[varindex] = x_plus[varindex] + h
    x_minus[varindex] = x_minus[varindex] - h
    return (f(*x_plus) - f(*x_minus)) / (2*h)


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
    plt.plot(x, y, linewidth=2, label='$f(x)$')
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
    b -- float, value compared to reference value
    
    returns: float, relative error
    """
    return abs(a-b) / abs(a)
    

def nr(f, xn, i):
    """
    Return the solution of a function by using Newton-Raphson method.
    
    f -- function object, mathematical function of ONE argument
    xn -- float, current initial guess
    i -- integer, current iteration
    
    returns: float
    x_n+1 = x_n - f(x_n)/df(x_n)
    
    Parameter i exists only because both newton_raphson() and riks() are
    called in the same function and for riks() this parameter is essential.
    For newton-raphson parameter i is not used, therefore we delete it.
    """
    del i
    return xn - f(xn) / derivative(f, xn)


def newton_raphson(f, x, i, jacobian=True):
    """
    Return the solution a system of equations by using Newton-Raphson method.
    
    f -- vector of function objects or tangential matrix
    x -- vector of floats or load vector
    i -- integer, current iteration
    jacobian -- boolean, by default indicates that jacobian matrix must be
    calculated, pass jacobian=False to skip the calculation of the jacobian
    
    returns: vector of floats
    
    Parameter i exists only because both newton_raphson() and riks() are
    called in the same function and for riks() this parameter is essential.
    For newton-raphson parameter i is not used, therefore we delete it.
    """
    del i
    
    if jacobian:
        F = -np.array([equation(*x) for equation in (f)])
        Kt = Jacobian(f, x)
        J = Kt
    else:
        F = np.array(x)
        Kt = f(x)
    
    if len(f) > len(x):
        # calculate pseudo-inverse for over-constrained systems of nonlinear
        # equations because the inverse of J does not exist in this case
        Jinv = np.linalg.inv(J.T@J) @ J.T
        dx = Jinv @ F
        # dx = lstsq(Kt, F)
    else:
        lu, piv = lu_factor(Kt)
        dx = lu_solve((lu, piv), F)
#         dx = np.linalg.solve(J, F)
        
    return x + dx


def Jacobian(f, x):
    """Calculate the Jacobian of f for value x.
    f -- vector of function objects
    x -- vector of floats
    
    returns: matrix of floats
    """
    J = np.zeros((len(f), len(x)))
    for i, equation in enumerate(f):
        for var in range(len(x)):
            J[i, var] = multivariate_derivative(equation, x, var)
    return J
    

def riks(f, xi, i=0, inct=0.002):
    """
    Return the solution of a function by using Riks (Arc Length) method.
    
    f -- function object, mathematical function of ONE argument
    xi -- float, current initial guess
    i -- integer, current iteration
    inct -- float, default increment
    
    returns: float
    
    i is by default 0 because this is the first step of Riks method. This way
    the Riks method can be used on its own outside an iteration loop.
    """
    if i != 0:
        xis = newton_raphson(f, xi, i)
        xit = newton_raphson(f, inct*xi, i)
    else:
        return newton_raphson(f, xi, i)


def iterate(f, x0, method=newton_raphson, tol=1e-7, imax=50, echo=False):
    """Return the solution of iteration procedure for the chosen method.
    
    parameters:
    f -- function object or vector of function objects
    x0 -- float or vector of floats, initial guess
    method -- function object
    tol -- float
    imax -- integer, maximum number of iterations
    echo -- boolean, call iterate() with echo=True to print iteration
    
    returns: float or a vector of floats
    
    raises:
    VauleError
    If f is an empty numpy array.
    """
    if callable(f):
        f = np.array([f])
        
    if not isinstance(x0, np.ndarray):
        x0 = np.array([x0])
                    
    if f.size == 0:
        raise ValueError('f cannot be an empty numpy array')
    
    i = 0
    with np.printoptions(precision=5):
        p = np.get_printoptions()['precision']
        nvars = len(f)
        s = 9  # standard length of array
        tab = 4
        w = (s+p) * nvars + tab
        
        if echo:
            print(f'Iteration: {i:<3}'
                  f'x0={np.array2string(x0):<{w}}'
                  f'xn={np.array2string(method(f, x0, i)):<{w}}'
                  f'Error={np.array2string(error(method(f, x0, i), x0))}')
    
        while np.any(error(method(f, x0, i), x0) > tol) and i < imax:
            x0 = method(f, x0, i)
            i += 1
            if echo:
                print(f'Iteration: {i:<3}'
                      f'x0={np.array2string(x0):<{w}}'
                      f'xn={np.array2string(method(f, x0, i)):<{w}}'
                      f'Error={np.array2string(error(method(f, x0, i), x0))}')
    
    if np.any(error(method(f, x0, i), x0) > tol) and i >= imax:
        warnings.warn(
            f'\nWARNING: Exceeded maximum number of iterations (imax={imax}).'
            ' Try changing initial guess x0 or increasing imax.',
            category=RuntimeWarning)
        
    if len(f) == 1:
        return float(method(f, x0, i))
    
    return method(f, x0, i)


def increment_it(f, x0, nsteps=10, echo=False):
    """Return the solution to a mathematical problem by using increments.
    
    At the moment this does not work.
    We have to update the tangential matrix with for each displacement.
    In the first increment, the tangential matrix is calculated from the
    boundary conditions.
    In every following increment, the tangential matrix is calculated
    with the displacement from the previous increment."""
    
    inc = 1/nsteps
    x = []
    for k in range(1, nsteps+1):
        xinc = k * inc * x0
        print(f'INCREMENT {k}')
        x.append(iterate(f, xinc, echo=echo))
    return x


class MaxNumberOfIterationsWarning(RuntimeWarning):
    pass


def sf1(x1, x2):
    return x1 + 2*x2 - 2
    
    
def sf2(x1, x2):
    return x1**2 + 4*x2**2 - 4


def g1(x1, x2):
    return math.sin(x1) + 2*x2 + 1


def g2(x1, x2):
    return x1 - 3*x2**3 + 2


def g3(x1, x2):
    return x1**2 - x2 - 1


def h1(x1, x2, x3):
    return 3*x1 - math.cos(x2*x3) - 1.5


def h2(x1, x2, x3):
    return 4*x1**2 - 625*x2**2 + 2*x3 - 1


def h3(x1, x2, x3):
    return 20*x3 - math.e**-(x1*x2) + 9


def k1(x1, x2, x3):
    return x1**2 - 2*x1 + x2**2 - x3 + 1


def k2(x1, x2, x3):
    return x1*x2**2 - x1 - 3*x2 + x2*x3 + 2


def k3(x1, x2, x3):
    return x1*x3**2 - 3*x3 + x2*x3**2 + x1*x2
    
    
def main():
    # print(iterate(func, 0, echo=True))
    # cProfile.run('iterate(func, 0)')
    # print(iterate(func, -100, imax=25, echo=True))
    # cProfile.run('iterate(func, -0.8)')
    # print(iterate(func, 0, method=riks))
    print(iterate(np.array([sf1, sf2]), np.array([1, 2]), echo=True))
    print(increment_it(np.array([sf1, sf2]), np.array([1, 2]), echo=True))
    # print(iterate(np.array([g1, g2, g3]), np.array([0.5, 1]), echo=True))
    # print(iterate(np.array([h1, h2, h3]), np.array([1, 1, 1]), echo=True))
    # print(iterate(np.array([k1, k2, k3]), np.array([1, 2, 3]), echo=True))
    # print(iterate(np.array([k1, k2, k3]), np.array([0, 0, 0]), echo=True))
    # print(iterate(np.array([]), np.array([1]), echo=True))
    
    # plot_func(func, -1.5, 1.5)
    

if __name__ == "__main__":
    main()
