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

import cProfile
import math
import warnings

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

from scipy.linalg import lu_factor, lu_solve


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
    return math.e ** (-x) - x


def derivative(f, x, h=1e-7):
    """Return approximate derivative by using symmetric difference quotient.

    f -- function object
    x -- float, point for which the derivative is calculated
    h -- float, small change in x

    returns: float
    """
    return (f(x + h) - f(x - h)) / (2 * h)


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
    x_plus[varindex] += h
    x_minus[varindex] -= h
    return (f(*x_plus) - f(*x_minus)) / (2 * h)


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
    x = tuple((a + i * (b - a) / nsteps for i in range(nsteps + 1)))
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


def error(a, b, tol=1e-8):
    """Return the relative error of two values.

    a -- float or array-like, reference value
    b -- float or array-like, value compared to reference value

    returns: float or array-like, relative error

    Also check if the reference value a is zero or contains an element that is
    zero and replace with tol. Otherwise, ZeroDivisionError is raised or NaN
    is calculated.
    """
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a == 0:
            a = tol
        return abs(a - b) / abs(a)

    if np.any(a == 0):
        a[a == 0] = tol
        # Does the same as the line above.
        # a = np.array([tol if element == 0 else element for element in np.nditer(a)])
        # print(a)
    return abs(a - b) / abs(a)


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


def newton_raphson(f, x, i, jacobian=True, inc=None):
    """
    Return the solution a system of equations by using Newton-Raphson method.

    f -- vector of function objects or tangential matrix
    x -- vector of floats or load vector
    i -- integer, current iteration
    jacobian -- boolean, by default indicates that jacobian matrix must be
    calculated, pass jacobian=False to skip the calculation of the jacobian
    inc -- float, increment that scales the variable F

    returns: vector of floats

    Parameter i exists only because both newton_raphson() and riks() are
    called in the same function and for riks() this parameter is essential.
    For newton-raphson parameter i is not used, therefore we delete it.
    """
    del i

    if jacobian:
        F = -np.array([equation(*x) for equation in (f)])
        # the commented line ensures scaling of F with lambda, but I don't
        # think I want to do this in this case
        # if inc:
            # F = F * inc
        Kt = Jacobian(f, x)
        J = Kt
    else:
        F = np.array(x)
        # scaling should be only done here
        if inc:
            # Here we should have the external load Ro and internal load Ri,
            # F should be inc*Ro-Ri(x), need to pass Ro and Ri as parameters
            # of newton_raphson and Ri has to change in each iteration, for
            # the next increment we use Ri from the last iteration of the
            # previous increment
            F = F * inc
        Kt = f(x)

    if len(f) > len(x):
        # calculate pseudo-inverse for over-constrained systems of nonlinear
        # equations because the inverse of J does not exist in this case
        Jinv = np.linalg.inv(J.T @ J) @ J.T
        dx = Jinv @ F
        # dx = lstsq(Kt, F)  # for solving non-square matrices from scipy
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


def riks(f, xi, i=0, inct=0.002, inc=None):
    """
    Return the solution of a function by using Riks (Arc Length) method.

    f -- function object, mathematical function of ONE argument
    xi -- float, current initial guess
    i -- integer, current iteration
    inct -- float, tangential increment
    inc -- float, increment

    returns: float

    i is by default 0 because this is the first step of Riks method. This way
    the Riks method can be used on its own outside an iteration loop.

    At this point of time it is not clear to me what riks should do.
    From my understanding, the first iteration is to do newton-raphson.
    After that we do the stuff that is described.
    Can you use riks without increments?
    This should work, but im not happy about globla variables. I could modify
    it to return the increments, but i don't want to do that.
    """

    if i == 0:
        global x0
        x0 = xi
        return newton_raphson(f, xi, i, inc=inc)

    xis = newton_raphson(f, xi, i, inc=inc)
    xit = newton_raphson(f, xi, i, inc=inct)
    dx0 = xi - x0
    dxs = xis - xi
    dxt = xit - xi
    coeff = (dx0.T @ dxs) / (inc * inct + dx0.T @ dxt)
    global inci
    inci = coeff * inct
    dxi = dxs - coeff * dxt
    return xi + dxi


def iterate(f, x0, method=newton_raphson, tol=1e-7, imax=50, echo=False,
            inc=None):
    """Return the solution of iteration procedure for the chosen method.

    parameters:
    f -- function object or vector of function objects
    x0 -- float or vector of floats, initial guess
    method -- function object
    tol -- float
    imax -- integer, maximum number of iterations
    echo -- boolean, call iterate() with echo=True to print iteration
    ik -- float, increment

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

    if method == riks:
        if not inc:
            inc = 1

    i = 0
    with np.printoptions(precision=5):
        p = np.get_printoptions()['precision']
        nvars = len(f)
        s = 9  # standard length of array
        tab = 4
        w = (s + p) * nvars + tab

        a2s = np.array2string
        if echo:
            print(f'Iteration: {i:<3}'
                  f'x0={a2s(x0):<{w}}'
                  f'xn={a2s(method(f, x0, i, inc=inc)):<{w}}'
                  f'Error={a2s(error(method(f, x0, i, inc=inc), x0))}')

        while np.any(error(method(f, x0, i, inc=inc), x0) > tol) and i < imax:
            if method == riks and i > 1:
                inc += inci
            x0 = method(f, x0, i, inc=inc)
            i += 1
            if echo:
                print(f'Iteration: {i:<3}'
                      f'x0={a2s(x0):<{w}}'
                      f'xn={a2s(method(f, x0, i, inc=inc)):<{w}}'
                      f'Error={a2s(error(method(f, x0, i, inc=inc), x0))}')

    if np.any(error(method(f, x0, i, inc=inc), x0) > tol) and i >= imax:
        warnings.warn(
            f'\nWARNING: Exceeded maximum number of iterations (imax={imax}).'
            ' Try changing initial guess x0 or increasing imax.',
            category=RuntimeWarning)

    if len(f) == 1:
        return float(method(f, x0, i, inc=inc))

    return method(f, x0, i, inc=inc)


def increment_it(f, x0, nsteps=10, echo=False):
    """Return the solution to a mathematical problem by using increments.

    At the moment this does not work.
    We have to update the tangential matrix with for each displacement.
    In the first increment, the tangential matrix is calculated from the
    boundary conditions.
    In every following increment, the tangential matrix is calculated
    with the displacement from the previous increment.

    I am still not confident. The solution to pass increment to iterate and
    newton-raphson to multiply the F variable with inc is not elegant.
    And I think it does not work as intended.
    It is a good starting point, this works mostly as I want it, i Just need
    to modify newton raphson to solve the KdV=F nonlienar system
    We shall see. I need to define some benchmark from simo and test it here"""

    inc = 1 / nsteps
    x = []
    x.append(x0)
    for k in range(1, nsteps + 1):
        inck = k * inc
        print(f'INCREMENT {k}')
        print(f'Lambda={inck}')
        x.append(iterate(f, x[k - 1], echo=echo, inc=inck))
        print(x[k - 1])
    return x


class MaxNumberOfIterationsWarning(RuntimeWarning):
    pass


def sf1(x1, x2):
    return x1 + 2 * x2 - 2


def sf2(x1, x2):
    return x1 ** 2 + 4 * x2 ** 2 - 4


def g1(x1, x2):
    return math.sin(x1) + 2 * x2 + 1


def g2(x1, x2):
    return x1 - 3 * x2 ** 3 + 2


def g3(x1, x2):
    return x1 ** 2 - x2 - 1


def rvs(dim=3):
    """Return a random orthogonal matrix where H.T@H = H@H.T = I.

    dim -- dimension of the matrix
    returns: random orthogonal matrix H

    I is the identity matrix.

    Example use:
    Note that you will get a different output when it comes to elements since
    the function returns a random matrix. The shape will be the same.
    >>> rvs()
    [[-0.48110765  0.52875011 -0.69925585]
     [-0.34761505 -0.84731    -0.40153399]
     [-0.80479762  0.04989078  0.59144882]]

    >>> rvs(dim=2)
    [[-0.55350488  0.83284594]
     [-0.83284594 -0.55350488]]

    Q = rvs()
    assert np.allclose(Q.T@Q, Q@Q.T) == True, 'Q is not an orthogonal matrix'
    """
    random_state = np.random
    H = np.eye(dim)
    D = np.ones((dim,))
    for n in range(1, dim):
        x = random_state.normal(size=(dim - n + 1,))
        D[n - 1] = np.sign(x[0])
        x[0] -= D[n - 1] * np.sqrt((x * x).sum())
        # Householder transformation
        Hx = (np.eye(dim - n + 1) - 2.*np.outer(x, x) / (x * x).sum())
        mat = np.eye(dim)
        mat[n - 1:, n - 1:] = Hx
        H = np.dot(H, mat)
        # Fix the last sign such that the determinant is 1
    D[-1] = (-1) ** (1 - (dim % 2)) * D.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D * H.T).T
    return H


def main():
    Q = rvs()
    assert np.allclose(Q.T @ Q, Q @ Q.T) is True, 'Q is not an orthogonal matrix'
    print(f'Q = \n{Q}')
    print(f'Q @ Q.T should be Identity matrix = \n{Q@Q.T}')
    print(f'Q.T @ Q should be Identity matrix = \n{Q.T@Q}')
    A = np.random.rand(3, 3)
    print(f'Q@A = \n{Q@A}')
    print(f'A@Q.T = \n{A@Q.T}')

    p = False
    if p:
        cProfile.run('iterate(func, 0)')
        cProfile.run('iterate(func, -0.8)')
    print(iterate(np.array([sf1, sf2]), np.array([1, 2]), echo=True))
    print(iterate(func, 0, method=riks, echo=True))
    print(iterate(np.array([sf1, sf2]), np.array([1, 2]), echo=True,
                  method=riks))
    print(increment_it(np.array([sf1, sf2]), np.array([1, 2]), echo=True))
    print(iterate(np.array([g1, g2, g3]), np.array([0.5, 1]), echo=True))
    # plot_func(func, -1.5, 1.5)


if __name__ == "__main__":
    main()
