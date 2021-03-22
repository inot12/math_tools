#! /home/toni/.pyenv/shims/python3
"""
Created on Oct 12, 2020

@author:toni

A collection of functions for numerical integration.
"""

import numpy as np
import scipy.integrate as scint

from scipy.linalg import expm

import expmap as em
from numsol import iterate, multivariate_newton_raphson


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
    (1/2)/3 = 1/6 and f(xi) = f(x+i*Dx) i=0,1,2.
    The Simpson rule exactly integrates a polynomial of order n-1 if n is even
    and of order n if n is odd. E.g. if n=3 the result is exact."""
    return (f(x) + 4*f(x + h/2) + f(x+h))/6.0


def faster_simpson(f, a, b, steps):
    """Integrate a function with a faster implementation of Simpson rule."""
    h = (b-a)/float(steps)
    a1 = a+h/2
    s1 = sum(f(a1+i*h) for i in range(0, steps))
    s2 = sum(f(a+i*h) for i in range(1, steps))
    return (h/6.0)*(f(a)+f(b)+4.0*s1+2.0*s2)


def newmark_ic(qn, vn, rn, qn1, beta, tau, h):
    """Initial guess for the Newmark time integration algorithm.
    qn, vn, rn -- floats, known state
    qn1 -- float, initial guess for displacement in the next time step
    """
    rn1 = (qn1-qn)/(beta*h**2) - vn/(beta*h) - (0.5-beta)*rn/beta
    vn1 = vn + h*((1-tau)*rn + tau*rn1)
    return (qn1, rn1, vn1)


def newmark_dynamics(M, D, K_T, h, beta, tau, F, P):
    pass


def newmark(f, x, h, beta=0.25, tau=0.5):
    """Newmark time integration algorithm.
    Newmark algorithm discretizes the nonlinear ordinary differential equations
    in a way that at each time step we have a set of nonlinear algebraic
    equations which can be solved with newton-raphson's method.
    
    f  -- matrix of function objects
    x -- vector of floats, dim(3),
         initial guess: displacement, velocity, acceleration
    h -- float, time step
    """
    iterate(f, x, method=multivariate_newton_raphson)


def gaussian_quadrature_rule(func, a, b):
    """Return the approximation of a definite integral of a function."""
    return scint.quadrature(func, a, b)
    

def gauss_legendre(func, a, b):
    """Gaussian quadrature rule where the domain of integration is [-1, 1]."""
    # TLDR; when you use a Gaussian quadrature rule, you use Gauss-Legendre.
    # Yields an exact results for polynomials of 2*n-1 or less.
    # The nodes x_i and weights w_i are chosen manually and i = 1, ..., n.
    return scint.quadrature(func, a, b)


def generalized_alpha(rho_inf):
    """Calculate the parameters for the generalized alpha method.
    
    rho_inf -- float, 0 <= rho_inf <=1, damping parameter
    
    Specifically, this is the Chung-Hulbert method.
    At each time step, this set of equations is solved for the unknowns
    q_n1; dq_n1, ddq_n1  and a_n1 by a Newton iterative procedure.
    """
    assert rho_inf >= 0 and rho_inf <= 1, 'Rho should be in interval [0,1].'
    
    alpha_m = (2*rho_inf-1)/(rho_inf+1)
    alpha_f = rho_inf / (rho_inf+1)
    gamma = 0.5 + alpha_f - alpha_m  # second-order accuracy condition
    beta = 0.25 * (gamma + 0.5)**2
    
    # see L5A, slide 13
    # q_n1 = q_n + h*dq_n + h**2 * (0.5-beta)*a_n + h**2 * beta*a_n1
    # dq_n1 = dq_n + h*(1-gamma)*a_n + h*gamma*a+n1
    # (1-alpha_m)*a_n1 + alpha_m*a_n = (1-alpha_f)*ddq_n1 + alpha_f*ddq_n
    
    # see L5A, slide 17
    # q_n1 = q_n * SO3_exponential_map(skew_matrix(ksi_n1))
    # in the slides q_n is not there in ksi_n1, is this a typo or correct?
    # ksi_n1 = q_n + h*dq_n + h**2 * (0.5-beta)*a_n + h**2 * beta*a_n1
    # dq_n1 = dq_n + h*(1-gamma)*a_n + h*gamma*a+n1
    # (1-alpha_m)*a_n1 + alpha_m*a_n = (1-alpha_f)*ddq_n1 + alpha_f*ddq_n
    return beta

    
def lie_group_generalized_alpha():
    pass
    
    
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


def runge_kutta_munthe_kaas(A, mn, h):
    """Numerically integrate."""
    A1 = h * A(mn)
    A2 = h * A(expm(1/2*A1)*mn)
    A3 = h * A(expm(1/2*A2 - 1/8*em.mc(A1, A2))*mn)
    A4 = h * A(expm(A3)*mn)
    return expm(1/6*(A1 + 2*A2 + 2*A3 + A4 - 1/2*em.mc(A1, A4))) * mn


def commutator_free_lie_group_method(mn, h, f):
    """Numerically integrate."""
    M1 = mn
    M2 = expm(1/2*h*f(M1))*mn
    M3 = expm(1/2*h*f(M2))*mn
    M4 = expm(h*f(M3) - 1/2*h*f(M1))*M2
    mn12 = expm(1/12*h*(3*f(M1) + 2*f(M2) + 2*f(M3) - f(M4)))*mn
    return expm(1/12*h*(-f(M1) + 2*f(M2) + 2*f(M3) + 3*f(M4)))*mn12

    
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
