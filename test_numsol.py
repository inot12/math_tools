#! /home/toni/.pyenv/shims/python3
"""
Created on Nov 2, 2020

@author:toni

"""

import unittest
import math
import numpy as np
import numsol as ns
import sympy as sp


class TestNumsol(unittest.TestCase):
    """Test the module numsol.py."""
    
    def setUp(self):
        def f(x):
            return math.e**(-x) - x
        self.f = f
        # self.df = ns.derive_func(f)[-1]
        
        def g(x):
            return x**2 - 4*x - 5
        self.g = g
        
        def h(x):
            return x**3 - x - 1
        self.h = h
        
        def sf1(x1, x2):
            return x1 + 2*x2 - 2
        
        def sf2(x1, x2):
            return x1**2 + 4*x2**2 - 4
        
        self.sf = np.array([sf1, sf2])
        
        def of(x1, x2):
            return x1**2 + 4*x2**2 - 4
        self.of = of
        
        def k(x):
            return sp.cos(x) - 2*x
        self.k = k
    
    def test_nr(self):
        """Test the function nr"""
        self.assertAlmostEqual(ns.nr(self.f, 0, 0), 0.5)
        
    def test_multivariate_derivative(self):
        """Test the function multivariate_derivative"""
        self.assertAlmostEqual(
            ns.multivariate_derivative(self.of, np.array([1, 2]), varindex=1),
            16)
        
    def test_Jacobian(self):
        """Test the function multivariate_Jacobian"""
        self.assertEqual(
            np.allclose(ns.Jacobian(self.sf, np.array([1, 2])),
                        np.array([[1, 2], [2, 16]])),
            True)
        
    def test_newton_raphson(self):
        """Test the function newton_raphson"""
        self.assertEqual(
            np.allclose(ns.newton_raphson(self.sf, np.array([1, 2]), 0),
                        np.array([-10/12, 17/12])),
            True)
        
    def test_iterate(self):
        """Test the function iterate()."""
        self.assertAlmostEqual(ns.iterate(self.f, 0), 0.5671, delta=1e-4)
        self.assertAlmostEqual(ns.iterate(self.g, 1), -1, delta=1e-4)
        self.assertAlmostEqual(ns.iterate(self.g, 10), 5, delta=1e-4)
        self.assertAlmostEqual(ns.iterate(self.h, 1), 1.3247, delta=1e-4)
        
        # test for symbolic functions from sympy
        self.assertAlmostEqual(ns.iterate(self.k, 0.5), 0.45018, delta=1e-4)
        
        # test iterate for multivariate nr; not working atm
        self.assertEqual(
            np.allclose(ns.iterate(self.sf, np.array([1, 2]),
                                   method=ns.newton_raphson),
                        np.array([0, 1])),
            True)


class IterateBadInput(unittest.TestCase):
    """Test for bad inputs of function newton_raphson."""

    def setUp(self):
        """Create an instance of the class for use in all test methods."""
        self.f = np.array([])

    def test_empty_array(self):  # test method names begin with 'test'
        """newton_raphson should fail if array is empty"""
        self.assertRaises(ValueError, ns.iterate, self.f, x0=0)


# class ClassNameFunctionNameKnownValues(unittest.TestCase):
#     """Test for known values of class ClassName or function FunctionName."""
#
#     def setUp(self):
#         """Create an instance of the class for use in all test methods."""
#         # only used for classes; not required for testing functions
#         # setUp method runs BEFORE EVERY SINGLE TEST
#         # example:
#         # self.instance_name = Class(*args)
#         pass  # instance initialization comes here
#
#     def tearDown(self):
#         """Destroy all instances of tested files for a clean state."""
#         # tearDown method runs AFTER EVERY SINGLE TEST
#         pass
#
#     @classmethod  # we work with the class and not the instance of the class
#     def setUpClass(cls):
#         """RUNS BEFORE ANYTHING IN THE TEST."""
#         # use for costly operations that you want to do only once
#         # this does not run before every single test individually,
#         # but before all tests, and it does it only once
#         pass
#
#     @classmethod
#     def tearDownClass(cls):
#         """RUNS AFTER ANYTHING IN THE TEST."""
#         pass
#
#     def test_case1(self):  # test method names begin with 'test'
#         """f should return known results for known values"""
#         self.assertEqual(f(known_value), known_result)
#
#     def test_case2(self):
#         """f should return known results for known values"""
#         self.assertEqual(f(known_value), known_result)

       
def main():
    unittest.main(verbosity=2)


if __name__ == "__main__":
    main()
