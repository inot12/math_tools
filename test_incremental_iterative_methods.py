#! /home/toni/.pyenv/shims/python3
"""
Created on Nov 2, 2020

@author:toni

"""

import unittest
import math
import sympy as sp
import incremental_iterative_methods as iim


class TestIncrementalIterativeMethods(unittest.TestCase):
    """Test the module incremental_iterative_methods.py."""
    
    def setUp(self):
        def f(x):
            return math.e**(-x) - x
        self.f = f
        self.df = iim.derive_func(f)[-1]
        
        def g(x):
            return x**2 - 4*x - 5
        self.g = g
        
        def h(x):
            return x**3 - x - 1
        self.h = h
        
        def k(x):
            return sp.cos(x) - 2*x
        self.k = k
    
    def test_newton_raphson(self):
        """Test the function newton_raphson"""
        self.assertAlmostEqual(iim.newton_raphson(self.f, 0, 0), 0.5)
        
    def test_iterate(self):
        """Test the function iterate()."""
        self.assertAlmostEqual(iim.iterate(self.f, 0), 0.5671, delta=1e-4)
        self.assertAlmostEqual(iim.iterate(self.g, 1), -1, delta=1e-4)
        self.assertAlmostEqual(iim.iterate(self.g, 10), 5, delta=1e-4)
        self.assertAlmostEqual(iim.iterate(self.h, 1), 1.3247, delta=1e-4)
        self.assertAlmostEqual(iim.iterate(self.k, 0.5), 0.45018, delta=1e-4)


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
# 
# 
# class ClassNameFunctionNameBadInput(unittest.TestCase):
#     """Test for bad inputs of class ClassName or function FunctionName."""
#     
#     def setUp(self):
#         """Create an instance of the class for use in all test methods."""
#         # only used for classes; not required for testing functions
#         pass  # instance initialization comes here
#     
#     def test_case1(self):  # test method names begin with 'test'
#         """f should fail if condition"""
#         # we test excetpions by assertRaises
#         # assertRaises takes 3 arguments
#         # 1. The error that should be raised
#         # 2. The function or method that should raise it
#         # 3. All the arguments of mdl.f that result in the exception
#         # Example: divide(a,b) which divides two numbers
#         # We test for zero division error
#         # self.assertRaises(ValueError, mdl.divide, 5, 0)
#         self.assertRaises(ErrorType, mdl.f, *args)
#         
#     def test_case2(self):
#         """f should fail if condition"""
#         self.assertRaises(ErrorType, mdl.f, *args)
#         
#     def test_case3(self):
#         """Alternative with context manager.
#         Use if you have more exceptions of the same type."""
#         with self.assertRaises(ValueError):
#             calc.divide(10, 0)
#             calc.divide(10, '1')

       
def main():
    unittest.main(verbosity=2)


if __name__ == "__main__":
    main()
