#! /home/toni/.pyenv/shims/python3
"""
Created on Feb 1, 2021

@author:inot

Test your script with the name <scriptname.py>.
Name convention for the unit test file: <test_scriptname.py>
Remove each unnecessary line with CTRL+D
"""

import unittest

import numpy as np
import matplotlib.pyplot as plt

import numint as ni

# from module import class  # import the class you are about to test
# from module import function  # import the function you are about to test


class TestNumint(unittest.TestCase):
    """Describe the test case."""

    def setUp(self):
        pass

    def test_mid_rect(self):
        """Describe the test.
        Methods are always named 'test_'+'function/class/method'."""
        # good tests test for all the related cases
        # good tests have more than one assert!
        self.assertEqual(f(known_value), known_result)  # general case
        self.assertEqual(f(known_value), known_result)  # edge case 1
        # if using AlmostEqual, if known result x decimals, delta=1e-x
        self.assertAlmostEqual(f(known_value), known_result, delta=1e-5)


class ClassNameFunctionNameKnownValues(unittest.TestCase):
    """Test for known values of class ClassName or function FunctionName."""

    def setUp(self):
        """Create an instance of the class for use in all test methods."""
        # only used for classes; not required for testing functions
        # setUp method runs BEFORE EVERY SINGLE TEST
        # example:
        # self.instance_name = Class(*args)
        pass  # instance initialization comes here

    def tearDown(self):
        """Destroy all instances of tested files for a clean state."""
        # tearDown method runs AFTER EVERY SINGLE TEST
        pass

    @classmethod  # we work with the class and not the instance of the class
    def setUpClass(cls):
        """RUNS BEFORE ANYTHING IN THE TEST."""
        # use for costly operations that you want to do only once
        # this does not run before every single test individually,
        # but before all tests, and it does it only once
        pass

    @classmethod
    def tearDownClass(cls):
        """RUNS AFTER ANYTHING IN THE TEST."""
        pass

    def test_case1(self):  # test method names begin with 'test'
        """f should return known results for known values"""
        self.assertEqual(f(known_value), known_result)

    def test_case2(self):
        """f should return known results for known values"""
        self.assertEqual(f(known_value), known_result)


class ClassNameFunctionNameBadInput(unittest.TestCase):
    """Test for bad inputs of class ClassName or function FunctionName."""

    def setUp(self):
        """Create an instance of the class for use in all test methods."""
        # only used for classes; not required for testing functions
        pass  # instance initialization comes here

    def test_case1(self):  # test method names begin with 'test'
        """f should fail if condition"""
        # we test excetpions by assertRaises
        # assertRaises takes 3 arguments
        # 1. The error that should be raised
        # 2. The function or method that should raise it
        # 3. All the arguments of mdl.f that result in the exception
        # Example: divide(a,b) which divides two numbers
        # We test for zero division error
        # self.assertRaises(ValueError, mdl.divide, 5, 0)
        self.assertRaises(ErrorType, mdl.f, *args)

    def test_case2(self):
        """f should fail if condition"""
        self.assertRaises(ErrorType, mdl.f, *args)

    def test_case3(self):
        """Alternative with context manager.
        Use if you have more exceptions of the same type."""
        with self.assertRaises(ValueError):
            calc.divide(10, 0)
            calc.divide(10, '1')


def main():
    # unittest.main() enables you to call the test from CLI as:
    # $ python3 <test_scriptname.py>
    unittest.main(verbosity=2)  # verbose output


if __name__ == "__main__":
    main()
