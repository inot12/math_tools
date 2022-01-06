#! /home/toni/.pyenv/shims/python3
"""
Created on Nov 2, 2020

@author:toni

Unittests for functions from numsol.py.
"""

import unittest
import math
import numpy as np
import numsol as ns


class TestNumsol(unittest.TestCase):
    """Test the module numsol.py."""

    def setUp(self):

        def f(x):
            return math.e ** (-x) - x

        self.f = f

        def g(x):
            return x ** 2 - 4 * x - 5

        self.g = g

        def h(x):
            return x ** 3 - x - 1

        self.h = h

        def sf1(x1, x2):
            return x1 + 2 * x2 - 2

        def sf2(x1, x2):
            return x1 ** 2 + 4 * x2 ** 2 - 4

        self.sf2 = sf2
        self.sf = np.array([sf1, sf2])

        def h1(x1, x2, x3):
            return 3 * x1 - math.cos(x2 * x3) - 1.5

        def h2(x1, x2, x3):
            return 4 * x1 ** 2 - 625 * x2 ** 2 + 2 * x3 - 1

        def h3(x1, x2, x3):
            return 20 * x3 - math.e ** (-(x1 * x2)) + 9

        self.hs = np.array([h1, h2, h3])

        def k1(x1, x2, x3):
            return x1 ** 2 - 2 * x1 + x2 ** 2 - x3 + 1

        def k2(x1, x2, x3):
            return x1 * x2 ** 2 - x1 - 3 * x2 + x2 * x3 + 2

        def k3(x1, x2, x3):
            return x1 * x3 ** 2 - 3 * x3 + x2 * x3 ** 2 + x1 * x2

        self.ks = np.array([k1, k2, k3])

    def test_nr(self):
        """Test the function nr"""
        self.assertAlmostEqual(ns.nr(self.f, 0, 0), 0.5)

    def test_multivariate_derivative(self):
        """Test the function multivariate_derivative"""
        self.assertAlmostEqual(
            ns.multivariate_derivative(self.sf2, np.array([1, 2]), varindex=1),
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
                        np.array([-10 / 12, 17 / 12])),
            True)

    def test_iterate(self):
        """Test the function iterate()."""
        self.assertAlmostEqual(ns.iterate(self.f, 0), 0.5671, delta=1e-4)
        self.assertAlmostEqual(ns.iterate(self.g, 1), -1, delta=1e-4)
        self.assertAlmostEqual(ns.iterate(self.g, 10), 5, delta=1e-4)
        self.assertAlmostEqual(ns.iterate(self.h, 1), 1.3247, delta=1e-4)

        self.assertEqual(
            np.allclose(ns.iterate(self.sf, np.array([1, 2]),
                                   method=ns.newton_raphson),
                        np.array([0, 1])),
            True)

        self.assertEqual(
            np.allclose(ns.iterate(self.hs, np.array([1, 1, 1])),
                        np.array([0.83329143, 0.03948188, -0.40161823])),
            True)

        self.assertEqual(
            np.allclose(ns.iterate(self.ks, np.array([1, 2, 3])),
                        np.array([1, 1, 1])),
            True)

        self.assertEqual(
            np.allclose(ns.iterate(self.ks, np.array([0, 0, 0])),
                        np.array([1.09894258, 0.36761668, 0.14493166])),
            True)


class IterateBadInput(unittest.TestCase):
    """Test for bad inputs of function newton_raphson."""

    def setUp(self):
        """Create an instance of the class for use in all test methods."""
        self.f = np.array([])

    def test_empty_array(self):
        """newton_raphson should fail if array is empty"""
        self.assertRaises(ValueError, ns.iterate, self.f, x0=0)

    def test_zero_division(self):
        """error should return a value and not Nan if a = 0"""
        pass


def main():
    unittest.main(verbosity=2)


if __name__ == "__main__":
    main()
