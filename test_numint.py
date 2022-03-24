#! /home/toni/.pyenv/shims/python3
"""
Created on Feb 1, 2021

@author:inot
"""

import unittest

import numpy as np

import numint as ni


class FasterSimpsonGoodInput(unittest.TestCase):
    """Test for good input of function faster_simpson()."""

    def setUp(self):
        self.a = 0
        self.b = 1
        self.steps = 1000

        def func(x):
            """Define the function for integration."""
            return x * x * x

        self.f = func

    def test_faster_simpson(self):
        """faster_simpson() should return known results for known values."""
        self.assertAlmostEqual(
            ni.faster_simpson(
                self.f, self.a, self.b, self.steps), 0.25, delta=1e-8)


def main():
    unittest.main(verbosity=2)


if __name__ == "__main__":
    main()
