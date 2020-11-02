#! /home/toni/.pyenv/shims/python3
"""
Created on Sep 14, 2020

@author:toni
"""

import unittest

import algebra_functions as af


class FactorialKnownValues(unittest.TestCase):
    """Test for known values of factorial"""
    
    def test_factorial_known_values(self):
        """factorial should give known results with known input"""
        self.assertEqual(120, af.factorial(5))
        

class FactorialBadInput(unittest.TestCase):
    """Tests for bad input for the function factorial."""
    
    def test_non_integer(self):
        """factorial should fail if the input is non-integer"""
        self.assertRaises(af.NotIntegerError, af.factorial, 1.5)
        
    def test_negative(self):
        """factorial should fail if the input is negative"""
        self.assertRaises(af.NotPositiveError, af.factorial, -1)

       
def main():
    unittest.main(verbosity=2)


if __name__ == "__main__":
    main()
