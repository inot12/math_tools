#! /home/toni/.pyenv/shims/python3
"""
Created on Sep 14, 2020

@author:toni
"""

known = {0: 1}


# @PrintTimeit
def factorial(n):
    """
    Return the factorial of n.
    
    n -- integer
    
    returns: factorial of number n
      
    The variable 'known' has to be global, if it were defined within the
    function, it would be local, and it would reset every time the function is
    ran, so this would do nothing in terms of performance and recursion depth.
    For example, the recursion depth gets exceeded at n = 996. If we first
    run the function with n = 100, we should be able to calculate the values
    for n = 996 + 100 without exceeding recursion limit since we know the
    value for 100 and will not need to backtrack beyond that.
    When you wrap the recursive function, the recursion depth gets exceeded
    earlier than without wrapping.
    You can obtain the factorial of any n that is stored in 'known':
    >>> known[n]  # if you have the factorial of n stored in known
    factorial(n)
    """
    if n in known:
        return known[n]
    
    if not isinstance(n, int):
        raise NotIntegerError('factorial is only defined for integers.')
    elif n < 0:
        raise NotPositiveError(
            'factorial is not defined for negative integers.')
    else:
        fact = n * factorial(n-1)
        known[n] = fact
        return fact
    
    
class NotIntegerError(TypeError):
    pass


class NotPositiveError(ValueError):
    pass
    

def main():
    print(factorial(5))
    print(known)
    print(factorial(0))
    print(known)
    print(factorial(1))
    print(known)
    print(factorial(-1))
    print(factorial(-1))
    print(factorial(1.5))
    print(factorial(12))
    print(known)
    print(factorial(10)/factorial(5))


if __name__ == "__main__":
    main()
