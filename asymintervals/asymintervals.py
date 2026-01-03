import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

class AIN:
    def __init__(self, lower: float, upper: float, expected: float = None):
        """
        Initialize an Asymmetric Interval Number (AIN) with specified bounds and an optional expected value.

        This constructor creates an instance of AIN using `lower` and `upper` bounds to define the interval.
        Optionally, an `expected` value within this range can be provided. If `expected` is not specified,
        it defaults to the midpoint of `lower` and `upper`. The `expected` value must lie within the interval
        `[lower, upper]`. The constructor also calculates the two parameters defining the AIN distribution (`alpha`, `beta`), 
        the degree of asymmetry (`asymmetry`), and the variance (`D2`) based on the specified bounds and expected value.

        Parameters
        ----------
        lower : float
            The lower bound of the interval. Must be less or equal than `upper`.
        upper : float
            The upper bound of the interval. Must be greater or equal than `lower`.
        expected : float, optional
            The expected value within the interval. Defaults to the midpoint `(lower + upper) / 2` if not provided.

        Raises
        ------
        TypeError
            If `lower`, `upper`, or `expected` are not of type float or int.
        ValueError
            If `expected` is not within the range `[lower, upper]`.

        Attributes
        ----------
        lower : float
            The lower bound of the interval.
        upper : float
            The upper bound of the interval.
        expected : float
            The expected value within the interval [`lower`, `upper`].
        alpha : float
            The distribution parameter for the interval [`lower`, `expected`], calculated when 
            `lower` is less than `upper`. If `lower` equals `upper`, the parameter is set to 1.
        beta : float
            The distribution parameter for the interval [`expected`, `upper`], calculated when 
            `lower` is less than `upper`. If `lower` equals `upper`, the parameter is set to 1.
        asymmetry : float
            The asymmetry degree of the interval, representing the relative position of `expected`
            between `lower` and `upper`. If `lower` equals `upper`, the `asymmetry` is set to 0.
        D2 : float
            A parameter representing the variance of the interval, derived from the degree of 
            asymmetry and the specified bounds.

        Examples
        --------
        Creating an AIN with a specified expected value:
        >>> a = AIN(0, 10, 8)
        >>> print(a)
        [0.0000, 10.0000]_{8.0000}

        Creating an AIN with a default expected value:
        >>> b = AIN(0, 10)
        >>> repr(b)
        'AIN(0, 10, 5.0)'

        Attempting to create an improper AIN:
        >>> c = AIN(1, 2, 3)
        Traceback (most recent call last):
        ...
        ValueError: It is not a proper AIN 1.0000, 2.0000, 3.0000
        """
        if not isinstance(lower, (int, float)):
            raise TypeError('lower must be int or float')
        if not isinstance(upper, (int, float)):
            raise TypeError('upper must be int or float')
        if not (expected is None or isinstance(expected, (int, float))):
            raise TypeError('expected must be int or float')
        if expected is None:
            expected = (lower + upper) / 2
        if not (lower <= expected <= upper):
            raise ValueError(f'It is not a proper AIN {lower:.4f}, {upper:.4f}, {expected:.4f}')
        self.lower = lower
        self.upper = upper
        self.expected = expected

        if self.lower == self.upper:
            if self.expected != self.lower:
                raise ValueError(
                    f"For lower==upper expected must equal bounds "
                    f"(got {self.expected:.4f})"
                )

            self.alpha = 1.0
            self.beta = 1.0
            self.asymmetry = 0.0
            self.D2 = 0.0
        else:
            if not (self.lower + np.finfo(float).eps < self.expected < self.upper - np.finfo(float).eps):
                raise ValueError(
                    f"expected must lie strictly inside (lower, upper) "
                    f"(got {self.lower:.4f}, {self.upper:.4f}, {self.expected:.4f})"
                )
            self.alpha = (self.upper - self.expected) / ((self.upper - self.lower) * (self.expected - self.lower))
            self.beta = (self.expected - self.lower) / ((self.upper - self.lower) * (self.upper - self.expected))
            self.asymmetry = (self.lower + self.upper - 2 * self.expected) / (self.upper - self.lower)
            self.D2 = self.alpha * (self.expected ** 3 - self.lower ** 3) / 3 + self.beta * (
                    self.upper ** 3 - self.expected ** 3) / 3 - self.expected ** 2

    def __repr__(self):
        """
        Return an unambiguous string representation of the AIN instance.  
        The representation includes the class name `AIN`, followed by the 
        `lower`, `upper`, and `expected` values enclosed in parentheses.

        Returns
        -------
        str
            A string that accurately reflects the construction of the instance.

        Examples
        --------
        >>> a = AIN(0, 10, 8)
        >>> repr(a)
        'AIN(0, 10, 8)'

        >>> b = AIN(0, 10)
        >>> repr(b)
        'AIN(0, 10, 5.0)'
        """
        return f"AIN({self.lower}, {self.upper}, {self.expected})"

    def __str__(self):
        """
        Return a human-readable string representation of the AIN instance.

        The string is formatted as '[lower, upper]_{expected}' where 'lower', 'upper',
        and 'expected' are displayed with four decimal places. This format is designed 
        to be clear, concise, and user-friendly, making it well-suited for printing and 
        easy interpretation by end-users.

        Returns
        -------
        str
            A string representation of the `AIN` instance, formatted to four decimal
            places for each numeric value.

        Examples
        --------
        >>> a = AIN(0, 10, 8)
        >>> print(a)
        [0.0000, 10.0000]_{8.0000}

        >>> b = AIN(0, 10)
        >>> print(b)
        [0.0000, 10.0000]_{5.0000}
        """
        return f"[{self.lower:.4f}, {self.upper:.4f}]_{{{self.expected:.4f}}}"

    def __neg__(self):
        """
        Returns a new `AIN` instance representing the negation of the current 
        instance (the additive inverse of the interval).

        The negation of an AIN instance is achieved by reversing the signs of the 
        `lower`, `upper`, and `expected` values:
        - The new `lower` bound becomes the negation of the original `upper` bound.
        - The new `upper` bound becomes the negation of the original `lower` bound.
        - The new `expected` value becomes the negation of the original `expected` value.

        Returns
        -------
        AIN
            A new `AIN` instance representing the additive inverse of the interval.

        Examples
        --------
        >>> a = AIN(1, 10, 8)
        >>> print(-a)
        [-10.0000, -1.0000]_{-8.0000}

        >>> a = AIN(2, 10)
        >>> print(-a)
        [-10.0000, -2.0000]_{-6.0000}

        >>> a = np.array([AIN(0, 10), AIN(2, 8, 7)])
        >>> print(-a)
        [AIN(-10, 0, -5.0) AIN(-8, -2, -7)]
        """
        return AIN(-self.upper, -self.lower, -self.expected)

    def __add__(self, other):
        """
        Adds either another `AIN` instance or a value of type `int` or `float` to the current 
        `AIN` instance. Returns a new `AIN` instance representing the result.
        
        - When adding another `AIN` instance, the resulting `lower`, `upper`, and `expected` 
        values are calculated by summing the corresponding values of both `AIN` instances.
        - When adding a value of type `int` or `float`, the value is added to each component 
        (`lower`, `upper`, and `expected`) of the current AIN instance.

        Parameters
        ----------
        other : AIN, float, or int
            The value to be added, which can be `AIN` instance, a `float`, or an `int`.

        Returns
        -------
        AIN
            Returns a new `AIN` instance representing the result of the addition, with the 
            `lower`, `upper`, and `expected` values updated accordingly based on the operation.

        Raises
        ------
        TypeError
            If `other` is not an instance of `AIN`, `float`, or `int`.

        Examples
        --------
        Addition with another `AIN` instance:
        >>> a = AIN(1, 10, 8)
        >>> b = AIN(0, 5, 2)
        >>> print(a + b)
        [1.0000, 15.0000]_{10.0000}

        Addition with a `float` or `int`:
        >>> a = AIN(1, 10, 8)
        >>> b = 2
        >>> print(a + b)
        [3.0000, 12.0000]_{10.0000}

        Performing addition with a `np.array` of `AIN` instances:
        >>> a = np.array([AIN(0, 10), AIN(2, 8, 7)])
        >>> print(a + 2)
        [AIN(2, 12, 7.0) AIN(4, 10, 9)]
        
        """
        if not isinstance(other, (AIN, float, int)):
            raise TypeError(f"other is not an instance of AIN or float or int")
        if isinstance(other, AIN):
            new_a = self.lower + other.lower
            new_b = self.upper + other.upper
            new_c = self.expected + other.expected
            res = AIN(new_a, new_b, new_c)
        else:
            res = AIN(self.lower + other, self.upper + other, self.expected + other)
        return res

    def __radd__(self, other):
        """
        Perform reflected (reverse) addition for an AIN instance.
        
        This method handles the addition of an Asymmetric Interval Number (`AIN`) instance 
        to a value of type `float` or `int` when the AIN appears on the right-hand side 
        of the addition (i.e., `other + self`). 
        
        It computes `other + self`, ensuring commutative addition between `AIN` instances 
        and numeric values (`float` or `int`).

        Parameters
        ----------
        other : float or int
            The value to add to the current AIN instance.

        Returns
        -------
        AIN
            A new `AIN` instance representing the result of the addition, with `lower`, `upper`,
            and `expected` values equal to the sum of `other` and the corresponding values of
            the current `AIN` instance.

        Raises
        ------
        TypeError
            If `other` is not a float or int.

        Examples
        --------
        >>> a = AIN(1, 10, 8)
        >>> b = 5
        >>> print(b + a)
        [6.0000, 15.0000]_{13.0000}

        >>> a = np.array([AIN(0, 10), AIN(2, 8, 7)])
        >>> b = 2
        >>> print(2 + a)
        [AIN(2, 12, 7.0) AIN(4, 10, 9)]
        """
        if not isinstance(other, (float, int)):
            raise TypeError("other must be of type float or int")
        return self + other

    def __sub__(self, other):
        """
        Subtract an `AIN` instance or a `float` or `int` from the current `AIN` instance.

        This method allows subtraction of either another `AIN` or a `float` or `int` from 
        the current `AIN` instance, returning a new `AIN` instance with the result. When 
        subtracting another `AIN`, the resulting bounds and expected value are computed by 
        subtracting the corresponding values of the operands. If subtracting a `float` or 
        `int`, the value is subtracted from each component of the current `AIN` instance.

        Parameters
        ----------
        other : AIN, float, or int
            The value to subtract, which can be an `AIN` instance, a `float`, or an `int`.

        Returns
        -------
        AIN
            A new `AIN` instance representing the result of the subtraction, with adjusted
            `lower`, `upper`, and `expected` values based on the operation.

        Raises
        ------
        TypeError
            If `other` is not an instance of `AIN`, `float`, or `int`.

        Examples
        --------
        Subtracting an `AIN` instance:
        >>> a = AIN(1, 10, 8)
        >>> b = AIN(0, 5, 2)
        >>> print(a - b)
        [-4.0000, 10.0000]_{6.0000}

        Subtracting a `float` or `int`:
        >>> a = AIN(1, 10, 8)
        >>> b = 2
        >>> print(a - b)
        [-1.0000, 8.0000]_{6.0000}

        Performing subtraction with a `np.array` of `AIN` instances:
        >>> a = np.array([AIN(0, 10), AIN(2, 8, 7)])
        >>> b = AIN(0, 5, 4)
        >>> print(a - b)
        [AIN(-5, 10, 1.0) AIN(-3, 8, 3)]
        """
        if not isinstance(other, (AIN, float, int)):
            raise TypeError("other is not an instance of AIN or float or int")
        if isinstance(other, AIN):
            new_a = self.lower - other.upper
            new_b = self.upper - other.lower
            new_c = self.expected - other.expected
            res = AIN(new_a, new_b, new_c)
        else:
            res = AIN(self.lower - other, self.upper - other, self.expected - other)
        return res

    def __rsub__(self, other):
        """
        Perform reflected (reverse) subtraction for an `AIN` instance.

        This method is invoked when an `AIN` instance appears on the right-hand side of 
        a subtraction operation (i.e., `other - self`) and the left operand (`other`) does 
        not support subtraction with an `AIN`. It calculates the result of `other - self`.

        Parameters
        ----------
        other : float or int
            The value from which the current `AIN` instance is subtracted.

        Returns
        -------
        AIN
            A new `AIN` instance representing the result of the subtraction. The resulting
            `AIN` has its `lower`, `upper`, and `expected` values computed as the difference between
            `other` and the respective values of the `AIN` instance.

        Raises
        ------
        TypeError
            If `other` is not a `float` or `int`.

        Examples
        --------
        >>> a = AIN(1, 10, 8)
        >>> b = 5
        >>> print(b - a)
        [-5.0000, 4.0000]_{-3.0000}

        >>> a = np.array([AIN(0, 10), AIN(2, 8, 7)])
        >>> print(2 - a)
        [AIN(-8, 2, -3.0) AIN(-6, 0, -5)]
        """
        if not isinstance(other, (float, int)):
            raise TypeError("other must be of type float or int")
        return -self + other

    def __mul__(self, other):
        """
        Perform multiplication of the current `AIN` instance with another `AIN` or a `float` or `int`.

        This method allows the multiplication of an `AIN` instance with another `AIN` instance 
        or a `float` or `int`, returning a new `AIN` instance that represents the result. When 
        multiplying with another `AIN`, the interval boundaries are computed based on the 
        combinations of bounds from both `AIN` instances.

        Parameters
        ----------
        other : AIN, float, or int
            The value to multiply with, which can be another `AIN` instance, a `float`, or an `int`.

        Returns
        -------
        AIN
            A new `AIN` instance representing the product of the multiplication.

        Raises
        ------
        TypeError
            If `other` is not an instance of `AIN`, `float`, or `int`.

        Examples
        --------
        Multiplying with another `AIN` instance:
        >>> a = AIN(1, 3, 2)
        >>> b = AIN(2, 4, 3)
        >>> print(a * b)
        [2.0000, 12.0000]_{6.0000}

        Multiplying with a `float` or `int`:
        >>> a = AIN(1, 3, 2)
        >>> b = 2
        >>> print(a * b)
        [2.0000, 6.0000]_{4.0000}

        Performing multiplication with a `np.array` of `AIN` instances:
        >>> a = np.array([AIN(0, 10), AIN(2, 8, 7)])
        >>> b = AIN(1,4,2)
        >>> print(a * b)
        [AIN(0, 40, 10.0) AIN(2, 32, 14)]
        """
        if not isinstance(other, (int,float, AIN)):
            raise TypeError('other must be an instance of AIN or int or float')
        if isinstance(other, AIN):
            new_a = min(self.lower * other.lower, self.upper * other.upper, self.lower * other.upper, self.upper * other.lower)
            new_b = max(self.lower * other.lower, self.upper * other.upper, self.lower * other.upper, self.upper * other.lower)
            new_c = self.expected * other.expected
            res = AIN(new_a, new_b, new_c)
        else:
            res = AIN(self.lower * other, self.upper * other, self.expected * other)
        return res

    def __rmul__(self, other):
        """
        Perform reverse multiplication for an `AIN` instance with a `float` or `int`.

        This method allows an `AIN` instance to be multiplied by a `float` or `int` in 
        cases where the `float` or `int` appears on the left side of the multiplication 
        (i.e., `other * self`). This enables commutative multiplication between `AIN` 
        and `float` or int values.

        Parameters
        ----------
        other : float or int
            An `float` or `int` value to multiply with the `AIN` instance.

        Returns
        -------
        AIN
            A new `AIN` instance representing the result of the multiplication.

        Raises
        ------
        TypeError
            If `other` is not a `float` or `int`.

        Examples
        --------
        >>> a = AIN(1, 3, 2)
        >>> b = 2
        >>> result = b * a
        >>> print(result)
        [2.0000, 6.0000]_{4.0000}

        >>> a = np.array([AIN(0, 10), AIN(2, 8, 7)])
        >>> b = 2
        >>> print(b * a)
        [AIN(0, 20, 10.0) AIN(4, 16, 14)]
        """
        if not isinstance(other, (float, int)):
            raise TypeError("other must be float or int")
        return self * other

    def __truediv__(self, other):
        """
        Perform division of the current `AIN` instance by another `AIN` instance or a `float` or `int`.

        This method supports division by either another `AIN` or a `float` or `int`, returning a new 
        `AIN` instance as the result. When dividing by an `AIN`, interval boundaries are calculated 
        by dividing the respective boundaries, while the expected value is adjusted based on logarithmic 
        calculations if the bounds differ.

        Parameters
        ----------
        other : AIN, float, or int
            The divisor, which can be an `AIN` instance, a `float`, or an `int`.

        Returns
        -------
        AIN
            A new `AIN` instance representing the result of the division.

        Raises
        ------
        TypeError
            If `other` is not an instance of `AIN`, `float`, or `int`.

        Examples
        --------
        Division with another AIN instance:
        >>> a = AIN(4, 8, 6)
        >>> b = AIN(2, 4, 3)
        >>> print(a / b)
        [1.0000, 4.0000]_{2.0794}

        Division with a float or int:
        >>> a = AIN(4, 8, 6)
        >>> b = 2
        >>> print(a / b)
        [2.0000, 4.0000]_{3.0000}
        
        Performing division with a np.array of AIN instances:
        >>> a = np.array([AIN(0, 10), AIN(2, 8, 7)])
        >>> b = AIN(1, 4, 2)
        >>> print(a / b)
        [AIN(0.0, 10.0, 2.8881132523331052) AIN(0.5, 8.0, 4.043358553266348)]
        """
        if not isinstance(other, (AIN, int, float)):
            raise TypeError(f"other must be an instance of AIN or float or int")
        if isinstance(other, AIN):
            new_a = min(self.lower / other.lower, self.upper / other.upper, self.lower / other.upper, self.upper / other.lower)
            new_b = max(self.lower / other.lower, self.upper / other.upper, self.lower / other.upper, self.upper / other.lower)
            if other.lower == other.upper:
                new_c = (new_a + new_b) / 2
            else:
                new_c = self.expected * (other.alpha * np.log(other.expected / other.lower) + other.beta * np.log(other.upper / other.expected))
            res = AIN(new_a, new_b, new_c)
        else:
            res = AIN(self.lower / other, self.upper / other, self.expected / other)
        return res

    def __rtruediv__(self, other):
        """
        Perform reverse true division of a float or int by an `AIN` instance.

        This method enables division where a float or int `other` is divided by an AIN instance (`self`),
        calculating the reciprocal of `self` and then scaling it by `other`. It returns a new `AIN`
        instance representing the outcome.

        Parameters
        ----------
        other : float or int
            The `float` or `int` to divide by the `AIN` instance.

        Returns
        -------
        AIN
            A new `AIN` instance representing `other` divided by `self`.

        Raises
        ------
        TypeError
            If `other` is not a `float` or `int`.

        Examples
        --------
        >>> a = AIN(2, 4, 3)
        >>> result = 10 / a
        >>> print(result)
        [2.5000, 5.0000]_{3.4657}

        >>> a = np.array([AIN(2,10), AIN(2,8, 7)])
        >>> result = 2 / a
        >>> print(result)
        [AIN(0.2, 1.0, 0.4023594781085251) AIN(0.25, 1.0, 0.3060698522738955)]
        """
        if not isinstance(other, (float, int)):
            raise TypeError(f"other variable is not a float or int")
        return other * self**(-1)

    def __pow__(self, n):
        """
        Raise an `AIN` instance to the power `n`.

        This method computes the result of raising the AIN instance to the specified exponent `n`.

        Parameters
        ----------
        n : int or float
            The exponent to which the `AIN` is raised. Valid values include positive or negative real numbers.

        Raises
        ------
        TypeError
            If `n` is not a `float` or `int`.
        ValueError
            If the operation would result in a complex number (e.g., taking the square root of a negative value),
            or if `n = -1` and the interval includes 0, as division by zero is undefined.
    
        Returns
        -------
        AIN
            A new `AIN` instance representing the interval raised to the power of `n`.

        Notes
        -----
        - For `n = -1`, the method checks if 0 is within the interval. If it is, the operation is undefined
          (division by zero) and raises a `ValueError`.
        - When `n` results in a complex output (e.g., fractional exponents for negative values), a `ValueError`
          is raised to indicate that complex results are unsupported.
        - For other exponents, the power is applied individually to `self.lower`, `self.upper`, and `self.expected`,
          with appropriate handling for intervals containing 0 to avoid undefined behaviors.

        Examples
        --------
        >>> a = AIN(4, 8, 5)
        >>> print(a**2)
        [16.0000, 64.0000]_{26.0000}

        >>> b = AIN(-2, 10, 3)
        >>> print(b**(-1))
        Traceback (most recent call last):
        ...
        ValueError: The operation cannot be execute because 0 is included in the interval.

        >>> c = AIN(-2, 10, 3)
        >>> print(c**(0.5))
        Traceback (most recent call last):
        ...
        ValueError: The operation cannot be execute because it will be complex number in result for n = 0.5

        >>> a = np.array([AIN(0, 9), AIN(2, 8, 5)])
        >>> print(a ** 2)
        [AIN(0, 81, 27.0) AIN(4, 64, 28.0)]
        """
        if not isinstance(n, (float, int)):
            raise TypeError('n must be float or int')
        if isinstance(self.lower**n, complex):
            raise ValueError(f'The operation cannot be execute because it will be complex number in result for n = {n}')
        if self.lower < 0 and self.upper > 0:
            new_a = min(0, self.lower ** n)
        else:
            new_a = min(self.lower ** n, self.upper ** n)
        new_b = max(self.lower ** n, self.upper ** n)
        if n == -1:
            if self.lower <= 0 <= self.upper:
                raise ValueError(f'The operation cannot be execute because 0 is included in the interval.')
            else:
                if self.lower == self.upper:
                    new_c = 1 / self.lower
                else:
                    new_c = self.alpha * np.log(self.expected / self.lower) + self.beta * np.log(self.upper / self.expected)
        else:
            new_c = self.alpha * (self.expected ** (n + 1) - self.lower ** (n + 1)) / (n + 1) + self.beta * (
                self.upper ** (n + 1) - self.expected ** (n + 1)) / (n + 1)
        if self.lower == self.upper:
            new_c = new_b
        res = AIN(new_a, new_b, new_c)
        return res


    def pdf(self, x):
        """
        Calculate the probability density function (PDF) value for the `AIN` at a given point `x`.

        This method calculates the probability density at `x` within the AIN-defined interval. The PDF describes
        how the density is distributed across the AIN interval, with distinct values in different segments:
        - Outside the interval `[self.lower, self.upper]`, the density is 0.
        - Between `self.lower` and `self.expected`, the density is equal to `self.alpha`.
        - Between `self.expected` and `self.upper`, the density is equal to `self.beta`.

        Parameters
        ----------
        x : int or float
            The point at which to evaluate the PDF. Should be a numeric value.

        Returns
        -------
        float
            The PDF value at the specified point `x`. The return value will be:
            - 0 if `x` is outside the interval `[self.lower, self.upper]`.
            - `self.alpha` if `x` is within the interval `[self.lower, self.expected]`.
            - `self.beta` if `x` is within the interval `[self.expected, self.upper]`.

        Raises
        ------
        TypeError
            If `x` is not an `int` or `float`.

        Examples
        --------
        >>> a = AIN(0, 10, 5)
        >>> a.pdf(-1)
        0.0
        
        >>> a.pdf(3)
        0.1
        
        >>> a.pdf(7)
        0.1
        
        >>> a.pdf(11)
        0.0
        """
        if not isinstance(x, (int, float)):
            raise TypeError(f'Argument x must be an integer, not {type(x)}')

        if x < self.lower:
            return 0.0
        elif x < self.expected:
            return self.alpha
        elif x < self.upper:
            return self.beta
        else:
            return 0.0

    def cdf(self, x):
        """
        Calculate the cumulative distribution function (CDF) value for a specified input `x`.

        This method evaluates the cumulative distribution function (CDF) of the `AIN` instance at
        the given value `x`, indicating the probability that a random variable takes a value less
        than or equal to `x`. The CDF value is computed based on the position of `x` relative to
        the instance's defined bounds (`self.lower`, `self.expected`, `self.upper`).

        Parameters
        ----------
        x : int or float
            The point at which to evaluate the CDF. This should be a numeric value.

        Returns
        -------
        float
            The computed CDF value at `x`, representing the cumulative probability up to `x`.
            The output will follow these cases:
            - 0 if `x` is less than the lower bound (`self.lower`).
            - A linearly interpolated value between the lower bound and the expected value
              if `x` is between `self.lower` and `self.expected`.
            - A linearly interpolated value between the expected value and the upper bound
              if `x` is between `self.expected` and `self.upper`.
            - 1 if `x` is greater than or equal to the upper bound (`self.upper`).

        Raises
        ------
        TypeError
            If `x` is not an `int` or `float`.

        Notes
        -----
        This method calculates the CDF using a piecewise approach:
        - For `x < self.lower`, it returns 0.
        - For `self.lower <= x < self.expected`, the CDF is calculated as `self.alpha * (x - self.lower)`.
        - For `self.expected <= x < self.upper`, the CDF is calculated as
          `self.alpha * (self.expected - self.lower) + self.beta * (x - self.expected)`.
        - For `x >= self.upper`, it returns 1.

        Examples
        --------
        >>> a = AIN(0, 10, 3)
        >>> a.cdf(1.5)
        0.35
        
        >>> a.cdf(3)
        0.7
        
        >>> a.cdf(8.5)
        0.9357142857142857
        
        >>> a.cdf(20)
        1
        """
        if not isinstance(x, (int, float)):
            raise TypeError(f'x must be an int or float value.')
        if x < self.lower:
            res = 0
        elif x < self.expected:
            res = self.alpha * (x - self.lower)
        elif x < self.upper:
            res = self.alpha * (self.expected - self.lower) + self.beta * (x - self.expected)
        else:
            res = 1
        return res

    def quantile(self, y):
        """
        Compute the quantile value (inverse cumulative distribution function) for a given probability.

        This method calculates the quantile, or the inverse cumulative distribution function (CDF),
        for the `AIN` instance at a specified probability level `y`. The quantile represents the value
        below which a given percentage of observations fall, based on the AIN instance’s parameters.
        The function only operates within the probability range [0, 1].

        Parameters
        ----------
        y : int or float
            The probability level at which to compute the quantile. Must be within the range [0, 1],
            where 0 represents the minimum and 1 represents the maximum of the distribution.

        Returns
        -------
        float
            The quantile value corresponding to the given probability `y`.

        Raises
        ------
        ValueError
            If `y` is outside the valid range [0, 1].

        TypeError
            If `y` is not a `float` or `int` value.

        Notes
        -----
        The method uses `self.alpha`, `self.beta`, `self.expected`, and `self.lower` attributes
        to compute the quantile based on a piecewise formula:
        - For values of `y` below `self.alpha * (self.expected - self.lower)`, the quantile
          is calculated as `y / self.alpha + self.lower`.
        - Otherwise, it is calculated as `(y - self.alpha * (self.expected - self.lower)) / self.beta + self.expected`.

        Examples
        --------
        >>> a = AIN(0, 10, 3)
        >>> a.quantile(0.25)
        1.0714285714285714
        
        >>> a.quantile(0.85)
        6.5
        
        >>> a.quantile(1.1)
        Traceback (most recent call last):
            ...
        ValueError: Argument y = 1.1 is out of range; it should be between 0 and 1.
        """
        if not isinstance(y, (int, float)):
            raise TypeError(f'Argument y = {y} is not an integer or float.')
        if 0 <= y <= 1:
            if y < self.alpha * (self.expected - self.lower):
                res = y / self.alpha + self.lower
            else:
                res = (y - self.alpha * (self.expected - self.lower)) / self.beta + self.expected
        else:
            raise ValueError(f'Argument y = {y} is out of range; it should be between 0 and 1.')
        return res

    def summary(self, precision=6):
        """
        Print a detailed, aligned summary of the AIN object's key attributes with specified precision.

        This method displays a formatted summary of the AIN object's primary attributes, including
        `alpha`, `beta`, `asymmetry`, expected value, variance, standard deviation, and midpoint.
        Each attribute is displayed with the specified number of decimal places, allowing for a concise
        or detailed view.

        Parameters
        ----------
        precision : int, optional
            The number of decimal places to display for floating-point values (default is 6).
            Must be an integer; otherwise, a ValueError is raised.

        Raises
        ------
        TypeError
            If `precision` is not an `int`, a ValueError is raised with an informative message.

        Example
        -------
        >>> a = AIN(0, 10, 2)
        >>> a.summary(precision=4)
        === AIN ============================
        [0.0000, 10.0000]_{2.0000}
        === Summary ========================
        Alpha        =     0.4000
        Beta         =     0.0250
        Asymmetry    =     0.6000
        Exp. val.    =     2.0000
        Variance     =     5.3333
        Std. dev.    =     2.3094
        Midpoint     =     5.0000
        ====================================

        Notes
        -----
        This method ensures a clean, easy-to-read summary by aligning values based on the longest
        entry, making it particularly useful for inspecting the AIN object's main parameters in detail.
        """
        if not isinstance(precision, int):
            raise TypeError(f'Argument precision = {precision} but it must be an integer.')
        print("=== AIN ============================")
        print(self)
        print("=== Summary ========================")

        elements = [
            ('Alpha', f'{self.alpha:.{precision}f}'),
            ('Beta', f'{self.beta:.{precision}f}'),
            ('Asymmetry', f'{self.asymmetry:.{precision}f}'),
            ('Exp. val.', f'{self.expected:.{precision}f}'),
            ('Variance', f'{self.D2:.{precision}f}'),
            ('Std. dev.', f'{(self.D2 ** 0.5):.{precision}f}'),
            ('Midpoint', f'{((self.lower + self.upper) / 2):.{precision}f}')
        ]

        max_length = max(len(str(value)) for _, value in elements) + 4

        for name, value in elements:
            print(f'{name:<12} = {value:>{max_length}}')

        print("====================================")

    def midpoint(self):
        """
        Return the midpoint of the interval.

        Returns
        -------
        float
            The midpoint ((upper + lower) / 2).

        Examples
        --------
        >>> x = AIN(1, 10, 5)
        >>> print(x.midpoint())
        5.5
        """
        return (self.upper + self.lower) / 2

    def width(self):
        """
        Return the width of the interval.

        Returns
        -------
        float
            The width (upper - lower).

        Examples
        --------
        >>> x = AIN(1, 10, 5)
        >>> print(x.width())
        9
        """
        return self.upper - self.lower

    def radius(self):
        """
        Return the radius of the interval.

        Returns
        -------
        float
            The radius ((upper - lower) / 2).

        Examples
        --------
        >>> x = AIN(1, 10, 5)
        >>> print(x.radius())
        4.5
        """
        return (self.upper - self.lower) / 2

    def is_degenerate(self):
        """
        Check if the interval is degenerate (a single point).

        Returns
        -------
        bool
            True if lower == upper, False otherwise.

        Examples
        --------
        >>> x = AIN(5, 5, 5)
        >>> print(x.is_degenerate())
        True

        >>> x = AIN(1, 10, 5)
        >>> print(x.is_degenerate())
        False
        """
        return self.lower == self.upper

    def is_positive(self):
        """
        Check if the entire interval is positive.

        Returns
        -------
        bool
            True if lower > 0, False otherwise.

        Examples
        --------
        >>> x = AIN(1, 10, 5)
        >>> print(x.is_positive())
        True

        >>> x = AIN(-5, 10, 2)
        >>> print(x.is_positive())
        False
        """
        return self.lower > 0

    def is_negative(self):
        """
        Check if the entire interval is negative.

        Returns
        -------
        bool
            True if upper < 0, False otherwise.

        Examples
        --------
        >>> x = AIN(-10, -1, -5)
        >>> print(x.is_negative())
        True

        >>> x = AIN(-5, 10, 2)
        >>> print(x.is_negative())
        False
        """
        return self.upper < 0

    def is_zero(self):
        """
        Check if the interval represents exactly zero.

        Returns
        -------
        bool
            True if AIN(0, 0, 0), False otherwise.

        Examples
        --------
        >>> x = AIN(0, 0, 0)
        >>> print(x.is_zero())
        True

        >>> x = AIN(-1, 1, 0)
        >>> print(x.is_zero())
        False
        """
        return self.lower == 0 and self.upper == 0 and self.expected == 0

    def isclose_to_zero(self, atol=1e-9):
        """
        Check if the interval is close to zero within a specified absolute tolerance.

        Parameters
        ----------
        atol : float, optional
            The absolute tolerance level (default is 1e-9).

        Returns
        -------
        bool
            True if both lower and upper bounds are within the absolute tolerance of zero, False otherwise.

        Examples
        --------
        >>> x = AIN(-1e-10, 1e-10, 0)
        >>> print(x.isclose_to_zero())
        True

        >>> x = AIN(-1e-5, 1e-5, 0)
        >>> print(x.isclose_to_zero(atol=1e-6))
        False
        """
        return abs(self.lower) <= atol and abs(self.upper) <= atol

    def has_zero(self):
        """
        Check if the interval contains zero.

        Returns
        -------
        bool
            True if lower <= 0 <= upper, False otherwise.

        Examples
        --------
        >>> x = AIN(-5, 5, 0)
        >>> print(x.has_zero())
        True

        >>> x = AIN(1, 10, 5)
        >>> print(x.has_zero())
        False
        """
        return self.lower <= 0 <= self.upper

    def is_symmetric(self):
        """
        Check if the interval is symmetric (expected at midpoint).

        Returns
        -------
        bool
            True if expected is at the midpoint (asymmetry ≈ 0), False otherwise.

        Examples
        --------
        >>> x = AIN(0, 10, 5)
        >>> print(x.is_symmetric())
        True

        >>> x = AIN(0, 10, 3)
        >>> print(x.is_symmetric())
        False
        """
        return np.isclose(self.asymmetry, 0, atol=1e-9)

    def overlaps(self, other):
        """
        Check if two intervals overlap.

        Parameters
        ----------
        other : AIN
            Another AIN instance.

        Returns
        -------
        bool
            True if intervals overlap, False otherwise.

        Examples
        --------
        >>> x = AIN(0, 10, 5)
        >>> y = AIN(5, 15, 10)
        >>> print(x.overlaps(y))
        True

        >>> x = AIN(0, 5, 2)
        >>> y = AIN(10, 15, 12)
        >>> print(x.overlaps(y))
        False
        """
        if not isinstance(other, AIN):
            raise TypeError("other must be an AIN instance")

        return not (self.upper < other.lower or self.lower > other.upper)

    def is_subset_of(self, other):
        """
        Check if this interval is a subset of another interval.

        An interval A is a subset of B if A is completely contained within B:
        B.lower <= A.lower and A.upper <= B.upper

        Parameters
        ----------
        other : AIN
            Another AIN instance.

        Returns
        -------
        bool
            True if self is a subset of other, False otherwise.

        Examples
        --------
        >>> x = AIN(2, 8, 5)
        >>> y = AIN(0, 10, 5)
        >>> print(x.is_subset_of(y))
        True

        >>> x = AIN(0, 10, 5)
        >>> y = AIN(2, 8, 5)
        >>> print(x.is_subset_of(y))
        False
        """
        if not isinstance(other, AIN):
            raise TypeError("other must be an AIN instance")

        return other.lower <= self.lower and self.upper <= other.upper

    def is_disjoint(self, other):
        """
        Check if two intervals are disjoint (do not overlap).

        Two intervals are disjoint if they have no common points.

        Parameters
        ----------
        other : AIN
            Another AIN instance.

        Returns
        -------
        bool
            True if intervals are disjoint, False otherwise.

        Examples
        --------
        >>> x = AIN(0, 5, 2)
        >>> y = AIN(10, 15, 12)
        >>> print(x.is_disjoint(y))
        True

        >>> x = AIN(0, 10, 5)
        >>> y = AIN(5, 15, 10)
        >>> print(x.is_disjoint(y))
        False
        """
        if not isinstance(other, AIN):
            raise TypeError("other must be an AIN instance")

        return not self.overlaps(other)

    def contains(self, value):
        """
        Check if a value is contained in the interval.

        Parameters
        ----------
        value : float or int
            The value to check.

        Returns
        -------
        bool
            True if lower <= value <= upper, False otherwise.

        Examples
        --------
        >>> x = AIN(0, 10, 5)
        >>> print(x.contains(5))
        True
        >>> print(x.contains(15))
        False
        """
        if not isinstance(value, (int, float)):
            raise TypeError("value must be a number")

        return self.lower <= value <= self.upper

    def winsorize(self, min_val, max_val):
        """
        The winsorize operator returns an AIN whose expected value equals the expectation of the clipped random variable.
        Clip the interval to [min_val, max_val].

        Parameters
        ----------
        min_val : float
            Minimum value.
        max_val : float
            Maximum value.

        Returns
        -------
        AIN
            A new AIN instance with values clamped to [min_val, max_val].

        Raises
        ------
        ValueError
            If min_val >= max_val.

        Examples
        --------
        >>> x = AIN(-5, 10, 2)
        >>> result = x.winsorize(0, 5)
        >>> print(result)
        [0.0000, 5.0000]_{2.2232}

        Notes
        -----
        Due to the nonlinear saturation introduced by the clipping operator, the expected value of the winsorized
        AIN is a non-monotonic function of the characteristic parameter, as directly implied by the LOTUS formulation.
        """
        if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
            raise TypeError("min_val and max_val must be numbers")
        if min_val >= max_val:
            raise ValueError("min_val must be less than max_val")

        # Clip the bounds
        new_lower = max(min_val, min(max_val, self.lower))
        new_upper = max(min_val, min(max_val, self.upper))

        # Compute expected value
        # This is complex because clip is piecewise
        if self.upper <= min_val:
            # Entire interval below min
            new_expected = min_val
        elif self.lower >= max_val:
            # Entire interval above max
            new_expected = max_val
        elif self.lower >= min_val and self.upper <= max_val:
            # Entire interval within bounds
            new_expected = self.expected
        else:
            # Interval spans clip boundaries - use LOTUS
            # We need to integrate piecewise
            total_expected = 0

            # Part 1: values cliped to min_val
            if self.lower < min_val:
                if self.expected < min_val:
                    prob_below = self.alpha * (self.expected - self.lower)
                    total_expected += min_val * prob_below
                else:
                    prob_below = self.alpha * (min_val - self.lower)
                    total_expected += min_val * prob_below

            # Part 2: values in [min_val, max_val]
            if self.lower < max_val and self.upper > min_val:
                a = max(self.lower, min_val)
                b = min(self.upper, max_val)
                c = self.expected

                if a <= c <= b:
                    # Expected is in unclamped region
                    expected_unclamped = (self.alpha * (c ** 2 / 2 - a ** 2 / 2) +
                                          self.beta * (b ** 2 / 2 - c ** 2 / 2))
                elif c < a:
                    # Expected is below - use beta distribution
                    expected_unclamped = self.beta * (b ** 2 / 2 - a ** 2 / 2)
                else:
                    # Expected is above - use alpha distribution
                    expected_unclamped = self.alpha * (b ** 2 / 2 - a ** 2 / 2)

                total_expected += expected_unclamped

            # Part 3: values clamped to max_val
            if self.upper > max_val:
                if self.expected > max_val:
                    prob_above = self.beta * (self.upper - self.expected)
                    total_expected += max_val * prob_above
                else:
                    prob_above = self.beta * (self.upper - max_val)
                    total_expected += max_val * prob_above

            new_expected = total_expected

        return AIN(new_lower, new_upper, new_expected)

    def copy(self):
        # WS_to_check_common_sense
        """
        Create a deep copy of the AIN instance.

        Returns
        -------
        AIN
            A new AIN instance with the same values.

        Examples
        --------
        >>> x = AIN(0, 10, 5)
        >>> y = x.copy()
        >>> print(y)
        [0.0000, 10.0000]_{5.0000}
        """
        return AIN(self.lower, self.upper, self.expected)


    def plot(self, ain_lw=2.0, ain_c='k', ain_label=''):
        """
        Plot the intervals and key values of an `AIN` instance.

        Visualizes the `AIN` instance by plotting its lower, expected, and upper values,
        along with corresponding alpha and beta levels.

        Parameters
        ----------
        ain_lw : float, optional
            Line width for the alpha and beta lines. Must be a positive float or integer. Default is 2.0.
        ain_c : str, optional
            Color for the interval lines. Accepts any valid matplotlib color string. Default is 'k' (black).
        ain_label : str, optional
            Label for the x-axis, representing the `AIN` instance. Default is an empty string.

        Returns
        -------
        matplotlib.axes.Axes
            The Axes object containing the plot.

        Raises
        ------
        ValueError
            If `ain_lw` is not a positive `float` or `int`.
        TypeError
            If `ain_c` or `ain_label` is not a string.

        Examples
        --------
        >>> # Uncomment to show this functionality
        >>> # ain = AIN(1, 10, 3)
        >>> # ain.plot(ain_label='Example')
        >>> # plt.show()

        Notes
        -----
        - Vertical dashed lines are placed at the lower, expected, and upper interval bounds.
        - Horizontal solid lines represent the alpha level between the lower and expected values,
          and the beta level between the expected and upper values.
        - Y-axis limits are automatically adjusted based on the maximum of alpha and beta, while
          the x-axis extends slightly beyond the interval bounds for readability.
        - The default y-axis label is set to 'pdf', and the x-axis label displays `ain_label`.
        """
        if not isinstance(ain_lw, (float, int)) or ain_lw <= 0:
            raise ValueError("ain_lw must be a positive float or integer.")

        if not isinstance(ain_c, str):
            raise TypeError("ain_c must be a string representing a valid matplotlib color.")

        if not isinstance(ain_label, str):
            raise TypeError("ain_label must be a string.")

        a, b, c = self.lower, self.upper, self.expected

        alpha, beta = self.alpha, self.beta

        vkw = dict(ls='--', lw=ain_lw/2, c='k')

        ax = plt.gca()

        ax.plot([a, a], [0, alpha], **vkw)
        ax.plot([c, c], [0, max(alpha, beta)], **vkw)
        ax.plot([b, b], [0, beta], **vkw)

        hkw = dict(ls='-', lw=ain_lw, c=ain_c)
        ax.plot([a, c], [alpha, alpha], **hkw)
        ax.plot([c, b], [beta, beta], **hkw)

        ax.set_ylim([0, max(alpha, beta) * 1.1])
        ax.set_yticks(sorted([alpha, beta]))
        yticklabels = [f"{alpha:.4f}", f"{beta:.4f}"]
        if alpha > beta:
            yticklabels.reverse()
        ax.set_yticklabels(yticklabels, fontsize=12)

        if a == b == c:
            after_a = a - 0.5  # Arbitrary small extension for visual clarity
            after_b = b + 0.5
            ax.plot(a, 1, 'ko', markersize=3)
        else:
            after_a = a - (b - a) * 0.05
            after_b = b + (b - a) * 0.05
        ax.set_xlim([after_a, after_b])

        ax.set_xticks([a, c, b])
        ax.set_xticklabels([f"{a:.4f}", f"{c:.4f}", f"{b:.4f}"], fontsize=12)

        ax.spines[["top", "right"]].set_visible(False)

        ax.plot(1, 0, ">k", transform=plt.gca().get_yaxis_transform(), clip_on=False)
        ax.plot(after_a, 1, "^k", transform=plt.gca().get_xaxis_transform(), clip_on=False)
        ax.set_ylabel('$pdf$')
        ax.set_ylabel('pdf',labelpad=-15)
        ax.set_xlabel(ain_label)
        return ax


    @staticmethod
    def get_y_scale_max(ains_list):
        """
        Calculate the maximum scale value (y-axis) from a list of `AIN` objects.

        Parameters
        ----------
        ains_list : list
            A list of `AIN` objects.

        Returns
        -------
        float
            The maximum scale value found in the list of `AIN` objects.

        Raises
        ------
        TypeError
            If ains_list is not a list or if any element in the list is not an `AIN` object.

        Notes
        -----
        This function computes the maximum of the alpha and beta values across all AIN objects
        in the provided list to determine the maximum scale value on the y-axis.

        Example
        -------
        Assuming `ains_list` is a list of `AIN` objects:

        >>> ains_list = [AIN(1, 10), AIN(2, 10, 4)]
        >>> max_value = AIN.get_y_scale_max(ains_list)
        >>> print(max_value)
        0.375
        """
        result = 0
        if not isinstance(ains_list, list):
            raise TypeError("ains_list should be a list")
        for el in ains_list:
            if not isinstance(el, AIN):
                raise TypeError("Each element in the list must be a AIN object")
            result = max(result, el.alpha, el.beta)
        return result

    def add_to_plot(self, ain_lw=2.0, ain_c='k', ain_label='', ax=None, y_scale_max=None):
        """
        Plot the intervals and key values of an `AIN` instance.

        Visualizes the `AIN` instance by plotting its `lower`, `expected`, and `upper` values,
        along with corresponding `alpha` and `beta` levels. The plot includes:

        - Vertical dashed lines at the lower, expected, and upper bounds of the interval.
        - Horizontal solid lines representing the alpha and beta values across the intervals.
        - Dynamically scales the x- and y-axes for clarity, with an optional global maximum
          for y-axis scaling.

        Parameters
        ----------
        ain_lw : float, optional
            Line width for the alpha and beta interval lines. Default is 2.0.
        ain_c : str, optional
            Color for the interval lines. Default is 'k' (black).
        ain_label : str, optional
            Label for the x-axis describing the plotted AIN instance. Default is ''.
        ax : matplotlib.axes.Axes, optional
            Matplotlib axis to add the plot to. If not provided, the current axis (`plt.gca()`) is used.
        y_scale_max : float or int, optional
            Maximum value for the y-axis to ensure consistent scaling across multiple AIN plots.
            If not provided, the y-axis is scaled to 1.1 times the maximum of alpha or beta for this AIN instance.

        Returns
        -------
        matplotlib.axes.Axes
            The matplotlib axis with the AIN plot.

        Raises
        ------
        ValueError
            If `ain_lw` is non-positive or if `y_scale_max` is negative.
        TypeError
            If `ain_lw` or `y_scale_max` are not numeric, or if `ain_c` or `ain_label` are not strings.

        Examples
        --------
        >>> # Uncomment to show this functionality
        >>> # ain = AIN(1, 10, 5)
        >>> # ain.add_to_plot(ain_label='Example Interval')
        >>> # plt.show()
        >>> # a = AIN(0, 10, 4.5)
        >>> # b = AIN(0, 10, 7.5)
        >>> # value_y_scale_max = AIN.get_y_scale_max([a, b])
        >>> # plt.figure(figsize=(8, 3))
        >>> # plt.subplot(1, 2, 1)
        >>> # a.add_to_plot(y_scale_max=value_y_scale_max)
        >>> # plt.subplot(1, 2, 2)
        >>> # b.add_to_plot(y_scale_max=value_y_scale_max)
        >>> # plt.tight_layout()
        >>> # plt.show() # Uncomment to display the plot

        Notes
        -----
        - Vertical dashed lines are positioned at the lower, expected, and upper bounds of the interval.
        - Horizontal solid lines represent the alpha level between the lower and expected values,
          and the beta level between the expected and upper values.
        - The y-axis limits are automatically adjusted based on the maximum of alpha and beta values unless
          `y_scale_max` is specified. The x-axis extends slightly beyond the interval bounds for readability.
        - The default y-axis label is set to '$pdf$', and the x-axis label is set to `ain_label`.
        """
        if not isinstance(ain_lw, (float, int)) or ain_lw <= 0:
            raise ValueError("ain_lw must be a positive float or integer.")

        if not isinstance(ain_c, str):
            raise TypeError("ain_c must be a string representing a valid matplotlib color.")

        if not isinstance(ain_label, str):
            raise TypeError("ain_label must be a string.")

        if ax is None:
            ax = plt.gca()

        a, b, c = self.lower, self.upper, self.expected

        alpha, beta = self.alpha, self.beta

        vkw = dict(ls='--', lw=ain_lw / 2, c='k')
        ax.plot([a, a], [0, alpha], **vkw)
        ax.plot([c, c], [0, max(alpha, beta)], **vkw)
        ax.plot([b, b], [0, beta], **vkw)

        hkw = dict(ls='-', lw=ain_lw, c=ain_c)
        ax.plot([a, c], [alpha, alpha], **hkw)
        ax.plot([c, b], [beta, beta], **hkw)

        if y_scale_max is None:
            ax.set_ylim([0, max(alpha, beta) * 1.1])
        else:
            if not isinstance(y_scale_max, (int, float)):
                raise TypeError("y_scale_max must be a float or integer")
            if y_scale_max < 0:
                raise ValueError("y_scale_max must be a positive value")
            ax.set_ylim([0, y_scale_max * 1.1])
        ax.set_yticks(sorted([alpha, beta]))
        yticklabels = [f"{alpha:.4f}", f"{beta:.4f}"]
        if alpha > beta:
            yticklabels.reverse()
        ax.set_yticklabels(yticklabels, fontsize=12)

        if a == b == c:
            after_a = a - 0.5  # Arbitrary small extension for visual clarity
            after_b = b + 0.5
            ax.plot(a, 1, 'ko', markersize=3)
        else:
            after_a = a - (b - a) * 0.05
            after_b = b + (b - a) * 0.05
        ax.set_xlim([after_a, after_b])

        ax.set_xticks([a, c, b])
        ax.set_xticklabels([f"{a:.4f}", f"{c:.4f}", f"{b:.4f}"], fontsize=12)

        ax.spines[["top", "right"]].set_visible(False)

        ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot(after_a, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
        ax.set_ylabel('$pdf$')
        ax.set_ylabel('pdf', labelpad=-15)
        ax.set_xlabel(ain_label)
        return ax

    def __abs__(self):
        """
        Compute the absolute value of an AIN instance.

        The absolute value operation handles three cases based on the position of the interval
        relative to zero. This implementation follows the LOTUS (Law of the Unconscious Statistician)
        methodology to compute the expected value of |X|.

        Returns
        -------
        AIN
            A new AIN instance representing the absolute value of the interval.

        Raises
        ------
        None

        Notes
        -----
        The method handles three distinct cases:

        **Case 1:** If the entire interval is non-negative (a ≥ 0), the absolute value
        does not change the interval: |[a, b]_c| = [a, b]_c

        **Case 2:** If the entire interval is non-positive (b ≤ 0), the absolute value
        negates and reverses the bounds: |[a, b]_c| = [-b, -a]_{-c}

        **Case 3:** If the interval contains zero (a < 0 < b), the absolute value results
        in an interval starting at zero. The expected value is computed using LOTUS:

        For c > 0 (expected value in positive part):
            E(|X|) = α(c²/2 - a²/2) + β(b²/2 - c²/2)

        For c ≤ 0 (expected value in negative part):
            E(|X|) = α(0 - a²/2) + β(b²/2 - 0)

        The upper bound becomes max(-a, b) to capture the maximum absolute value.

        Examples
        --------
        Case 1: Non-negative interval
        >>> a = AIN(1, 4, 2)
        >>> print(abs(a))
        [1.0000, 4.0000]_{2.0000}

        Case 2: Non-positive interval
        >>> b = AIN(-4, -1, -2)
        >>> print(abs(b))
        [1.0000, 4.0000]_{2.0000}

        Case 3: Interval containing zero (c > 0)
        >>> c = AIN(-2, 3, 1)
        >>> result = abs(c)
        >>> print(result)
        [0.0000, 3.0000]_{1.5333}

        Case 3: Interval containing zero (c ≤ 0)
        >>> d = AIN(-3, 2, -1)
        >>> result = abs(d)
        >>> print(result)
        [0.0000, 3.0000]_{1.5333}

        Symmetric interval around zero
        >>> e = AIN(-2, 2, 0)
        >>> result = abs(e)
        >>> print(result)
        [0.0000, 2.0000]_{1.0000}
        """
        # Case 1: Interval is non-negative (a ≥ 0)
        if self.lower >= 0:
            return AIN(self.lower, self.upper, self.expected)

        # Case 2: Interval is non-positive (b ≤ 0)
        elif self.upper <= 0:
            return AIN(-self.upper, -self.lower, -self.expected)

        # Case 3: Interval contains zero (a < 0 < b)
        else:
            new_a = 0
            new_b = max(-self.lower, self.upper)

            # Degenerate case
            if self.lower == self.upper:
                new_c = abs(self.expected)
            else:
                # Expected value is in the positive part (c > 0)
                # E(|X|) = α∫_a^0(-x)dx + α∫_0^c(x)dx + β∫_c^b(x)dx
                #        = α·a²/2 + α·c²/2 + β·b²/2 - β·c²/2
                if self.expected > 0:
                    new_c = (self.alpha * self.lower ** 2 / 2 +
                             self.alpha * self.expected ** 2 / 2 +
                             self.beta * self.upper ** 2 / 2 -
                             self.beta * self.expected ** 2 / 2)
                # Expected value is in the negative part or at zero (c ≤ 0)
                # E(|X|) = α∫_a^c(-x)dx + β∫_c^0(-x)dx + β∫_0^b(x)dx
                #        = α·a²/2 - α·c²/2 + β·c²/2 + β·b²/2
                else:
                    new_c = (self.alpha * self.lower ** 2 / 2 -
                             self.alpha * self.expected ** 2 / 2 +
                             self.beta * self.expected ** 2 / 2 +
                             self.beta * self.upper ** 2 / 2)

            return AIN(new_a, new_b, new_c)

    def __rpow__(self, a):
        """
        Compute a^x where `a` is the base and `self` (x) is the `AIN` instance.

        Allows expressions like `2 ** AIN(1, 2, 1.5)`.

        Parameters
        ----------
        a : float or int
            The base of the power function. Must be positive and not equal to 1.

        Returns
        -------
        AIN
            A new `AIN` instance representing the result of a^x operation.

        Raises
        ------
        TypeError
            If `a` is not a number (int or float).
        ValueError
            If `a <= 0` or `a == 1`.

        Examples
        --------
        >>> a = AIN(1, 2, 1.5)
        >>> print(2 ** a)
        [2.0000, 4.0000]_{2.8854}
        """


        # Usage of np.array
        if isinstance(self, np.ndarray):
            return np.array([a ** item for item in self])

        if not isinstance(a, (int, float)):
            raise TypeError(f"a is not a number (int or float)")

        if a <= 0 or a == 1:
            raise ValueError(f"a must be positive and not equal to 1")

        new_lower = float(a ** self.lower)
        new_upper = float(a ** self.upper)

        if self.lower == self.upper:
            new_expected = float(a ** self.expected)
        else:
            # We reverse the subtraction so that the difference is positive.
            new_expected = float((self.alpha * ((a ** self.expected) - (a ** self.lower)) +
                                  self.beta * ((a ** self.upper) - (a ** self.expected))) / np.log(a))

        res = AIN(new_lower, new_upper, new_expected)
        return res

    def __gt__(self, other):
        # Check validity of AINs
        """
        Compute the probability P(X > Y) where X and Y are AIN instances.

        Parameters:
        -----------
        other : AIN, int, float
            Another AIN instance or a numeric value to compare with.

        Returns:
        --------
        float
        Probability P(X > Y)
        Raises:
        -------
        TypeError
            If `ain` is not an instance of `AIN`.
        Examples:
        ---------
        >>> a = AIN(0, 10, 5)
        >>> b = AIN(4, 14, 9)
        >>> print(f"{a > b:.4f}")  # Compute P(a > b)
        0.1800
        >>> a > [1.00]  # Compare with a non-AIN object
        Traceback (most recent call last):
        ...
        TypeError: other is not an instance of AIN, int or float
        >>> a > 1.00
        0.9
        """
        def pos_part(x):
            """Positive part function: (x)_+ = max(0, x)"""
            return max(0, x)

        def integral_r_minus_max_y_s(p, q, r, s):
            """
            Compute the integral: (r - max(y, s))_+ dy

            Parameters:
            -----------
            p : float
                Lower integration limit
            q : float
                Upper integration limit
            r : float
                Constant in the integrand
            s : float
                Threshold in max(y, s)

            Returns:
            --------
            float
                Value of the integral
            """
            # Case 1: r <= p or p >= q
            if r <= p or p >= q:
                return 0.0

            # Case 2: s <= p < q <= r
            if s <= p < q <= r:
                return (r - p) * (q - p) - 0.5 * (q - p) ** 2

            # Case 3: s <= p < r < q
            if s <= p < r < q:
                return 0.5 * (r - p) ** 2

            # Case 4: s >= q
            if s >= q:
                return pos_part(r - s) * (q - p)

            # Case 5: p < s < q (need to compute K)
            if p < s < q:
                # Compute K based on subcases
                if r >= q:
                    K = (r - s) * (q - s) - 0.5 * (q - s) ** 2
                elif s < r < q:
                    K = 0.5 * (r - s) ** 2
                else:  # r <= s
                    K = 0.0

                return pos_part(r - s) * (s - p) + K

            # Default case (should not reach here if all cases are covered)
            raise ValueError(f"Uncovered case: p={p}, q={q}, r={r}, s={s}")

        if not isinstance(other, (AIN, int, float)):
            raise TypeError(f"other is not an instance of AIN, int or float")
        if isinstance(other, (int, float)):
            return 1 - self.cdf(other)

        a, b, c = self.lower, self.upper, self.expected
        d, e, f = other.lower, other.upper, other.expected

        # Case 1: Complete separation (X always less than Y)
        if b <= d:
            return 0.0

        # Case 2: Complete separation (X always greater than Y)
        if e <= a:
            return 1.0

        # Case 3: Overlapping case - compute using explicit formula
        # Compute density parameters
        alpha, beta = self.alpha, self.beta
        gamma, omega = other.alpha, other.beta

        # Four integral terms from the explicit formula
        I1 = gamma * alpha * integral_r_minus_max_y_s(d, min(f, c), c, a)
        I2 = gamma * beta * integral_r_minus_max_y_s(d, min(f, b), b, c)
        I3 = omega * alpha * integral_r_minus_max_y_s(f, min(e, c), c, a)
        I4 = omega * beta * integral_r_minus_max_y_s(f, min(e, b), b, c)

        return I1 + I2 + I3 + I4

    def __ge__(self, other):
        """
        Calculates the probability (between 0 and 1) that this AIN instance is greater than or equal to another object, which can be an AIN instance, a float, or an int.

        Parameters
        ----------
        other : AIN, float, or int
            The object to compare with this AIN instance.

        Returns
        -------
        float
            A probability between 0 and 1 representing the likelihood that this AIN instance is greater than or equal to `other`.

        Raises
        ------
        TypeError
            If `other` is not an instance of AIN, float, or int.

        Notes
        -----
        - If `other` is a float or int, the probability is computed as `1 - self.cdf(other) + self == other`, where:
            - `1 - self.cdf(other)` represents the probability that this AIN instance is greater than `other`.
            - `self == other` adds the probability of equality.
        - If `other` is an AIN instance, the probability is calculated by combining the results of `self > other` and `self == other`, yielding the total probability that this AIN instance is greater than or equal to the `other` AIN instance.

        Examples
        --------
        >>> a = AIN(2, 10, 5)
        >>> b = AIN(4, 12, 6)
        >>> print(f"{a >= b:.5f}")
        0.33125

        >>> a = AIN(3, 7, 5)
        >>> a >= 6
        0.25
        """
        if not isinstance(other, (AIN, float, int)):
            raise TypeError("other must be an instance of AIN, float, or int")
        return max(self > other, self == other)

    def __lt__(self, other):
        """
        Calculates the probability (between 0 and 1) that this AIN instance is less than another object, which can be an AIN instance, a float, or an int.

        Parameters
        ----------
        other : AIN, float, or int
            The object to compare with this AIN instance.

        Returns
        -------
        float
            A probability between 0 and 1 representing the likelihood that this AIN instance is less than `other`.

        Raises
        ------
        TypeError
            If `other` is not an instance of AIN, float, or int.

        Notes
        -----
        - If `other` is a float or int, the method uses `self.cdf(other)`, which returns the cumulative distribution function (CDF) value representing the probability that this AIN instance is less than `other`.
        - If `other` is an AIN instance, the method calculates the probability by checking if `other > self`, which should be defined to return the probability that `other` is greater than `self`.

        Examples
        --------
        >>> a = AIN(1, 8, 4)
        >>> b = AIN(5, 12, 6)
        >>> a < b
        0.7653061224489796

        >>> a = AIN(3, 7, 5)
        >>> a < 6
        0.75
        """
        if not isinstance(other, (int, float, AIN)):
            raise TypeError('other must be an integer, float or AIN')
        if isinstance(other, (int, float)):
            return self.cdf(other)
        if other.is_degenerate():
            return self.cdf(other.expected)
        return other > self

    def __le__(self, other):
        """
        Calculates the probability (between 0 and 1) that this AIN instance is less than or equal to another object, which can be an AIN instance, a float, or an int.

        Parameters
        ----------
        other : AIN, float, or int
            The object to compare with this AIN instance.

        Returns
        -------
        float
            A probability between 0 and 1 representing the likelihood that this AIN instance is less than or equal to `other`.

        Raises
        ------
        TypeError
            If `other` is not an instance of AIN, float, or int.

        Notes
        -----
        - If `other` is a float or int, the probability is calculated using `self.cdf(other) + self == other`, where:
            - `self.cdf(other)` represents the probability that this AIN instance is less than `other`.
            - `self == other` adds the probability of equality.
        - If `other` is an AIN instance, the probability is computed by combining the results of `self < other` and `self == other`, representing the total likelihood that this AIN instance is less than or equal to the `other` AIN instance.

        Examples
        --------
        >>> a = AIN(1, 8, 4)
        >>> b = AIN(5, 12, 6)
        >>> print(f"{a <= b:.4f}")
        0.7653

        >>> a = AIN(3, 7, 5)
        >>> a <= 6
        0.75
        """
        if not isinstance(other, (AIN, float, int)):
            raise TypeError("other must be an instance of AIN, float, or int")
        return max(self < other, self == other)


    def log(self):
        """
        Computes the natural logarithm (ln(x)) of the current `AIN` instance.
        Returns a new `AIN` instance representing the result.

        - When computing ln(x) of an `AIN` instance, the resulting `lower` and `upper`
        values are calculated by applying the natural logarithm function to the
        corresponding bounds of the current `AIN` instance.
        - The `expected` value is calculated using the formula:
          c_ln = α(c·ln(c) - c - a·ln(a) + a) + β(b·ln(b) - b - c·ln(c) + c)
          where a = lower, b = upper, c = expected, α = alpha, β = beta

        Parameters
        ----------
        None

        Returns
        -------
        AIN
            Returns a new `AIN` instance representing the result of the natural logarithm operation,
            with the `lower`, `upper`, and `expected` values updated accordingly based
            on the operation.

        Raises
        ------
        TypeError
            If `self` is not an instance of `AIN`.
        ValueError
            If `lower <= 0`, as the natural logarithm is undefined for non-positive values.

        Examples
        --------
        Natural logarithm of an `AIN` instance:
        >>> a = AIN(1, np.e, 2)
        >>> print(a.log())
        [0.0000, 1.0000]_{0.6587}
        """

        if not isinstance(self, AIN):
            raise TypeError(f"self is not an instance of AIN")

        if self.lower <= 0:
            raise ValueError(
                f"lower must be positive (> 0), got lower={self.lower}. Natural logarithm is undefined for non-positive values.")

        new_lower = np.log(self.lower)
        new_upper = np.log(self.upper)

        new_expected = (self.alpha * (self.expected * np.log(self.expected) - self.expected -
                                      self.lower * np.log(self.lower) + self.lower) +
                        self.beta * (self.upper * np.log(self.upper) - self.upper -
                                     self.expected * np.log(self.expected) + self.expected))

        res = AIN(new_lower, new_upper, new_expected)
        return res

    def log2(self):
        """
        Compute base-2 logarithm of an AIN instance.

        Returns
        -------
        AIN
            A new AIN instance representing log2(X).

        Raises
        ------
        ValueError
            If lower bound is non-positive.

        Examples
        --------
        >>> x = AIN(1, 8, 4)
        >>> result = x.log2()
        >>> print(result)
        [0.0000, 3.0000]_{1.7954}
        """
        if self.lower <= 0:
            raise ValueError("log2 requires positive values")

        # log2(x) = ln(x) / ln(2)
        result_ln = self.log()
        return AIN(result_ln.lower / np.log(2),
                   result_ln.upper / np.log(2),
                   result_ln.expected / np.log(2))

    def log10(self):
        """
        Compute base-10 logarithm of an AIN instance.

        Returns
        -------
        AIN
            A new AIN instance representing log10(X).

        Raises
        ------
        ValueError
            If lower bound is non-positive.

        Examples
        --------
        >>> x = AIN(1, 100, 10)
        >>> result = x.log10()
        >>> print(result)
        [0.0000, 2.0000]_{0.7677}
        """
        if self.lower <= 0:
            raise ValueError("log10 requires positive values")

        # log10(x) = ln(x) / ln(10)
        result_ln = self.log()
        return AIN(result_ln.lower / np.log(10),
                   result_ln.upper / np.log(10),
                   result_ln.expected / np.log(10))


    def exp(self):
        """
        Computes the exponential (e^x) of the current `AIN` instance.
        Returns a new `AIN` instance representing the result.

        - When computing exp() of an `AIN` instance, the resulting `lower` and `upper`
        values are calculated by applying the exponential function to the corresponding values.
        - The `expected` value is calculated as the mean value of exp(x) over the interval
        [lower, upper], given by (e^upper - e^lower) / (upper - lower).

        Parameters
        ----------
        None

        Returns
        -------
        AIN
            Returns a new `AIN` instance representing the result of the exponential operation,
            with the `lower`, `upper`, and `expected` values updated accordingly.

        Raises
        ------
        TypeError
            If `self` is not an instance of `AIN`.
        ValueError
            If `lower >= upper`.

        Examples
        --------
        Exponential of an `AIN` instance:
        >>> a = AIN(0, 1, 0.5)
        >>> print(a.exp())
        [1.0000, 2.7183]_{1.7183}
        """

        if not isinstance(self, AIN):
            raise TypeError(f"self is not an instance of AIN")

        if self.lower >= self.upper:
            raise ValueError(f"lower ({self.lower}) must be less than upper ({self.upper})")

        new_lower = np.exp(self.lower)
        new_upper = np.exp(self.upper)
        new_expected = self.alpha*(np.exp(self.expected)-np.exp(self.lower))+self.beta*(np.exp(self.upper)-np.exp(self.expected))

        res = AIN(new_lower, new_upper, new_expected)
        return res



    def sin(self):
        """
        Compute sine of an AIN instance.

        Returns
        -------
        AIN
            A new AIN instance representing sin(X).

        Examples
        --------
        >>> x = AIN(0, np.pi/2, np.pi/4)
        >>> result = x.sin()
        >>> print(result)
        [0.0000, 1.0000]_{0.6366}

        >>> x = AIN(0, 0, 0)
        >>> result = x.sin()
        >>> print(result)
        [0.0000, 0.0000]_{0.0000}
        """

        # Degenerate interval case
        if self.lower == self.upper:
            val = np.sin(self.expected)
            return AIN(val, val, val)

        # Find min and max of sine on [lower_bound, upper_bound]
        lower_bound, upper_bound, expected_value = self.lower, self.upper, self.expected

        candidates = [np.sin(lower_bound), np.sin(upper_bound)]

        # Check whether the interval contains sine maxima (pi/2 + 2*k*pi)
        k_max_start = int(np.ceil((lower_bound - np.pi / 2) / (2 * np.pi)))
        k_max_end = int(np.floor((upper_bound - np.pi / 2) / (2 * np.pi)))
        for k in range(k_max_start, k_max_end + 1):
            x_max = np.pi / 2 + 2 * k * np.pi
            if lower_bound <= x_max <= upper_bound:
                candidates.append(1.0)

        # Check whether the interval contains sine minima (-np.pi/2 + 2*k*np.pi)
        k_min_start = int(np.ceil((lower_bound + np.pi / 2) / (2 * np.pi)))
        k_min_end = int(np.floor((upper_bound + np.pi / 2) / (2 * np.pi)))
        for k in range(k_min_start, k_min_end + 1):
            x_min = -np.pi / 2 + 2 * k * np.pi
            if lower_bound <= x_min <= upper_bound:
                candidates.append(-1.0)

        new_a = min(candidates)
        new_b = max(candidates)

        # Expected value using LOTUS (Law of the Unconscious Statistician)
        new_c = (self.alpha * (np.cos(lower_bound) - np.cos(expected_value)) +
                 self.beta * (np.cos(expected_value) - np.cos(upper_bound)))

        return AIN(new_a, new_b, new_c)


    def cos(self):
        """
        Compute cosine of an AIN instance.

        Returns
        -------
        AIN
            A new AIN instance representing cos(X).

        Examples
        --------
        >>> x = AIN(0, np.pi/2, np.pi/4)
        >>> result = x.cos()
        >>> print(result)
        [0.0000, 1.0000]_{0.6366}

        >>> x = AIN(0, np.pi, np.pi/2)
        >>> result = x.cos()
        >>> print(result)
        [-1.0000, 1.0000]_{0.0000}
        """

        # Degenerate case
        if self.lower == self.upper:
            val = np.cos(self.expected)
            return AIN(val, val, val)

        # Find min and max of cosine on [lower_bound, upper_bound]
        lower_bound, upper_bound, expected_value = self.lower, self.upper, self.expected

        candidates = [np.cos(lower_bound), np.cos(upper_bound)]

        # Check whether the interval contains cosine maxima (2*k*pi)
        k_max_start = int(np.ceil(lower_bound / (2 * np.pi)))
        k_max_end = int(np.floor(upper_bound / (2 * np.pi)))
        for k in range(k_max_start, k_max_end + 1):
            x_max = 2 * k * np.pi
            if lower_bound <= x_max <= upper_bound:
                candidates.append(1.0)

        # Check whether the interval contains cosine minima (pi + 2*k*pi)
        k_min_start = int(np.ceil((lower_bound - np.pi) / (2 * np.pi)))
        k_min_end = int(np.floor((upper_bound - np.pi) / (2 * np.pi)))
        for k in range(k_min_start, k_min_end + 1):
            x_min = np.pi + 2 * k * np.pi
            if lower_bound <= x_min <= upper_bound:
                candidates.append(-1.0)

        new_a = min(candidates)
        new_b = max(candidates)
        # Expected value using LOTUS
        new_c = (self.alpha * (np.sin(expected_value) - np.sin(lower_bound)) +
                 self.beta * (np.sin(upper_bound) - np.sin(expected_value)))

        return AIN(new_a, new_b, new_c)

    def tan(self):
        """
        Compute tangent of an AIN instance.

        Returns
        -------
        AIN
            A new AIN instance representing tan(X).

        Raises
        ------
        ValueError
            If the interval contains or touches a discontinuity (asymptote) of tan(x).
            tan(x) has asymptotes at x = π/2 + kπ for any integer k.

        Examples
        --------
        >>> x = AIN(0, np.pi/4, np.pi/8)
        >>> result = x.tan()
        >>> print(result)
        [0.0000, 1.0000]_{0.4413}

        >>> x = AIN(-np.pi/4, np.pi/4, 0)
        >>> result = x.tan()
        >>> print(result)
        [-1.0000, 1.0000]_{0.0000}
        """

        # Degenerate case
        if self.lower == self.upper:
            # Check whether the point is an asymptote
            if np.abs(np.cos(self.expected)) < 1e-10:
                raise ValueError(
                    f"tan(x) is undefined at x = {self.expected:.4f} (asymptote at π/2 + kπ)"
                )
            val = np.tan(self.expected)
            return AIN(val, val, val)

        a, b, c = self.lower, self.upper, self.expected

        # Check whether the interval contains or touches a tangent asymptote (pi/2 + k*pi)
        k_asymp_start = int(np.ceil((a - np.pi / 2) / np.pi))
        k_asymp_end = int(np.floor((b - np.pi / 2) / np.pi))

        for k in range(k_asymp_start, k_asymp_end + 1):
            x_asymp = np.pi / 2 + k * np.pi
            if a <= x_asymp <= b:
                raise ValueError(
                    f"The interval [{a:.4f}, {b:.4f}] contains a discontinuity of tan(x) at x = {x_asymp:.4f}. "
                    f"tan(x) is undefined at x = π/2 + kπ."
                )

        # If there are no asymptotes, tan is monotonically increasing
        new_a = np.tan(a)
        new_b = np.tan(b)

        new_c = (self.alpha * np.log(np.abs(np.cos(a) / np.cos(c))) +
                 self.beta * np.log(np.abs(np.cos(c) / np.cos(b))))

        return AIN(new_a, new_b, new_c)

    @classmethod
    def from_samples(cls, data, method='minmax', clip_outliers=False):
        """
        Create AIN from empirical data using various methods.

        This method constructs an AIN instance from a collection of samples,
        offering multiple strategies for determining interval bounds and handling outliers.

        Parameters
        ----------
        data : array-like
            Sample data (list, numpy array, etc.)
        method : str, optional
            Method for determining bounds (default: 'minmax'):
            - 'minmax': Use minimum and maximum values
            - 'percentile': Use 1st and 99th percentiles
            - 'iqr': Use interquartile range (Q1-Q3)
            - 'std': Use mean ± 3*sigma
            - 'mad': Use median ± 3*MAD (Median Absolute Deviation)
        clip_outliers : bool, optional
            Whether to remove outliers before calculations using IQR method (default: False)

        Returns
        -------
        AIN
            A new AIN instance with bounds determined by the chosen method
            and expected value equal to the sample mean.

        Raises
        ------
        ValueError
            If data is empty or method is unknown.
        TypeError
            If data cannot be converted to numpy array.

        Examples
        --------
        Basic usage with default method:
        >>> data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> x = AIN.from_samples(data)
        >>> print(x)
        [1.0000, 10.0000]_{5.5000}

        Using percentile method (robust to outliers):
        >>> data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]  # 100 is outlier
        >>> x = AIN.from_samples(data, method='percentile')
        >>> print(x)
        [1.0900, 91.8100]_{14.5000}

        Using IQR method (focuses on central 50%):
        >>> data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> x = AIN.from_samples(data, method='iqr')
        >>> print(x)
        [3.2500, 7.7500]_{5.5000}

        Removing outliers before analysis:
        >>> data = [100, 102, 98, 101, 99, 103, 500, 97, 102, 100]
        >>> x = AIN.from_samples(data, method='minmax', clip_outliers=True)
        >>> print(x)
        [97.0000, 103.0000]_{100.2222}

        Using MAD method (most robust):
        >>> data = [10, 11, 10.5, 11.2, 10.8, 999]
        >>> x = AIN.from_samples(data, method='mad')
        >>> print(x)
        [9.8500, 11.9500]_{10.7000}

        Notes
        -----
        Method selection guide:
        - 'minmax': Best for clean data without outliers
        - 'percentile': Good balance between robustness and data retention
        - 'iqr': Very robust, focuses on central tendency
        - 'std': Assumes normal distribution, good for theoretical bounds
        - 'mad': Most robust to outliers, works with any distribution

        The clip_outliers option uses the IQR method with 1.5*IQR rule to
        identify and remove outliers before applying the selected method.
        """
        # Convert to numpy array
        try:
            data = np.array(data, dtype=float)
        except (ValueError, TypeError) as e:
            raise TypeError(f"Data must be convertible to numpy array: {e}")

        if len(data) == 0:
            raise ValueError("Data cannot be empty")

        # Remove outliers if requested
        if clip_outliers:
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data = data[(data >= lower_bound) & (data <= upper_bound)]

            if len(data) == 0:
                raise ValueError("All data points were identified as outliers")

        # Compute bounds based on method
        if method == 'minmax':
            lower = np.min(data)
            upper = np.max(data)

        elif method == 'percentile':
            lower = np.percentile(data, 1)
            upper = np.percentile(data, 99)

        elif method == 'iqr':
            lower = np.percentile(data, 25)  # Q1
            upper = np.percentile(data, 75)  # Q3

        elif method == 'std':
            mean = np.mean(data)
            std = np.std(data)
            lower = mean - 3 * std
            upper = mean + 3 * std

        elif method == 'mad':
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            lower = median - 3 * mad
            upper = median + 3 * mad

        else:
            raise ValueError(f"Unknown method: '{method}'. Use 'minmax', 'percentile', 'iqr', 'std', or 'mad'.")

        # Expected value is always the mean of the (possibly clipped) data
        if method == 'mad':
            inliers = data[(data >= lower) & (data <= upper)]
            expected = np.mean(inliers) if len(inliers) > 0 else np.median(data)
        else:
            expected = np.mean(data)

        # Ensure bounds are valid
        if lower > upper:
            lower, upper = upper, lower

        if lower == upper:
            # Degenerate case - all data points are identical
            return cls(lower, upper, expected)

        # Ensure expected is within bounds
        if expected < lower:
            expected = lower
        elif expected > upper:
            expected = upper

        return cls(lower, upper, expected)

    def samples(self, n, rounding_precision=4, rng=None):
        """
        Generate random samples from the AIN distribution.

        Parameters
        ----------
        n : int
            Number of random samples to generate.
        rounding_precision : int, optional
            Number of decimal places to round the samples (default is 4).
        rng : numpy.random.Generator, optional
            A NumPy random number generator instance for reproducibility (default is None, which uses the global random state).

        Returns
        -------
        numpy.ndarray
            An array of `n` random samples drawn from the AIN distribution.

        Raises
        ------
        ValueError
            If `n` is not a positive integer.

        Examples
        --------
        >>> ain = AIN(0, 10, 5)
        >>> rng = np.random.default_rng(seed=42)
        >>> data = ain.samples(5, 4, rng)
        >>> print(data)
        [7.7396 4.3888 8.586  6.9737 0.9418]

        Notes
        -----
        This method uses inverse transform sampling to generate samples according to the AIN distribution.
        """
        if not isinstance(n, int) or n <= 1:
            raise ValueError("n must be a positive integer, greater than 1.")
        if rng is not None:
            u = rng.uniform(0, 1, n)
        else:
            u = np.random.uniform(0, 1, n)

        samples = np.array([self.quantile(el) for el in u])
        return np.round(samples, rounding_precision)


    def to_list(self):
        # WS_to_check_common_sense
        """
        Convert AIN to list [lower, upper, expected].

        Returns
        -------
        list
            [lower, upper, expected]

        Examples
        --------
        >>> x = AIN(0, 10, 5)
        >>> print(x.to_list())
        [0, 10, 5]
        """
        return [self.lower, self.upper, self.expected]

    @classmethod
    def from_list(cls, lst):
        # WS_to_check_common_sense
        """
        Create AIN from list [lower, upper, expected].

        Parameters
        ----------
        lst : list
            List containing [lower, upper, expected]

        Returns
        -------
        AIN
            AIN instance created from the list.

        Raises
        ------
        ValueError
            If the list does not contain exactly three elements.

        Examples
        --------
        >>> lst = [0, 10, 5]
        >>> x = AIN.from_list(lst)
        >>> print(x)
        [0.0000, 10.0000]_{5.0000}
        """
        if not isinstance(lst, list) or len(lst) != 3:
            raise ValueError("Input must be a list of three elements: [lower, upper, expected]")
        return cls(lst[0], lst[1], lst[2])


    def to_numpy(self):
        # WS_to_check_common_sense
        """
        Convert AIN to numpy array [lower, upper, expected].

        Returns
        -------
        numpy.ndarray
            Array containing [lower, upper, expected]

        Examples
        --------
        >>> x = AIN(0., 10., 5.)
        >>> arr = x.to_numpy()
        >>> print(arr)
        [ 0. 10.  5.]
        >>> print(type(arr))
        <class 'numpy.ndarray'>
        """
        return np.array([self.lower, self.upper, self.expected])

    @classmethod
    def from_numpy(cls, arr):
        # WS_to_check_common_sense
        """
        Create AIN from numpy array [lower, upper, expected].

        Parameters
        ----------
        arr : numpy.ndarray
            Numpy array containing [lower, upper, expected]

        Returns
        -------
        AIN
            AIN instance created from the numpy array.

        Raises
        ------
        ValueError
            If the array does not contain exactly three elements.

        Examples
        --------
        >>> arr = np.array([0., 10., 5.])
        >>> x = AIN.from_numpy(arr)
        >>> print(x)
        [0.0000, 10.0000]_{5.0000}
        """
        if not isinstance(arr, np.ndarray) or arr.shape != (3,):
            raise ValueError("Input must be a numpy array of shape (3,): [lower, upper, expected]")
        return cls(arr[0], arr[1], arr[2])

    def to_dict(self):
        """
        Export AIN to a dictionary.

        Returns
        -------
        dict
            Dictionary containing all AIN attributes.

        Examples
        --------
        >>> x = AIN(0, 4, 2)
        >>> d = x.to_dict()
        >>> print(d)
        {'lower': 0, 'upper': 4, 'expected': 2}
        """
        return {
            'lower': self.lower,
            'upper': self.upper,
            'expected': self.expected,
        }

    @classmethod
    def from_dict(cls, d):
        """
        Create AIN from a dictionary.

        Parameters
        ----------
        d : dict
            Dictionary containing at least 'lower', 'upper', and optionally 'expected'.

        Returns
        -------
        AIN
            A new AIN instance.

        Examples
        --------
        >>> d = {'lower': 0, 'upper': 10, 'expected': 5}
        >>> x = AIN.from_dict(d)
        >>> print(x)
        [0.0000, 10.0000]_{5.0000}
        """
        if not isinstance(d, dict):
            raise TypeError("Input must be a dictionary")

        if 'lower' not in d or 'upper' not in d:
            raise ValueError("Dictionary must contain 'lower' and 'upper' keys")

        return cls(d['lower'], d['upper'], d.get('expected'))

    def to_tuple(self):
        """
        Convert AIN to tuple (lower, upper, expected).

        Returns
        -------
        tuple
            (lower, upper, expected)

        Examples
        --------
        >>> x = AIN(0, 10, 5)
        >>> print(x.to_tuple())
        (0, 10, 5)
        """
        return (self.lower, self.upper, self.expected)

    @classmethod
    def from_tuple(cls, t):
        """
        Parameters
        ----------
        t : tuple
            Tuple containing at least 'lower', 'upper', and optionally 'expected'.

        Returns
        -------
        AIN
            A new AIN instance.

        Examples
        --------
        >>> x = (0, 10, 5)
        >>> ain = AIN.from_tuple(x)
        >>> ain
        AIN(0, 10, 5)
        """
        return cls(t[0], t[1], t[2])

    def to_json(self):
        """
        Convert AIN to JSON string.

        Returns
        -------
        str
            JSON string representation of the AIN

        Examples
        --------
        >>> x = AIN(0, 10, 5)
        >>> j = x.to_json()
        >>> print(j)
        {"lower": 0, "upper": 10, "expected": 5}
        """
        import json
        data = {
            'lower': self.lower,
            'upper': self.upper,
            'expected': self.expected
        }
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str):
        """
        Create AIN from a JSON string.

        This is the inverse of to_json(). Parses a JSON string and creates
        an AIN instance from the data.

        Parameters
        ----------
        json_str : str
            JSON string containing AIN data. Must contain at least 'lower' and 'upper' keys.

        Returns
        -------
        AIN
            A new AIN instance.

        Raises
        ------
        TypeError
            If json_str is not a string
        ValueError
            If JSON is invalid or missing required keys

        Examples
        --------
        >>> json_str = '{"lower": 0, "upper": 10, "expected": 5}'
        >>> x = AIN.from_json(json_str)
        >>> print(x)
        [0.0000, 10.0000]_{5.0000}

        >>> # Round-trip conversion
        >>> original = AIN(1, 5, 3)
        >>> json_str = original.to_json()
        >>> restored = AIN.from_json(json_str)
        >>> print(restored)
        [1.0000, 5.0000]_{3.0000}
        """
        import json

        if not isinstance(json_str, str):
            raise TypeError("Input must be a string")

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {e}")

        if not isinstance(data, dict):
            raise ValueError("JSON must represent a dictionary/object")

        if 'lower' not in data or 'upper' not in data:
            raise ValueError("JSON must contain 'lower' and 'upper' keys")

        return cls(data['lower'], data['upper'], data.get('expected'))
    @classmethod
    def normalize_ains_list(cls, list_ain, mode='minmax', type='profit'):
        """
        Normalize a list of AIN instances.

        Parameters
        ----------
        list_ain : list
            List of AIN instances to normalize.
        mode : str, optional
            Normalization mode. Reserved for future use; currently, only the default
            behavior is applied regardless of the mode. Defaults to 'minmax'.

        Returns
        -------
        list
            List of normalized AIN instances representing a probability distribution.

        Raises
        ------
        ValueError
            If the input list is empty or contains non-AIN elements.

        Examples
        --------
        >>> a1 = AIN(2, 10, 5)
        >>> a2 = AIN(10, 20, 15)
        >>> normalized = AIN.normalize_ains_list([a1, a2])
        >>> for ain in normalized:
        ...     print(ain)
        [0.0000, 0.4444]_{0.1667}
        [0.4444, 1.0000]_{0.7222}
        """
    # Ensure all elements are AIN instances before computing the total
        for ain in list_ain:
            if not isinstance(ain, AIN):
                raise ValueError("All elements in the list must be AIN instances.")

        max_val = max([ain.upper for ain in list_ain])
        min_val = min([ain.lower for ain in list_ain])

        if mode == 'minmax':
            if type=='profit':
                normalized_list = [(ain-min_val)/(max_val-min_val) for ain in list_ain]
            elif type=='cost':
                normalized_list = [(max_val-ain)/(max_val-min_val) for ain in list_ain]
            else:
                raise ValueError(f"Unknown type: '{type}'. Currently, only 'profit' and 'cost' are supported.")
        else:
            raise ValueError(f"Unknown normalization mode: '{mode}'. Currently, only 'minmax' is supported.")

        return normalized_list


class GraphAIN:
    """
    A class for creating and visualizing graphs with AIN (Asymmetric Interval Number) nodes.

    Supports both directed and undirected graphs where edges are weighted based on
    the probability relationships between AIN instances.

    Parameters
    ----------
    directed : bool, optional
        If True, creates a directed graph. If False, creates an undirected graph.
        Default is False.

    Attributes
    ----------
    graph : networkx.Graph or networkx.DiGraph
        The underlying NetworkX graph object.
    nodes_data : dict
        Dictionary mapping node names to their AIN instances.
    directed : bool
        Whether the graph is directed or undirected.

    Examples
    --------
    Creating an undirected graph:
    >>> A = AIN(0, 10, 2)
    >>> B = AIN(2, 8, 3)
    >>> C = AIN(4, 12, 5)
    >>> D = AIN(6, 14, 11)
    >>>
    >>> g = GraphAIN(directed=False)
    >>> g.add_node("A", A)
    >>> g.add_node("B", B)
    >>> g.add_node("C", C)
    >>> g.add_node("D", D)
    >>>
    >>> g.plot() # doctest: +SKIP

    Creating a directed graph:
    >>> g_directed = GraphAIN(directed=True)
    >>> g_directed.add_node("A", A)
    >>> g_directed.add_node("B", B)
    >>> _ = g_directed.plot() # doctest: +SKIP
    """

    def __init__(self, directed=False, edge_threshold=0.0, dominance_only=False):
        """
        Initialize a GraphAIN instance.

        Parameters
        ----------
        directed : bool, optional
            If True, creates a directed graph. Default is False (undirected).
        edge_threshold : float, optional
            Minimum edge weight required to add an edge.
            Edges with weight <= edge_threshold are ignored.
            Default is 0.0.
        dominance_only: bool, optional
            If True (directed graphs only), for each pair of nodes (A, B)
            only the direction with the larger weight is added

        Examples
        --------
        >>> g = GraphAIN(directed=True)
        >>> print(g.directed)
        True
        >>> g2 = GraphAIN()
        >>> print(g2.directed)
        False
        """

        if not isinstance(directed, bool):
            raise TypeError("directed must be a boolean")
        if not isinstance(dominance_only, bool):
            raise TypeError("dominance_only must be a boolean")
        if not isinstance(edge_threshold, float):
            raise TypeError("edge_threshold must be a float")
        if edge_threshold < 0.0 or edge_threshold > 1.0:
            raise ValueError("edge_threshold must be between 0.0 and 1.0")

        self.directed = directed
        self.edge_threshold = edge_threshold
        self.dominance_only = dominance_only
        if directed:
            self.graph = nx.DiGraph()
        else:
            self.graph = nx.Graph()

        self.nodes_data = {}

    def add_node(self, name, ain_instance):
        """
        Add a node to the graph with an associated AIN instance.

        Parameters
        ----------
        name : str
            The name/label of the node.
        ain_instance : AIN
            The AIN instance associated with this node.

        Raises
        ------
        TypeError
            If name is not a string or ain_instance is not an AIN instance.
        ValueError
            If a node with this name already exists.

        Examples
        --------
        >>> g = GraphAIN()
        >>> a = AIN(0, 10, 5)
        >>> g.add_node("A", a)
        >>> "A" in g.nodes_data
        True
        """
        if not isinstance(name, str):
            raise TypeError("Node name must be a string")
        if not isinstance(ain_instance, AIN):
            raise TypeError("ain_instance must be an AIN instance")
        if name in self.nodes_data:
            raise ValueError(f"Node '{name}' already exists in the graph")

        self.graph.add_node(name)
        self.nodes_data[name] = ain_instance

        for other, other_ain in self.nodes_data.items():
            if other == name:
                continue

            if self.directed:
                if self.dominance_only:
                    self._add_directed_edge_max(name, other)

                else:
                    self._add_directed_edge(name, other)
                    self._add_directed_edge(other, name)
            else:
                self._add_undirected_edge(name, other)

    def _add_directed_edge_max(self, u, v):
        p_uv = self.nodes_data[u] > self.nodes_data[v]
        p_vu = self.nodes_data[v] > self.nodes_data[u]

        if p_uv > p_vu:
            weight = p_uv
            src, dst = u, v
        elif p_vu > p_uv:
            weight = p_vu
            src, dst = v, u
        else:
            src, dst = (u, v) if u < v else (v, u)
            weight = p_uv  # == p_vu

        if weight > self.edge_threshold:
            self.graph.add_edge(src, dst, weight=weight)

    def _add_directed_edge(self, u, v):
        p = self.nodes_data[u] > self.nodes_data[v]
        w = float(f"{p:.4f}")
        if w > self.edge_threshold:
            self.graph.add_edge(u, v, weight=w)

    def _add_undirected_edge(self, u, v):
        p = self.nodes_data[v] > self.nodes_data[u]
        w = float(f"{4 * p * (1 - p):.4f}")
        if w > self.edge_threshold:
            self.graph.add_edge(u, v, weight=w)

    def plot(self, figsize=(5, 4), node_size=1000, font_size=12,
             layout='spring', seed=42, save_path=None, dpi=300, edge_decimals = 2):
        """
        Visualize the graph using matplotlib and networkx.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size as (width, height). Default is (5, 4).
        node_size : int, optional
            Size of the nodes. Default is 1000.
        font_size : int, optional
            Font size for node labels. Default is 12.
        layout : str, optional
            Layout algorithm: 'spring', 'circular', 'kamada_kawai', 'random'.
            Default is 'spring'.
        seed : int, optional
            Random seed for layout algorithms. Default is 42.
        save_path : str, optional
            If provided, saves the figure to this path. Default is None.
        dpi : int, optional
            Resolution for saved figure. Default is 300.

        Returns
        -------
        matplotlib.figure.Figure
            The matplotlib figure object.

        Examples
        --------
        >>> g = GraphAIN()
        >>> # ... add nodes ...
        >>> _ = g.plot(layout='circular', save_path='my_graph.pdf')  # doctest: +SKIP
        """
        import networkx as nx
        import matplotlib.pyplot as plt

        if len(self.graph.nodes()) == 0:
            raise ValueError("Graph has no nodes to plot")

        fig, ax = plt.subplots(figsize=figsize)

        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(self.graph, seed=seed)
        elif layout == 'circular':
            pos = nx.circular_layout(self.graph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(self.graph)
        elif layout == 'random':
            pos = nx.random_layout(self.graph, seed=seed)
        else:
            raise ValueError(f"Unknown layout: '{layout}'")

        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph, pos,
            node_size=node_size,
            node_color='lightblue'
        )

        # Draw labels
        nx.draw_networkx_labels(
            self.graph, pos,
            font_size=font_size
        )

        if self.directed:
            # Draw directed edges with curvature for bidirectional pairs
            for (u, v) in self.graph.edges():
                rad = 0.12 if self.graph.has_edge(v, u) else 0.0
                nx.draw_networkx_edges(
                    self.graph, pos,
                    edgelist=[(u, v)],
                    arrows=True,
                    arrowstyle="-|>",
                    arrowsize=20,
                    edge_color='gray',
                    connectionstyle=f"arc3,rad={rad}",
                    min_source_margin=15,
                    min_target_margin=15
                )
        else:
            # Draw undirected edges
            nx.draw_networkx_edges(
                self.graph, pos,
                edge_color='gray'
            )

        # Draw edge labels (weights)
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        if edge_labels:
            if self.directed:
                # Place labels slightly off the edge to avoid overlap
                for (u, v), w in edge_labels.items():
                    x1, y1 = pos[u]
                    x2, y2 = pos[v]

                    xm = 0.5 * (x1 + x2)
                    ym = 0.5 * (y1 + y2)

                    rad = 0.12 if self.graph.has_edge(v, u) else 0.0
                    dx = (y2 - y1) * rad * 0.7
                    dy = -(x2 - x1) * rad * 0.7

                    plt.text(
                        xm + dx, ym + dy,
                        f"{w:.{edge_decimals}f}",
                        fontsize=font_size - 2,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.9)
                    )
            else:
                nx.draw_networkx_edge_labels(
                    self.graph, pos,
                    edge_labels=edge_labels,
                    font_size=font_size - 2
                )

        plt.gca().set_axis_off()

        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

        plt.show()
        plt.close(fig)
        return fig

    def get_adjacency_matrix(self):
        """
        Get the adjacency matrix of the graph.

        Returns
        -------
        numpy.ndarray
            The adjacency matrix with edge weights.

        Examples
        --------
        >>> g = GraphAIN(directed=True)
        >>> a = AIN(0, 10, 2)
        >>> b = AIN(2, 8, 3)
        >>> g.add_node("A", a)
        >>> g.add_node("B", b)
        >>> M = g.get_adjacency_matrix()
        >>> M.shape
        (2, 2)
        >>> float(M[0, 1]) == (g.get_edge_weight("A", "B") or 0.0)
        True
        """
        return nx.adjacency_matrix(self.graph).todense()

    def get_edge_weight(self, node1, node2):
        """
        Get the weight of an edge between two nodes.

        Parameters
        ----------
        node1 : str
            Name of the first node.
        node2 : str
            Name of the second node.

        Returns
        -------
        float or None
            The edge weight, or None if no edge exists.

        Examples
        --------
        >>> g = GraphAIN()
        >>> # ... add nodes and edges ...
        >>> weight = g.get_edge_weight("A", "B")
        """
        if self.graph.has_edge(node1, node2):
            return self.graph[node1][node2]['weight']
        return None

    def get_node_degree(self, node):
        """
        Get the degree of a node.

        Parameters
        ----------
        node : str
            Name of the node.

        Returns
        -------
        int
            The degree of the node.

        Examples
        --------
        >>> g = GraphAIN()
        >>> # ... add nodes and edges ...
        >>> degree = g.get_node_degree("A")
        """
        return self.graph.degree(node)

    def summary(self):
        """
        Print a summary of the graph.

        Examples
        --------
        >>> g = GraphAIN(directed=True)
        >>> a = AIN(0, 10, 2)
        >>> b = AIN(2, 8, 3)
        >>> g.add_node("A", a)
        >>> g.add_node("B", b)
        >>> g.summary()
        ==================================================
        Graph Type: Directed
        Number of Nodes: 2
        Number of Edges: 2
        ==================================================
        Nodes:
          A: [0.0000, 10.0000]_{2.0000}
          B: [2.0000, 8.0000]_{3.0000}
        ==================================================
        Edges (with weights):
          A -> B: 0.1750
          B -> A: 0.8250
        ==================================================
        """
        print("=" * 50)
        print(f"Graph Type: {'Directed' if self.directed else 'Undirected'}")
        print(f"Number of Nodes: {self.graph.number_of_nodes()}")
        print(f"Number of Edges: {self.graph.number_of_edges()}")
        print("=" * 50)
        print("Nodes:")
        for node, ain in self.nodes_data.items():
            print(f"  {node}: {ain}")
        print("=" * 50)
        print("Edges (with weights):")
        for u, v, data in self.graph.edges(data=True):
            print(f"  {u} -> {v}: {data['weight']:.4f}")
        print("=" * 50)

    def __repr__(self):
        """String representation of the GraphAIN instance."""
        graph_type = "Directed" if self.directed else "Undirected"
        return (f"GraphAIN({graph_type}, "
                f"nodes={self.graph.number_of_nodes()}, "
                f"edges={self.graph.number_of_edges()})")




A = AIN(0, 10, 2)
B = AIN(2, 8, 3)
C = AIN(4, 12, 5)
D = AIN(6, 14, 11)
g = GraphAIN(directed=False, edge_threshold=0.0, dominance_only=True)
g.add_node("A", A)
g.add_node("B", B)
g.add_node("C", C)
g.add_node("D", D)
_ = g.plot(layout='circular')



# A = AIN(0, 10, 2)
B = AIN(2, 8, 3)
C = AIN(4, 12, 5)
D = AIN(6, 14, 11)
g = GraphAIN(directed=True, edge_threshold=0.0, dominance_only=True)
g.add_node("A", A)
g.add_node("B", B)
g.add_node("C", C)
g.add_node("D", D)
g.summary()
_ = g.plot(layout='circular', edge_decimals=3)




