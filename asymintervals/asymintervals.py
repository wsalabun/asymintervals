import numpy as np
import matplotlib.pyplot as plt


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
            self.alpha = 1.0
            self.beta = 1.0
            self.asymmetry = 0.0
            self.D2 = 0.0
        else:
            self.alpha = (self.upper - self.expected) / ((self.upper - self.lower) * (self.expected - self.lower))
            self.beta = (self.expected - self.lower) / ((self.upper - self.lower) * (self.upper - self.expected))
            self.asymmetry = (self.lower + self.upper - 2 * self.expected) / (self.upper - self.lower)
            self.D2 = self.alpha * (self.expected ** 3 - self.lower ** 3) / 3 + self.beta * (
                    self.upper ** 3 - self.expected ** 3) / 3 - expected ** 2

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
        [9.8500, 11.9500]_{11.9500}

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
        >>> samples = ain.samples(5, rng)
        >>> print(samples)
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


# print([method for method in dir(AIN) if not method.startswith('_')])


Added_function_names = ['sin()', 'cos()', 'from_samples()', 'samples()']

x = AIN(0,10,2)

data = x.samples(100)
print(data)
print(len(data))
print(np.mean(data))