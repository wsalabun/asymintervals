import numpy as np
import matplotlib.pyplot as plt

class AIN:
    def __init__(self, lower: float, upper: float, expected: float = None):
        """
        Initialize an Asymmetric Interval Number (AIN) with specified bounds and an optional expected value.

        This constructor creates an instance of AIN using `lower` and `upper` bounds to define the interval.
        Optionally, an `expected` value within this range can be provided. If `expected` is not specified,
        it defaults to the midpoint of `lower` and `upper`. The `expected` value must lie within the interval
        `[lower, upper]`. Additionally, asymmetry coefficients (`alpha`, `beta`) and the degree of asymmetry
        (`asymmetry`) are calculated based on the specified bounds and expected value.

        Parameters
        ----------
        lower : float
            The lower bound of the interval. Must be less than `upper`.
        upper : float
            The upper bound of the interval. Must be greater than `lower`.
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
            The expected value within the interval.
        alpha : float
            The asymmetry coefficient for the interval, calculated if `lower` is not equal to `upper`.
        beta : float
            The asymmetry coefficient for the interval, calculated if `lower` is not equal to `upper`.
        asymmetry : float
            The asymmetry degree of the interval, representing the relative position of `expected`
            between `lower` and `upper`.
        D2 : float
            A parameter derived from asymmetry calculations (if applicable).

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
        else:
            self.alpha = (self.upper - self.expected) / ((self.upper - self.lower) * (self.expected - self.lower))
            self.beta = (self.expected - self.lower) / ((self.upper - self.lower) * (self.upper - self.expected))
            self.asymmetry = (self.lower + self.upper - 2 * self.expected) / (self.upper - self.lower)
            self.D2 = self.alpha * (self.expected ** 3 - self.lower ** 3) / 3 + self.beta * (
                    self.upper ** 3 - self.expected ** 3) / 3 - expected ** 2

    def __repr__(self):
        """
        Return a string representation of the AIN instance that is as unambiguous as possible.
        The string representation includes the class name `AIN` followed by its 'lower',
        'upper', and 'expected' values in parentheses.

        Returns
        -------
        str
            A string that closely represents how the instance was constructed.

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

        The string format is '[lower, upper]_{expected}' where 'lower', 'upper',
        and 'expected' are formatted to four decimal places. This format is intended
        to provide a clear and concise description of the AIN instance that is
        suitable for printing and easy for an end-user to read.

        Returns
        -------
        str
            A string representation of the AIN instance, formatted to four decimal
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
        Returns a new AIN instance representing the negation of the current instance.

        The negation of an AIN instance involves inverting the 'lower' and 'upper'
        bounds as well as the 'expected' value. Specifically, the new 'lower' bound is
        the negation of the original 'upper' bound, the new 'upper' bound is the negation
        of the original 'lower' bound, and the new 'expected' value is the negation of
        the original 'expected' value.

        Returns
        -------
        AIN
            A new AIN instance with negated 'lower', 'upper', and 'expected' values.

        Examples
        --------
        >>> a = AIN(1, 10, 8)
        >>> print(-a)
        [-10.0000, -1.0000]_{-8.0000}

        >>> b = AIN(2, 10)
        >>> print(-b)
        [-10.0000, -2.0000]_{-6.0000}
        """
        return AIN(-self.upper, -self.lower, -self.expected)

    def __add__(self, other):
        """
        Add an AIN instance or a float or int to the current AIN instance.

        This method enables addition of either another Asymmetric Interval Number (AIN)
        instance or a float or int to the current AIN instance, returning a new AIN
        instance representing the result. When adding another AIN, the resulting lower,
        upper, and expected values are computed by summing the respective values of both AINs.
        When adding a float or int, the scalar is added to each component of the AIN instance.

        Parameters
        ----------
        other : AIN, float, or int
            The value to add, which can be another AIN instance, a float, or an int.

        Returns
        -------
        AIN
            A new AIN instance representing the result of the addition, with updated
            lower, upper, and expected values based on the operation.

        Raises
        ------
        TypeError
            If `other` is not an instance of AIN, float, or int.

        Examples
        --------
        Addition with another AIN instance:
        >>> a = AIN(1, 10, 8)
        >>> b = AIN(0, 5, 2)
        >>> print(a + b)
        [1.0000, 15.0000]_{10.0000}

        Addition with a float or int:
        >>> a = AIN(1, 10, 8)
        >>> b = 2
        >>> print(a + b)
        [3.0000, 12.0000]_{10.0000}
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

        This method enables the addition of an Asymmetric Interval Number (AIN) instance
        to a float or int when the AIN appears on the right side of the addition (i.e., `other + self`).
        It calculates `other + self`, allowing for commutative addition with floats and ints.

        Parameters
        ----------
        other : float or int
            The value to add to the current AIN instance.

        Returns
        -------
        AIN
            A new AIN instance representing the result of the addition, with lower, upper,
            and expected values equal to the sum of `other` and the corresponding values of
            the current AIN instance.

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
        """
        if not isinstance(other, (float, int)):
            raise TypeError("other must be of type float or int")
        return self + other

    def __sub__(self, other):
        """
        Subtract an AIN instance or a float or int from the current AIN instance.

        This method allows subtraction of either another Asymmetric Interval Number (AIN)
        or a float or int from the current AIN instance, returning a new AIN instance
        with the result. When subtracting another AIN, the resulting bounds and expected
        value are computed by subtracting the corresponding values of the operands. If
        subtracting a float or int, the scalar is subtracted from each component of the
        current AIN instance.

        Parameters
        ----------
        other : AIN, float, or int
            The value to subtract, which can be an AIN instance, a float, or an int.

        Returns
        -------
        AIN
            A new AIN instance representing the result of the subtraction, with adjusted
            lower, upper, and expected values based on the operation.

        Raises
        ------
        TypeError
            If `other` is not an instance of AIN, float, or int.

        Examples
        --------
        Subtracting an AIN instance:
        >>> a = AIN(1, 10, 8)
        >>> b = AIN(0, 5, 2)
        >>> print(a - b)
        [-4.0000, 10.0000]_{6.0000}

        Subtracting a float or int:
        >>> a = AIN(1, 10, 8)
        >>> b = 2
        >>> print(a - b)
        [-1.0000, 8.0000]_{6.0000}
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
        Perform reflected (reverse) subtraction for an AIN instance.

        This method is invoked when an Asymmetric Interval Number (AIN) instance appears
        on the right-hand side of a subtraction operation (i.e., `other - self`) and the
        left operand (`other`) does not support subtraction with an AIN. It calculates
        the result of `other - self`.

        Parameters
        ----------
        other : float or int
            The value from which the current AIN instance is subtracted.

        Returns
        -------
        AIN
            A new AIN instance representing the result of the subtraction. The resulting
            AIN has its lower, upper, and expected values computed as the difference between
            `other` and the respective values of the AIN instance.

        Raises
        ------
        TypeError
            If `other` is not a float or int.

        Examples
        --------
        >>> a = AIN(1, 10, 8)
        >>> b = 5
        >>> print(b - a)
        [-5.0000, 4.0000]_{-3.0000}
        """
        if not isinstance(other, (float, int)):
            raise TypeError("other must be of type float or int")
        return -self + other

    def __mul__(self, other):
        """
        Perform multiplication of the current AIN instance with another AIN or a float or int.

        This method allows the multiplication of an Asymmetric Interval Number (AIN) instance
        with another AIN instance or a float or int, returning a new AIN instance that
        represents the result. When multiplying with another AIN, the interval boundaries
        are computed based on the combinations of bounds from both AIN instances.

        Parameters
        ----------
        other : AIN, float, or int
            The value to multiply with, which can be another AIN instance, a float, or an int.

        Returns
        -------
        AIN
            A new AIN instance representing the product of the multiplication.

        Raises
        ------
        TypeError
            If `other` is not an instance of AIN, float, or int.

        Examples
        --------
        Multiplying with another AIN instance:
        >>> a = AIN(1, 3, 2)
        >>> b = AIN(2, 4, 3)
        >>> print(a * b)
        [2.0000, 12.0000]_{6.0000}

        Multiplying with a float or int:
        >>> a = AIN(1, 3, 2)
        >>> b = 2
        >>> print(a * b)
        [2.0000, 6.0000]_{4.0000}
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
        Perform reverse multiplication for an AIN instance with a float or int.

        This method allows an Asymmetric Interval Number (AIN) instance to be multiplied
        by a float or int in cases where the float or int appears on the left side of
        the multiplication (i.e., `other * self`). This enables commutative multiplication
        between AIN and float or int values.

        Parameters
        ----------
        other : float or int
            The float or int value to multiply with the AIN instance.

        Returns
        -------
        AIN
            A new AIN instance representing the result of the multiplication.

        Raises
        ------
        TypeError
            If `other` is not a float or int.

        Examples
        --------
        >>> a = AIN(1, 3, 2)
        >>> scalar = 2
        >>> result = scalar * a
        >>> print(result)
        [2.0000, 6.0000]_{4.0000}
        """
        if not isinstance(other, (float, int)):
            raise TypeError("other must be float or int")
        return self * other

    def __truediv__(self, other):
        """
        Perform division of the current AIN instance by another AIN instance or a float or int.

        This method supports division by either another Asymmetric Interval Number (AIN) or a
        float or int, returning a new AIN instance as the result. When dividing by an AIN,
        interval boundaries are calculated by dividing the respective boundaries, while the
        expected value is adjusted based on logarithmic calculations if the bounds differ.

        Parameters
        ----------
        other : AIN, float, or int
            The divisor, which can be an AIN instance, a float, or an int.

        Returns
        -------
        AIN
            A new AIN instance representing the result of the division.

        Raises
        ------
        TypeError
            If `other` is not an instance of AIN, float, or int.

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
        Perform reverse true division of a float or int by an Asymmetric Interval Number (AIN) instance.

        This method enables division where a float or int `other` is divided by an AIN instance (`self`),
        calculating the reciprocal of `self` and then scaling it by `other`. It returns a new AIN
        instance representing the outcome.

        Parameters
        ----------
        other : float or int
            The float or int to divide by the AIN instance.

        Returns
        -------
        AIN
            A new AIN instance representing `other` divided by `self`.

        Raises
        ------
        TypeError
            If `other` is not a float or int.

        Examples
        --------
        >>> a = AIN(2, 4, 3)
        >>> result = 10 / a
        >>> print(result)
        [2.5000, 5.0000]_{3.4657}
        """
        if not isinstance(other, (float, int)):
            raise TypeError(f"other variable is not a float or int")
        return other * self**(-1)

    def __pow__(self, n):
        """
        Raise the Asymmetric Interval Number (AIN) instance to the power `n`.

        This method computes the result of raising the AIN instance to the specified exponent `n`.

        Parameters
        ----------
        n : int or float
            The exponent to which the AIN is raised. Valid values include positive or negative real numbers.

        Raises
        ------
        TypeError
            If `n` is not a float or int.
        ValueError
            If the operation would result in a complex number (e.g., taking the square root of a negative value),
            or if `n = -1` and the interval includes 0, as division by zero is undefined.

        Returns
        -------
        AIN
            A new AIN instance representing the interval raised to the power of `n`.

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
        Calculate the probability density function (PDF) value for the Asymmetric Interval Number (AIN) at a given point `x`.

        This method evaluates the probability density at `x` within the AIN-defined interval. The PDF describes
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
            If `x` is not an integer or float.

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

        This method evaluates the cumulative distribution function (CDF) of the AIN instance at
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
            If `x` is not an int or float.

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
        for the AIN instance at a specified probability level `y`. The quantile represents the value
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
            If `y` is not a float or int value.

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
            If `precision` is not an integer, a ValueError is raised with an informative message.

        Example
        -------
        >>> a = AIN(0, 10, 2)
        >>> a.summary(precision=4)
        === AIN ============================
        [0.0000, 10.0000]_{2.0000}
        === Summary ========================
        Alpha        =     0.4000
        Beta         =     0.0250
        Assymetry    =     0.6000
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
            ('Assymetry', f'{self.asymmetry:.{precision}f}'),
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
        Plots the intervals and key values of an Asymmetric Interval Number (AIN) instance.

        This method visualizes the AIN instance by plotting the lower, expected, and upper values,
        along with the corresponding alpha and beta levels. The plot includes:
        - Vertical dashed lines at the lower, expected, and upper bounds of the interval.
        - Horizontal solid lines representing the alpha and beta values across the intervals.
        - Dynamic x- and y-axis scaling to ensure clarity.

        Parameters
        ----------
        ain_lw : float, optional
            Line width for the alpha and beta lines. Must be a positive float or integer. Default is 2.0.
        ain_c : str, optional
            Color for the interval lines. Accepts any valid matplotlib color string. Default is 'k' (black).
        ain_label : str, optional
            Label for the alpha and beta interval lines, used in plot legends. Default is an empty string.

        Raises
        ------
        ValueError
            Raised if `ain_lw` is not a positive float or integer.
        TypeError
            Raised if `ain_c` or `ain_label` is not a string.

        Examples
        --------
        >>> ain = AIN(1, 10, 3)
        >>> ain.plot(ain_label='Example')

        Notes
        -----
        - Vertical dashed lines are positioned at the lower, expected, and upper interval bounds.
        - Horizontal solid lines show the alpha level between the lower and expected values, and
          the beta level between the expected and upper values.
        - Y-axis limits are automatically adjusted based on the maximum of alpha and beta, while
          the x-axis extends slightly beyond the interval bounds for readability.
        - The default y-axis label is set to 'pdf', and the default x-axis label displays `ain_label`.
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
        plt.plot([a, a], [0, alpha], **vkw)
        plt.plot([c, c], [0, max(alpha, beta)], **vkw)
        plt.plot([b, b], [0, beta], **vkw)

        hkw = dict(ls='-', lw=ain_lw, c=ain_c)
        plt.plot([a, c], [alpha, alpha], **hkw)
        plt.plot([c, b], [beta, beta], **hkw)

        plt.gca().set_ylim([0, max(alpha, beta) * 1.1])
        plt.gca().set_yticks(sorted([alpha, beta]))
        yticklabels = [f"{alpha:.4f}", f"{beta:.4f}"]
        if alpha > beta:
            yticklabels.reverse()
        plt.gca().set_yticklabels(yticklabels, fontsize=12)

        after_a = a - (b - a) * 0.05
        after_b = b + (b - a) * 0.05
        plt.gca().set_xlim([after_a, after_b])

        plt.gca().set_xticks([a, c, b])
        plt.gca().set_xticklabels([f"{a:.4f}", f"{c:.4f}", f"{b:.4f}"], fontsize=12)

        plt.gca().spines[["top", "right"]].set_visible(False)

        plt.plot(1, 0, ">k", transform=plt.gca().get_yaxis_transform(), clip_on=False)
        plt.plot(after_a, 1, "^k", transform=plt.gca().get_xaxis_transform(), clip_on=False)
        plt.ylabel('$pdf$')
        plt.ylabel('pdf',labelpad=-15)
        plt.xlabel(ain_label)
        plt.show()

    @staticmethod
    def get_global_max(ains_list):
        result = 0
        if not isinstance(ains_list, list):
            raise TypeError("ains_list should be a list")
        for el in ains_list:
            if not isinstance(el, AIN):
                raise TypeError("each el must be a AIN object")
            result = max(result, el.alpha, el.beta)
        return result

    def add_to_plot(self, ain_lw=2.0, ain_c='k', ain_label='', ax=None, global_max=None):
        """
        Plots the intervals and key values of an Asymmetric Interval Number (AIN) instance.

        This method visualizes the AIN instance by plotting its lower, expected, and upper values,
        along with the corresponding alpha and beta levels. The plot includes:
        - Vertical dashed lines at the lower, expected, and upper bounds of the interval.
        - Horizontal solid lines representing the alpha and beta values across the intervals.
        - Dynamic x- and y-axis scaling to ensure clarity, with an optional global maximum for y-axis scaling.

        Parameters
        ----------
        ain_lw : float, optional
            Line width for the alpha and beta interval lines. Must be a positive float or integer. Default is 2.0.
        ain_c : str, optional
            Color for the interval lines. Accepts any valid Matplotlib color string. Default is 'k' (black).
        ain_label : str, optional
            Label for the x-axis to describe the plotted AIN. Default is an empty string.
        ax : matplotlib.axes.Axes, optional
            Matplotlib axis to add the plot to. If not provided, the current axis (`plt.gca()`) is used.
        global_max : float or int, optional
            Maximum value for the y-axis to maintain consistent scaling across multiple AIN plots. If not provided,
            the y-axis is scaled to 1.1 times the maximum of alpha or beta for this AIN instance.

        Raises
        ------
        ValueError
            If `ain_lw` is non-positive or if `global_max` is negative.
        TypeError
            If `ain_lw` or `global_max` are not numeric, or if `ain_c` or `ain_label` are not strings.

        Examples
        --------
        >>> ain = AIN(1, 10, 5)
        >>> ain.add_to_plot(ain_label='Example Interval')

        >>> a = AIN(0, 10, 4)
        >>> b = AIN(0, 10, 7.5)
        >>> gl = AIN.get_global_max([a, b])
        >>> plt.figure(figsize=(8, 3))
        <Figure size 800x300 with 0 Axes>
        >>> plt.subplot(1, 2, 1)
        <Axes: >
        >>> a.add_to_plot(global_max=gl)
        >>> plt.subplot(1, 2, 2)
        <Axes: >
        >>> b.add_to_plot(global_max=gl)
        >>> plt.show()

        Notes
        -----
        - Vertical dashed lines are positioned at the lower, expected, and upper bounds of the interval.
        - Horizontal solid lines represent the alpha level between the lower and expected values,
          and the beta level between the expected and upper values.
        - The y-axis limits are automatically adjusted based on the maximum of alpha and beta values unless
          `global_max` is specified, while the x-axis extends slightly beyond the interval bounds for readability.
        - The default y-axis label is set to 'pdf', and the x-axis label displays `ain_label`.
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

        if global_max is None:
            ax.set_ylim([0, max(alpha, beta) * 1.1])
        else:
            if not isinstance(global_max,(int, float)):
                raise TypeError("global_max must be a float or integer")
            if global_max < 0:
                raise ValueError("global_max must be a positive value")
            ax.set_ylim([0, global_max * 1.1])
        ax.set_ylim([0, global_max * 1.1])
        ax.set_yticks(sorted([alpha, beta]))
        yticklabels = [f"{alpha:.4f}", f"{beta:.4f}"]
        if alpha > beta:
            yticklabels.reverse()
        ax.set_yticklabels(yticklabels, fontsize=12)

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

a = AIN(0,10,4)
b = AIN(0, 10, 7.5)


gl = AIN.get_global_max([a, b])
plt.figure(figsize=(8, 3))
plt.subplot(1, 2, 1)
a.add_to_plot(global_max=gl)
plt.subplot(1, 2, 2)
b.add_to_plot(global_max=gl)
plt.show()
