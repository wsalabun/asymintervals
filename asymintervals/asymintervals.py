import numpy as np

class AIN:
    def __init__(self, lower: float, upper: float, expected: float = None):
        """
        Initializes a new instance of the Asymmetric Interval Number (AIN) with specified 'lower' and 'upper'
        bounds, and optionally, an 'expected' value.

        Parameters
        ----------
        lower : float
            The lower bound of the interval ('lower' must be less than 'upper')
        upper : float
            The upper bound of the interval ('upper' must be greater than 'lower')
        expected : float
            The 'expected' value of the interval within the range. If not provided,
            defaults to the average of 'lower' and 'upper'.

        Examples
        ----------
        >>> a = AIN(0,10,8)
        >>> print(a)
        [0.0000, 10.0000]_{8.0000}

        >>> b = AIN(0, 10)
        >>> repr(b)
        'AIN(0, 10, 5.0)'

        >>> c = AIN(1,2,3)
        Traceback (most recent call last):
        ...
        ValueError: It is not a proper AIN 1.0000, 2.0000, 3.0000
        """
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
        Add two AINs (Asymmetric Interval Numbers) or an AIN and a scalar.

        Parameters
        ----------
        other : AIN or scalar
            The AIN or scalar to add to the current AIN instance. If A is an instance
            of AIN, the lower, upper, and expected values of both AINs are added
            component-wise. If A is a scalar, it is added to each component of the
            current AIN.

        Returns
        -------
        AIN
            A new AIN instance representing the result of the addition. The new instance
            will have its lower, upper, and expected values as the sum of the corresponding
            values of the operands.

        Examples
        --------
        >>> a = AIN(1, 10, 8)
        >>> b = AIN(0, 5, 2)
        >>> print(a+b)
        [1.0000, 15.0000]_{10.0000}

        >>> a = AIN(1, 10, 8)
        >>> b = 2
        >>> print(a+b)
        [3.0000, 12.0000]_{10.0000}
        """
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
        Perform the reflected (reversed) addition operation for AIN (Asymmetric Interval Number).

        This method is called when the AIN instance appears on the right-hand side of the addition
        operator and the left-hand operand does not support the addition operation with an AIN instance.
        Essentially, it computes `other + self`.

        Parameters
        ----------
        other : scalar
            The scalar value from which the current AIN instance is to be subtracted.

        Returns
        -------
        AIN
            A new AIN instance representing the result of the addition. The new instance
            will have its lower, upper, and expected values as the sum of `other` and
            the corresponding values of the current AIN instance.

        Examples
        --------
        >>> a = AIN(1, 10, 8)
        >>> b = 5
        >>> print(b + a)
        [6.0000, 15.0000]_{13.0000}
        """
        return self + other

    def __sub__(self, other):
        """
        Subtract an AIN (Asymmetric Interval Number) or a scalar from the current AIN instance.

        Parameters
        ----------
        other : AIN or scalar
            The AIN or scalar to subtract from the current AIN instance. If A is an instance
            of AIN, the lower, upper, and expected values of A are subtracted from the
            upper, lower, and expected values of the current AIN instance, respectively.
            If A is a scalar, it is subtracted from each component of the current AIN.

        Returns
        -------
        AIN
            A new AIN instance representing the result of the subtraction. The new instance
            will have its lower, upper, and expected values as the difference of the corresponding
            values of the operands.

        Examples
        --------
        >>> a = AIN(1, 10, 8)
        >>> b = AIN(0, 5, 2)
        >>> print(a - b)
        [-4.0000, 10.0000]_{6.0000}

        >>> a = AIN(1, 10, 8)
        >>> b = 2
        >>> print(a - b)
        [-1.0000, 8.0000]_{6.0000}
        """
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
        Perform the reflected (reversed) subtraction operation for AIN (Asymmetric Interval Number).

        This method is called when the AIN instance appears on the right-hand side of the subtraction
        operator and the left-hand operand does not support the subtraction operation with an AIN instance.
        Essentially, it computes `other - self`.

        Parameters
        ----------
        other : scalar
            The scalar value from which the current AIN instance is to be subtracted.

        Returns
        -------
        AIN
            A new AIN instance representing the result of the subtraction. The new instance
            will have its lower, upper, and expected values as the difference of `other` and
            the corresponding values of the current AIN instance.

        Examples
        --------
        >>> a = AIN(1, 10, 8)
        >>> b = 5
        >>> print(b - a)
        [-5.0000, 4.0000]_{-3.0000}
        """
        return -self + other

    def __mul__(self, other):
        """
        Multiplies the current instance with another instance of AIN or a numeric value.

        Parameters
        ----------
        other : AIN or scalar
            The value to multiply with. It can be another instance of AIN or a numeric value.


        Returns
        -------
        AIN
            A new instance of AIN representing the result of the multiplication.

        Examples
        --------
        >>> a = AIN(1, 3, 2)
        >>> b = AIN(2, 4, 3)
        >>> print(a * b)
        [2.0000, 12.0000]_{6.0000}

        >>> a = AIN(1, 3, 2)
        >>> b = 2
        >>> print(a * b)
        [2.0000, 6.0000]_{4.0000}
        """
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
        Implements the reverse multiplication operation for the AIN class.

        This method is called when an instance of AIN is multiplied by another object
        and the other object does not have a __mul__ method that supports the instance
        of AIN. This allows for commutative multiplication operations with AIN.

        Parameters
        ----------
        other : scalar
            The value to multiply with the instance of AIN.

        Returns
        -------
        AIN
            A new instance of AIN representing the result of the multiplication.

        Examples
        --------
        >>> a = AIN(1, 3, 2)
        >>> scalar = 2
        >>> result = scalar * a
        >>> print(result)
        [2.0000, 6.0000]_{4.0000}
        """
        return self * other

    def __truediv__(self, other):
        """
        Divides the current instance by another instance of AIN or a scalar value.

        Parameters
        ----------
        other : AIN or scalar
            The value to divide by. It can be another instance of AIN or a scalar value.

        Returns
        -------
        AIN
            A new instance of AIN representing the result of the division.

        Examples
        --------
        >>> a = AIN(4, 8, 6)
        >>> b = AIN(2, 4, 3)
        >>> print(a / b)
        [1.0000, 4.0000]_{2.0794}

        >>> a = AIN(4, 8, 6)
        >>> b = 2
        >>> print(a / b)
        [2.0000, 4.0000]_{3.0000}
        """
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
        Computes the division of a scalar by an Asymmetric Interval Number (AIN) instance.

        This method implements the reverse true division (i.e., `other / self`), where `self`
        is an AIN object and `other` is a scalar. The operation is performed by calculating
        the reciprocal of the AIN and multiplying it by `other`.

        Parameters
        ----------
        other : float or int
            The scalar value to be divided by the AIN.

        Returns
        -------
        AIN
            A new AIN instance representing the result of the division.

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

        This method displays a formatted summary of the AIN object's core attributes, each aligned
        for readability. It includes a validation step for the `precision` parameter to ensure proper
        input, raising an error if the value is not an integer.

        Parameters
        ----------
        precision : int, optional
            The number of decimal places to display for floating-point values (default is 6).
            Must be an integer; otherwise, a ValueError is raised.

        Raises
        ------
        TypeError
            If `precision` is not an integer, a ValueError is raised with an informative message.

        Attributes Printed
        ------------------
        - Alpha       : The `alpha` attribute of the AIN object.
        - Beta        : The `beta` attribute of the AIN object.
        - Assymetry   : The `asymmetry` attribute of the AIN object.
        - Exp. val.   : The expected value of the AIN object.
        - Variance    : The variance of the AIN object.
        - Std. dev.   : The standard deviation of the AIN object.
        - Midpoint    : The midpoint of the AIN object.

        Output Format
        -------------
        Each attribute is displayed in a labeled row, with attribute names left-aligned and values
        right-aligned to match the longest value string for uniform formatting. A header and footer
        frame the summary for clarity.

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

# a = AIN(0, 10, 2)
# a.summary(precision=4)