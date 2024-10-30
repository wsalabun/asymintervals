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
        return other * self**(-1)

    def __pow__(self, n):
        """
        Compute the power of an Asymmetric Interval Number (AIN) raised to the power of `n`.

        Parameters
        ----------
        n : int or float
            The exponent to raise the AIN to.

        Raises
        ------
        ValueError
            If the resulting operation yields a complex number due to an invalid exponent, or if `n = -1`
            and the interval includes 0, as division by zero is undefined for this operation.

        Returns
        -------
        AIN
            A new AIN object representing the result of the power operation.

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
        Computes the probability density function (PDF) value for the Asymmetric Interval Number (AIN)
        at a specified point x.

        The PDF is defined piecewise for the AIN interval:
        - Returns 0 if x is outside the interval defined by lower and upper.
        - Returns alpha if x is between lower and expected.
        - Returns beta if x is between expected and upper.

        Parameters
        ----------
        x : float
            The point at which to evaluate the PDF.

        Returns
        -------
        float
            The PDF value at the specified point x. Returns 0 if x is outside the interval.

        Examples
        --------
        >>> a = AIN(0, 10, 5)
        >>> print(a.pdf(-1))
        0
        >>> print(a.pdf(3))
        0.1
        >>> print(a.pdf(7))
        0.1
        >>> print(a.pdf(11))
        0
        """
        if x < self.lower:
            return 0
        elif x < self.expected:
            return self.alpha
        elif x < self.upper:
            return self.beta
        else:
            return 0

    def cdf(self, x):
        """
        Computes the cumulative distribution function (cdf) value for the AIN instance at a given value of x.

        Parameters
        ----------
        x : scalar
            The value at which to evaluate the cdf.

        Returns
        -------
        float
            The CDF value at x. The result will be:
            - 0 if x is less than the lower bound.
            - A linear interpolation between the lower bound and the expected value if x is between the lower bound and the expected value.
            - A linear interpolation between the expected value and the upper bound if x is between the expected value and the upper bound.
            - 1 if x is greater than or equal to the upper bound.

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
        Computes the inverse cumulative distribution function (quantile function) for the AIN instance
        at a given value of y. This function is the inverse of the CDF.

        Parameters
        ----------
        y : scalar
            The probability value for which to compute the quantile. It should be in the range [0, 1].

        Returns
        -------
        float or None
            The quantile value corresponding to the given probability y. If y is outside the range [0, 1],
            the method returns None and prints an error message.

        Raises
        ------
        ValueError
            If y is outside the range [0, 1].

        Examples
        --------
        >>> a = AIN(0, 10, 3)
        >>> print(a.quantile(0.25))
        1.0714285714285714
        >>> print(a.quantile(0.85))
        6.5
        >>> print(a.quantile(1.1))
        Traceback (most recent call last):
        ...
        ValueError: Argument y = 1.1 is out of range; it should be between 0 and 1.
        """
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
        ValueError
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
            raise ValueError(f'Argument precision = {precision} but it must be an integer.')
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

a = AIN(0, 10, 2)
a.summary(precision=4)