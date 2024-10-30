The `asymintervals` library introduces a novel and unique approach with **Asymmetric Interval Numbers (AINs)**, combining the simplicity of classical interval numbers with advanced capabilities for modeling uncertainty.

AINs integrate the expected value with the interval, offering a more accurate representation of data uncertainty compared to traditional interval numbers. This library provides a complete toolkit, including basic arithmetic operations. The theoretical foundations of AINs, along with detailed discussions on properties, rigorous mathematical proofs, and theorems on symmetry and asymmetry for both binary and unary operations, are introduced in [1], further enhancing the mathematical framework of AINs. Practical examples illustrate the versatility of AINs in various scientific and technical applications. AINs represent a significant advancement in interval arithmetic, paving the way for further research and applications across diverse fields.

Documentation is avaliable on [readthedocs](https://asymintervals.readthedocs.io/en/latest/).


### Installation

You can download and install `asymintervals` library using pip:

```Bash
pip install asymintervals
```

You can run all tests with following command from the root of the project:

```Bash
python -m doctest -v asymintervals\asymintervals.py
```

### Example

A simple example demonstrating how to use the library.

```python
# Import the AIN (Asymmetric Interval Number) class from the asymintervals module
from asymintervals import AIN  
import matplotlib.pyplot as plt

# Initialize two AIN instances with specified lower, upper, and expected values
a = AIN(0, 10, 2)  # Interval 'a' with lower=0, upper=10, expected=2
b = AIN(2, 8, 3)   # Interval 'b' with lower=2, upper=8, expected=3

# Perform arithmetic operations between interval 'a' and interval 'b'
c = a + b          # Addition of intervals 'a' and 'b'
d = a * b          # Multiplication of intervals 'a' and 'b'
e = c / d          # Division of interval 'c' by interval 'd'

# Plot the resulting intervals from the arithmetic operations
plt.figure(figsize=(7,2))
plt.subplot(1,3,1)
c.plot()           # Plot interval 'c' resulting from addition
plt.subplot(1,3,2)
d.plot()           # Plot interval 'd' resulting from multiplication
plt.subplot(1,3,3)
e.plot()           # Plot interval 'e' resulting from division
plt.show()

# Print the results of the operations for each interval
print(c)           # Output the details of interval 'c'
print(d)           # Output the details of interval 'd'
print(e)           # Output the details of interval 'e'

# Print summaries for each interval to provide key statistics or characteristics
print("Summary for interval 'a':")
a.summary()
print("Summary for interval 'b':")
b.summary()
print("Summary for interval 'c':")
c.summary()
print("Summary for interval 'd':")
d.summary()
print("Summary for interval 'e':")
e.summary()
```


### Reference

If the `asymintervals` library has contributed to a scientific publication, we kindly request acknowledgment by citing it.

```plaintext
[1] Sałabun, W. (2024). Asymmetric Interval Numbers: a new approach to modeling uncertainty.
    Fuzzy Sets and Systems, (in press).
```

```bibtex
@article{salabun2024,
   title={Asymmetric Interval Numbers: a new approach to modeling uncertainty},
   author={Sałabun, Wojciech},
   journal={Fuzzy Sets and Systems},
   volume={in press},
   number={in press},
   pages={in press},
   year={2024},
   publisher={Elsevier}
}
```


