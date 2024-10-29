The `PyAIN` library introduces a novel and unique approach with **Asymmetric Interval Numbers (AINs)**, combining the simplicity of classical interval numbers with advanced capabilities for modeling uncertainty.

AINs integrate the expected value with the interval, offering a more accurate representation of data uncertainty compared to traditional interval numbers. This library provides a complete toolkit, including basic arithmetic operations. The theoretical foundations of AINs, along with detailed discussions on properties, rigorous mathematical proofs, and theorems on symmetry and asymmetry for both binary and unary operations, are introduced in [1], further enhancing the mathematical framework of AINs. Practical examples illustrate the versatility of AINs in various scientific and technical applications. AINs represent a significant advancement in interval arithmetic, paving the way for further research and applications across diverse fields.


### Installation

You can download and install `pymcdm` library using pip:

```Bash
pip install pymcdm
```

You can run all tests with following command from the root of the project:

```Bash
python -m doctest -v pyain\pyain.py
```

### Example

A simple example demonstrating how to use the library.

```
from pyain import AIN
a = AIN(0, 10, 2)
b = AIN(2, 8, 3)
c = a + b
print(c)
```


### Reference

If the `PyAIN` library has contributed to a scientific publication, we kindly request acknowledgment by citing it.

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


