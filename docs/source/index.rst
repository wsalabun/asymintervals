.. asymintervals documentation master file, created by
   sphinx-quickstart on Mon Oct 28 23:29:08 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to asymintervals' documentation!
========================================
..
   .. image:: https://img.shields.io/badge/powered%20by-NumFOCUS-orange.svg?style=flat&colorA=E1523D&colorB=007D8A
     :target: https://numfocus.org
   .. image:: https://img.shields.io/pypi/dm/scipy.svg?label=Pypi%20downloads
     :target: https://pypi.org/project/scipy/
   .. image:: https://img.shields.io/conda/dn/conda-forge/scipy.svg?label=Conda%20downloads
     :target: https://anaconda.org/conda-forge/scipy
   .. image:: https://img.shields.io/badge/stackoverflow-Ask%20questions-blue.svg
     :target: https://stackoverflow.com/questions/tagged/scipy
   .. image:: https://img.shields.io/badge/DOI-10.1038%2Fs41592--019--0686--2-blue.svg
     :target: https://www.nature.com/articles/s41592-019-0686-2

The ``asymintervals`` library introduces a novel and unique approach with **Asymmetric Interval Numbers (AINs)**, combining the simplicity of classical interval numbers with advanced capabilities for modeling uncertainty.

AINs integrate the expected value with the interval, offering a more accurate representation of data uncertainty compared to traditional interval numbers. This library provides a complete toolkit, including basic arithmetic operations. The theoretical foundations of AINs, along with detailed discussions on properties, rigorous mathematical proofs, and theorems on symmetry and asymmetry for both binary and unary operations, are introduced in [1], further enhancing the mathematical framework of AINs. Practical examples illustrate the versatility of AINs in various scientific and technical applications. AINs represent a significant advancement in interval arithmetic, paving the way for further research and applications across diverse fields.

Reference
^^^^^^^^^

If the ``asymintervals`` library has contributed to a scientific publication, we kindly request acknowledgment by citing it.

.. code-block::

   [1] Sałabun, W. (2024). Asymmetric Interval Numbers: a new approach to modeling uncertainty.
       Fuzzy Sets and Systems, (in press).

.. code-block::

   @article{salabun2024,
      title={Asymmetric Interval Numbers: a new approach to modeling uncertainty},
      author={Sałabun, Wojciech},
      journal={Fuzzy sets and systems},
      volume={in press},
      number={in press},
      pages={in press},
      year={2024},
      publisher={Elsevier}
# Print the results of the operations for each interval
   }

Example
^^^^^^^

A simple example demonstrating how to use the library.

.. code-block:: Python

   # Import the AIN (Asymmetric Interval Number) class from the asymintervals module
   from asymintervals import AIN
   import matplotlib.pyplot as plt

   # Initialize two AIN instances with specified lower, upper, and expected values
   a = AIN(0, 10, 2)  # Interval 'a' with lower=0, upper=10, expected=2
   b = AIN(2, 8, 3)   # Interval 'b' with lower=2, upper=8, expected=3

   # Perform arithmetic operations between interval 'a' and interval 'b'
   c = a + b          # Addition of intervals 'a' and 'b'
   d = a * b          # Multiplication of intervals 'a' and 'b'
   e = a / b          # Division of interval 'c' by interval 'd'

   # Plot the resulting intervals from the arithmetic operations
   c.plot()           # Plot interval 'c' resulting from addition
   d.plot()           # Plot interval 'd' resulting from multiplication
   e.plot()           # Plot interval 'e' resulting from division

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


Full class description
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: asymintervals.AIN
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource