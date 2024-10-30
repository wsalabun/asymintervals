.. asymintervals documentation master file, created by
   sphinx-quickstart on Mon Oct 28 23:29:08 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to asymintervals' documentation!
=================================

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
   }

Example
^^^^^^^

A simple example demonstrating how to use the library.

.. code-block:: Python

   from asymintervals import AIN
   a = AIN(0, 10, 2)
   b = AIN(2, 8, 3)
   c = a + b
   print(c)

Full class description
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: asymintervals.AIN
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource