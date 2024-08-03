# K-Means Algorithm Test Suite

This repository contains comprehensive test suites for the K-Means clustering algorithm implementation. These tests are designed to ensure the correctness and robustness of both the C module and the Python implementation.

## Test Files

1. `tests_for_c_module.py`: Tests for the C module implementation
2. `unittest_for_pyfile.py`: Unit tests for the Python implementation

## tests_for_c_module.py

This file contains tests specifically designed for the C module of the K-Means algorithm. It includes various test cases to verify the functionality, convergence, and edge cases of the C implementation.

**Important Note:** The tests in this file are dependent on the specific implementation of the C module. Users may need to adjust input parameters and function calls based on their particular C module structure.

Key features:
- Tests basic functionality
- Verifies convergence
- Checks behavior with different dimensions and cluster numbers
- Includes edge cases and invalid input handling

## unittest_for_pyfile.py

This file contains unit tests for the Python implementation of the K-Means algorithm. It uses Python's unittest framework to provide a structured testing environment.

Key features:
- Comprehensive test cases for various scenarios
- Tests for different dimensions, cluster numbers, and dataset sizes
- Verification of convergence and stability
- Edge case handling

## Usage

To run the tests:

1. Ensure you have the necessary dependencies installed.
2. For C module tests: python3 tests_for_c_module.py
3. For Python implementation unit tests: python3 unittest_for_pyfile.py


## Contributing

Feel free to use these tests, adapt them to your specific implementation, and contribute additional test cases to improve the robustness of the K-Means algorithm testing.


