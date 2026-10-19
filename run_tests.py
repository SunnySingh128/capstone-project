#!/usr/bin/env python3
"""
Test Runner for Subscription Revenue Forecasting Service.

Discovers and executes all unit and integration tests across tests/
Ensures tests run in isolation using temporary directories.
"""

import sys
import unittest


def main():
    print("=" * 70)
    print("Running Subscription Revenue Forecasting Service Test Suite")
    print("=" * 70)

    loader = unittest.TestLoader()
    suite = loader.discover(start_dir="tests", pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("=" * 70)
    if result.wasSuccessful():
        print(f"ALL TESTS PASSED! Total tests run: {result.testsRun}")
        sys.exit(0)
    else:
        print(f"TEST FAILURES DETECTED: {len(result.failures)} failures, {len(result.errors)} errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()
