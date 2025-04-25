import pytest
import sys
import os

def run_test(test_name):
    """Run a single test and return its result"""
    result = pytest.main([f"tests/test_gradient_boosting.py::{test_name}", "-v"])
    return result == 0

def main():
    tests = [
        "test_toy_data",
        "test_noisy_data",
        "test_collinear_data",
        "test_imbalanced_data",
        "test_nonlinear_data",
        "test_edge_cases",
        "test_probability_outputs"
    ]
    
    print("Running tests...\n")
    results = {}
    
    for test in tests:
        print(f"Running {test}...")
        success = run_test(test)
        results[test] = success
        print(f"{'✓' if success else '✗'} {test}\n")
    
    print("\nTest Results:")
    print("------------")
    for test, success in results.items():
        print(f"{test}: {'PASS' if success else 'FAIL'}")
    
    print(f"\nTotal: {sum(results.values())}/{len(results)} tests passed")

if __name__ == "__main__":
    main() 