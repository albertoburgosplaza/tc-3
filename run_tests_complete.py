#!/usr/bin/env python3
"""
Complete test suite runner for the object detection service.

This script runs all tests (unit, integration, performance) and generates
comprehensive coverage reports. It ensures coverage meets the 80% requirement.

Usage:
    python run_tests_complete.py
    python run_tests_complete.py --quick  # Skip performance tests for faster execution
    python run_tests_complete.py --coverage-only  # Only run tests needed for coverage
"""

import argparse
import subprocess
import sys
import os
import time
from pathlib import Path


class TestSuiteRunner:
    """Complete test suite runner with coverage reporting."""

    def __init__(self):
        self.start_time = time.time()
        self.results = {}

    def run_command(self, cmd: list, description: str) -> bool:
        """Run a command and capture its result."""
        print(f"\n{'='*60}")
        print(f"🔄 {description}")
        print(f"{'='*60}")
        print(f"Command: {' '.join(cmd)}")
        print()

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            self.results[description] = "✅ PASSED"
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ FAILED: {description}")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
            print(f"Return code: {e.returncode}")
            self.results[description] = "❌ FAILED"
            return False

    def check_prerequisites(self) -> bool:
        """Check that all prerequisites are met."""
        print("🔍 Checking prerequisites...")
        
        # Check if we're in the right directory
        if not Path("pytest.ini").exists():
            print("❌ pytest.ini not found. Run this script from the project root.")
            return False
        
        if not Path("app").exists():
            print("❌ app directory not found. This should be the project root.")
            return False
        
        if not Path("tests").exists():
            print("❌ tests directory not found.")
            return False

        print("✅ All prerequisites met.")
        return True

    def run_unit_tests(self) -> bool:
        """Run unit tests."""
        cmd = ["pytest", "tests/unit/", "-v", "-m", "unit", "--tb=short"]
        return self.run_command(cmd, "Running Unit Tests")

    def run_integration_tests(self) -> bool:
        """Run integration tests."""
        cmd = ["pytest", "tests/integration/", "-v", "-m", "integration", "--tb=short"]
        return self.run_command(cmd, "Running Integration Tests")

    def run_performance_tests(self) -> bool:
        """Run performance tests."""
        cmd = ["pytest", "tests/performance/", "-v", "-m", "performance", "--tb=short"]
        return self.run_command(cmd, "Running Performance Tests")

    def run_coverage_tests(self) -> bool:
        """Run all tests with coverage reporting."""
        cmd = [
            "pytest", 
            "tests/", 
            "-v", 
            "--cov=app", 
            "--cov-report=html:htmlcov",
            "--cov-report=xml",
            "--cov-report=term-missing",
            "--cov-fail-under=80",
            "--tb=short"
        ]
        return self.run_command(cmd, "Running Full Test Suite with Coverage")

    def run_quick_coverage_tests(self) -> bool:
        """Run unit and integration tests with coverage (skip performance)."""
        cmd = [
            "pytest", 
            "tests/unit/", 
            "tests/integration/",
            "-v", 
            "--cov=app", 
            "--cov-report=html:htmlcov",
            "--cov-report=xml",
            "--cov-report=term-missing",
            "--cov-fail-under=80",
            "--tb=short"
        ]
        return self.run_command(cmd, "Running Unit & Integration Tests with Coverage")

    def generate_summary_report(self, mode: str):
        """Generate a summary report of all test results."""
        end_time = time.time()
        duration = end_time - self.start_time

        print(f"\n{'='*60}")
        print(f"📊 TEST SUITE SUMMARY REPORT ({mode.upper()} MODE)")
        print(f"{'='*60}")
        print(f"⏱️  Total execution time: {duration:.2f} seconds")
        print()

        # Results summary
        passed = sum(1 for result in self.results.values() if "PASSED" in result)
        total = len(self.results)
        
        print(f"📈 Results: {passed}/{total} test categories passed")
        print()

        for description, result in self.results.items():
            print(f"  {result} {description}")

        print()
        
        # Coverage report location
        if Path("htmlcov").exists():
            print("📋 Coverage Reports Generated:")
            print(f"  📄 HTML Report: file://{Path.cwd().absolute()}/htmlcov/index.html")
            print(f"  📄 XML Report:  {Path.cwd().absolute()}/coverage.xml")
        
        print()
        
        # Overall status
        if passed == total:
            print("🎉 ALL TEST CATEGORIES PASSED!")
            print("✅ Test suite execution completed successfully.")
            return True
        else:
            print(f"❌ {total - passed} test categories failed.")
            print("🔧 Review the failed tests and fix issues before deployment.")
            return False

def main():
    """Main function to run the complete test suite."""
    parser = argparse.ArgumentParser(description="Run complete test suite for object detection service")
    parser.add_argument("--quick", action="store_true", 
                       help="Skip performance tests for faster execution")
    parser.add_argument("--coverage-only", action="store_true",
                       help="Only run tests needed for coverage (unit + integration)")
    
    args = parser.parse_args()

    runner = TestSuiteRunner()

    print("🧪 Object Detection Service - Complete Test Suite Runner")
    print("=" * 60)

    # Check prerequisites
    if not runner.check_prerequisites():
        sys.exit(1)

    success = True

    if args.coverage_only:
        # Coverage-only mode: just unit + integration with coverage
        print("\n🎯 Running in COVERAGE-ONLY mode")
        success = runner.run_quick_coverage_tests()
        runner.generate_summary_report("coverage-only")
        
    elif args.quick:
        # Quick mode: all tests individually, then coverage without performance
        print("\n⚡ Running in QUICK mode (skipping performance tests)")
        
        success &= runner.run_unit_tests()
        success &= runner.run_integration_tests()
        success &= runner.run_quick_coverage_tests()
        
        runner.generate_summary_report("quick")
        
    else:
        # Full mode: all tests individually, then complete coverage
        print("\n🔥 Running in FULL mode (all tests including performance)")
        
        success &= runner.run_unit_tests()
        success &= runner.run_integration_tests()
        success &= runner.run_performance_tests()
        success &= runner.run_coverage_tests()
        
        runner.generate_summary_report("full")

    # Exit with appropriate code
    if success:
        print("\n🏆 Test suite completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Test suite failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()