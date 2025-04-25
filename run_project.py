import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a command and print its output"""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*50}\n")
    
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    if process.returncode != 0:
        print(f"\nError: Command failed with return code {process.returncode}")
        sys.exit(1)

def main():
    # Ensure we're in the project root directory
    project_root = Path(__file__).parent.absolute()
    os.chdir(project_root)
    
    # Activate virtual environment
    run_command(
        "source venv/bin/activate",
        "Activating virtual environment"
    )
    
    # Generate test data with smaller samples
    run_command(
        "python3 -c \"from generate_classification_data import generate_all_datasets; generate_all_datasets(n_samples=100)\"",
        "Generating test datasets with smaller samples"
    )
    
    # Run tests
    run_command(
        "python3 -m pytest tests/test_gradient_boosting.py -v",
        "Running gradient boosting tests"
    )
    
    # Run example usage with smaller dataset
    run_command(
        "python3 -c \"from example_usage import main; main(n_samples=200)\"",
        "Running example usage script with smaller dataset"
    )
    
    # Generate visualizations with smaller dataset
    run_command(
        "python3 -c \"from visualize_gradient_boosting import main; main(n_samples=200)\"",
        "Generating visualizations with smaller dataset"
    )
    
    print("\nProject execution completed successfully!")

if __name__ == "__main__":
    main() 