# This script aids the installation of the dependencies in requirements.txt

import subprocess
import sys

def run_pip_command(command):
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    print(result.stdout)
    return True

def main():
    # Uninstall any existing Qiskit packages
    run_pip_command("pip uninstall -y qiskit qiskit-terra qiskit-aer qiskit-ibm-runtime qiskit-ibmq-provider qiskit-optimization")
    
    # Parse requirements file
    with open("requirements.txt", "r") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    
    # Filter out standalone qiskit
    requirements = [req for req in requirements if not req == "qiskit"]
    
    # Core dependencies that should be installed first
    core_deps = [req for req in requirements if any(req.startswith(pkg) for pkg in 
                ["numpy", "scipy", "sympy", "symengine", "rustworkx", "dill", 
                 "stevedore", "ply", "python-dateutil", "psutil", "typing-extensions"])]
    
    # Install core dependencies first (without no-cache)
    for dep in core_deps:
        run_pip_command(f"pip install {dep}")
    
    # Install qiskit-terra without dependencies (with no-cache)
    terra_pkg = next((req for req in requirements if req.startswith("qiskit-terra")), "qiskit-terra==0.45.3")
    run_pip_command(f"pip install --no-cache-dir --no-deps {terra_pkg}")
    
    # Install other Qiskit packages without dependencies (with no-cache)
    qiskit_pkgs = [req for req in requirements if req.startswith("qiskit-") and not req.startswith("qiskit-terra")]
    for pkg in qiskit_pkgs:
        run_pip_command(f"pip install --no-cache-dir --no-deps {pkg}")
    
    # Install remaining packages (excluding those already installed, without no-cache)
    installed = core_deps + [terra_pkg] + qiskit_pkgs
    remaining = [req for req in requirements if req not in installed]
    
    for req in remaining:
        # Skip any plain qiskit package
        if req.startswith("qiskit ") or req == "qiskit":
            continue
        run_pip_command(f"pip install {req}")
    
    # Verify installation
    print("\nVerifying Qiskit transpile module...")
    verify_cmd = "python -c \"from qiskit.compiler import transpile; print('Successfully imported transpile!')\""
    run_pip_command(verify_cmd)

if __name__ == "__main__":
    main()