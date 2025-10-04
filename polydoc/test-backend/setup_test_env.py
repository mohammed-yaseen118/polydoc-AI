#!/usr/bin/env python3
"""
PolyDoc Test Backend Setup Script
=================================

This script helps set up the test environment and verifies everything is working correctly.
"""

import subprocess
import sys
import os
from pathlib import Path
import venv

def run_command(cmd, description=""):
    """Run a command and return success status"""
    try:
        if description:
            print(f"🔧 {description}")
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Success: {description or cmd}")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Failed: {description or cmd}")
            if result.stderr.strip():
                print(f"   Error: {result.stderr.strip()}")
            return False
            
    except Exception as e:
        print(f"❌ Exception during: {description or cmd}")
        print(f"   Error: {e}")
        return False

def check_python_version():
    """Check if Python version is adequate"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major >= 3 and version.minor >= 7:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is supported")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is too old. Requires Python 3.7+")
        return False

def create_virtual_environment():
    """Create a virtual environment if requested"""
    current_dir = Path(__file__).parent
    venv_path = current_dir / "test_env"
    
    if venv_path.exists():
        print(f"ℹ️  Virtual environment already exists at {venv_path}")
        return True
    
    try:
        print("🔧 Creating virtual environment...")
        venv.create(venv_path, with_pip=True)
        print(f"✅ Virtual environment created at {venv_path}")
        print(f"   To activate: {venv_path / 'Scripts' / 'activate'} (Windows) or source {venv_path / 'bin' / 'activate'} (Linux/Mac)")
        return True
    except Exception as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    return run_command(
        f'pip install -r "{requirements_file}"',
        "Installing Python packages"
    )

def run_framework_test():
    """Run the framework verification test"""
    print("🧪 Running framework verification test...")
    
    test_script = Path(__file__).parent / "test_framework.py"
    if not test_script.exists():
        print("❌ test_framework.py not found")
        return False
    
    return run_command(
        f'python "{test_script}"',
        "Running framework test"
    )

def run_basic_ml_test():
    """Run a basic ML test to ensure everything works"""
    print("🤖 Running basic ML test...")
    
    run_tests_script = Path(__file__).parent / "run_tests.py"
    if not run_tests_script.exists():
        print("❌ run_tests.py not found")
        return False
    
    # First try with mock mode to see if basic functionality works
    return run_command(
        f'python "{run_tests_script}" --test basic',
        "Running basic ML pipeline test"
    )

def main():
    """Main setup function"""
    print("=" * 60)
    print("🚀 PolyDoc Test Backend Setup")
    print("=" * 60)
    
    # Step 1: Check Python version
    if not check_python_version():
        print("\n💥 Setup failed: Python version incompatible")
        sys.exit(1)
    
    # Step 2: Ask about virtual environment
    create_venv = input("\n❓ Create virtual environment? (recommended) [y/N]: ").lower().strip()
    if create_venv in ['y', 'yes']:
        if not create_virtual_environment():
            print("\n⚠️  Virtual environment creation failed, continuing with system Python")
    
    # Step 3: Install dependencies
    if not install_dependencies():
        print("\n💥 Setup failed: Could not install dependencies")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ Installation completed! Running verification tests...")
    print("=" * 60)
    
    # Step 4: Run framework test
    framework_test_passed = run_framework_test()
    
    # Step 5: Run basic ML test (may fail if full backend not available, but that's OK)
    print("\n" + "=" * 40)
    print("🤖 Testing ML Pipeline (may use mock mode)")
    print("=" * 40)
    ml_test_passed = run_basic_ml_test()
    
    # Final summary
    print("\n" + "=" * 60)
    print("📊 Setup Summary")
    print("=" * 60)
    print(f"✅ Framework Test: {'Passed' if framework_test_passed else 'Failed'}")
    print(f"🤖 ML Pipeline Test: {'Passed' if ml_test_passed else 'Failed (may be expected)'}")
    
    if framework_test_passed:
        print("\n🎉 Setup completed successfully!")
        print("\n📚 Next steps:")
        print("  1. Run 'python test_framework.py' to verify the framework")
        print("  2. Run 'python run_tests.py --test basic' to test ML pipeline")
        print("  3. Check README.md for more detailed usage instructions")
        
        if create_venv in ['y', 'yes']:
            venv_path = Path(__file__).parent / "test_env"
            if os.name == 'nt':  # Windows
                print(f"  4. Activate venv with: {venv_path / 'Scripts' / 'activate'}")
            else:  # Linux/Mac
                print(f"  4. Activate venv with: source {venv_path / 'bin' / 'activate'}")
    else:
        print("\n💥 Setup encountered issues. Check the output above for details.")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Setup interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Setup failed with error: {e}")
        sys.exit(1)