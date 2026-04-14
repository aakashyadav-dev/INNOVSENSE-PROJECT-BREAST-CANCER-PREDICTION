"""
Test script to verify all dependencies are installed correctly
"""

import sys
import importlib

def test_import(module_name):
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name} - OK")
        return True
    except ImportError as e:
        print(f"❌ {module_name} - Failed: {e}")
        return False

print("=" * 50)
print("Testing Python Environment")
print("=" * 50)
print(f"Python version: {sys.version}")
print()

# Test core packages
packages = [
    "fastapi",
    "uvicorn",
    "sqlalchemy",
    "pydantic",
    "pandas",
    "numpy",
    "sklearn",
    "xgboost",
    "joblib",
    "requests",
    "matplotlib",
    "seaborn"
]

all_good = True
for package in packages:
    if not test_import(package):
        all_good = False

print("\n" + "=" * 50)
if all_good:
    print("✅ All packages installed successfully!")
    print("\nYou can now run:")
    print("  Backend: uvicorn app.main:app --reload")
    print("  Frontend: streamlit run streamlit_app.py")
else:
    print("❌ Some packages failed to install. Please check the errors above.")
print("=" * 50)