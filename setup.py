# setup.py
from setuptools import setup, find_packages

setup(
    name="ADLR",
    version="0.1.0",
    python_requires=">=3.8",
    # finds packages under src/, including ext/jepa3d & ext/pointnet2
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
