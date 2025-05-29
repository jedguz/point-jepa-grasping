# setup.py
from setuptools import setup, find_packages

setup(
    name="ADLR",
    version="0.1.0",
    python_requires=">=3.8",
    # finds packages under src/, including ext/jepa3d & ext/pointnet2
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=1.12",
        # add any other global deps here
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
