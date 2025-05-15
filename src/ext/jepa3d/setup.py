from setuptools import setup, find_packages

setup(
  name="jepa3d",
  version="0.1",              # or match upstream version
  packages=find_packages("models"),
  package_dir={"": "models"},
)