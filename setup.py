from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(name="Py-Q", 
      version="1.0.0", 
      description="Quantum Circuit Benchmarking", 
      long_description=long_description, 
      long_description_content_type="text/markdown", 
      url="https://github.com/Deftioon/QuantPy", 
      packages=find_packages(), 
      python_requires=">=3.11", )