""" Setup file. """

from setuptools import find_packages, setup

with open("README.rst", "r") as readme_file:
    README = readme_file.read()

version = {}
with open("src/edo/version.py", "r") as f:
    exec(f.read(), version)

setup(
    name="edo",
    version=version["__version__"],
    description="Generating artificial datasets through evolution.",
    long_description=README,
    long_description_content_type="text/x-rst",
    url="https://github.com/daffidwilde/edo",
    author="Henry Wilde",
    author_email="henrydavidwilde@gmail.com",
    license="MIT",
    keywords=["evolutionary algorithm", "artificial data", "evolution"],
    packages=find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.5",
    tests_require=["pytest", "pytest-cov", "hypothesis", "numpy"],
)
