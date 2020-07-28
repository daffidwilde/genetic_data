""" Setup file. """

from setuptools import find_packages, setup

with open("README.rst", "r") as readme:
    README = readme.read()

version = {}
with open("src/edo/version.py", "r") as vers:
    exec(vers.read(), version)

requirements = []
with open("requirements.txt", "r") as reqs:
    for line in reqs.read().splitlines():
        if not line.startswith("#"):
            requirements.append(line)

setup(
    name="edo",
    version=version["__version__"],
    description="Generating artificial datasets through evolution.",
    long_description=README,
    long_description_content_type="text/x-rst",
    url="https://github.com/daffidwilde/edo",
    author="Henry Wilde, Vincent Knight, Jonathan Gillard",
    author_email="henrydavidwilde@gmail.com",
    license="MIT",
    keywords=["evolutionary algorithm", "artificial data", "evolution"],
    packages=find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.6",
    install_requires=requirements,
    tests_require=["pytest", "pytest-cov", "hypothesis", "numpy"],
)
