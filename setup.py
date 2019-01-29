""" Setup script. """

from setuptools import find_packages, setup

with open("README.rst", "r") as readme_file:
    README = readme_file.read()

setup(
    name="edo",
    version="0.0.1a",
    description="A library for generating artificial datasets through genetic \
    evolution.",
    long_description=README,
    url="https://github.com/daffidwilde/edo",
    author="Henry Wilde",
    author_email="henrydavidwilde@gmail.com",
    license="MIT",
    keywords=["genetic-algorithm" "data" "evolution"],
    packages=find_packages("src"),
    package_dir={"": "src"},
)
