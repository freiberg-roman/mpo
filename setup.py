from pathlib import Path
from setuptools import setup, find_packages


def parse_requirements_file(path):
    return [line.rstrip() for line in open(path, "r")]


reqs_main = parse_requirements_file("requirements/main.txt")
reqs_dev = parse_requirements_file("requirements/dev.txt")

with open("README.md", "r") as f:
    long_description = f.read()


init_str = Path("mpo/__init__.py").read_text()
version = init_str.split("__version__ = ")[1].rstrip().strip('"')

setup(
    name="mpo",
    version=version,
    author="Roman Freiberg",
    description="Implementation of Maximum a Posteriori Policy Optimization Algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/freiberg-roman/mpo",
    packages=find_packages(),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    install_requires=reqs_main,
    extras_require={
        "dev": reqs_main + reqs_dev,
    },
    include_package_data=True,
    python_requires=">=3.9",
    zip_safe=False,
)
