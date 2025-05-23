# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from setuptools import find_packages, setup


VERSION = "0.0.0.dev"

setup(
    name="cute-kernels",
    version=VERSION,
    author="Mayank Mishra",
    author_email="mayank31398@gmail.com",
    url="https://github.com/mayank31398/cute-kernels",
    packages=find_packages("./"),
    include_package_data=True,
    package_data={"": ["**/*.cu", "**/*.cpp", "**/*.cuh", "**/*.h", "**/*.pyx", "**/*.yml"]},
)
