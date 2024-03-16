# -*- coding: utf-8 -*-

from __future__ import print_function

from setuptools import setup
import sys, re, os, pathlib

this_directory = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="umbra",
    version="0.0.5",
    author="Markus Worchel",
    author_email="m.worchel@tu-berlin.de",
    description="UMBRA: A Concurrent, Interactive 3D Viewer for Python",
    url="https://github.com/mworchel/umbra",
    license="MIT",
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=['umbra'],
    install_requires=["glfw>=2.5.0", "moderngl>=5.6.0", "pyopengl>=3.1.6", "numpy>=1.22.0", "imgui>=1.4.1"],
    python_requires=">=3.8"
)