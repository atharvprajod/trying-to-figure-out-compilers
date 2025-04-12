#!/usr/bin/env python3
"""
Setup script for RL-Fusion package.
"""

from setuptools import setup, find_packages

setup(
    name="rl-fusion",
    version="0.1.0",
    description="Reinforcement Learning for Operator Fusion in TVM",
    author="AI Assistant",
    author_email="example@example.com",
    url="https://github.com/example/rl-fusion",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.20.0",
        "torch>=1.10.0",
        "torch-geometric>=2.0.0",
        "gym>=0.21.0",
        "stable-baselines3>=1.5.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.62.0",
        "networkx>=2.6.0",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 