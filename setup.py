#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


from setuptools import setup, find_packages
import sys

if sys.version_info < (3, 6):
    sys.exit("Sorry, Python >=3.6 is required for EmpatheticDialogues.")

with open("README.md", encoding="utf8") as f:
    readme = f.read()

with open("LICENSE") as f:
    license_ = f.read()

setup(
    name="empchat",
    version="0.1.0",
    description=(
        "PyTorch original implementation of Towards Empathetic Open-domain "
        "Conversation Models: a New Benchmark and Dataset "
        "(https://arxiv.org/abs/1811.00207)."
    ),
    long_description=readme,
    url="https://arxiv.org/abs/1811.00207",
    license=license_,
    python_requires=">=3.6",
    packages=find_packages(),
)
