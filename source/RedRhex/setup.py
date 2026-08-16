# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'RedRhex' python package."""

import os
import toml

from setuptools import find_packages, setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "psutil",
    # Sensor V2 simulator/action/export paths import the separately versioned
    # policy-I/O package.  scripts/install_redrhex.py installs that local
    # distribution first; keeping this pin makes any out-of-order install fail
    # during dependency resolution instead of much later at gym.make().
    "redrhex-policy-io==2.0.0",
]

# Installation operation
setup(
    name="RedRhex",
    packages=find_packages(include=("RedRhex", "RedRhex.*")),
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    license="Apache-2.0",
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Isaac Sim :: 4.5.0",
        "Isaac Sim :: 5.0.0",
        "Isaac Sim :: 5.1.0",
    ],
    zip_safe=False,
)
