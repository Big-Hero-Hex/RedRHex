from glob import glob
from setuptools import find_packages, setup

package_name = "redrhex_five_leg_mode"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Jason Liao",
    maintainer_email="jason@example.com",
    description="Five-leg RedRhex policy mode for running with Rinbo L3 disabled.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "five_leg_rl_controller = redrhex_five_leg_mode.five_leg_controller_node:main",
        ],
    },
)
