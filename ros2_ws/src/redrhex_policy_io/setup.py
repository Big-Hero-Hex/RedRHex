"""Ament adapter that installs the canonical source/redrhex_policy_io package."""

from pathlib import Path

from setuptools import setup


package_name = "redrhex_policy_io"
adapter_root = Path(__file__).resolve().parent
repository_root = adapter_root.parents[2]
canonical_package = (
    repository_root / "source" / "redrhex_policy_io" / package_name
)
canonical_package_relative = Path("../../../source/redrhex_policy_io") / package_name
if not (canonical_package / "__init__.py").is_file():
    raise RuntimeError(
        "canonical source/redrhex_policy_io/redrhex_policy_io package is missing"
    )


setup(
    name="redrhex-policy-io",
    version="2.0.0",
    packages=[package_name],
    package_dir={package_name: canonical_package_relative.as_posix()},
    package_data={package_name: ["data/*.json"]},
    include_package_data=False,
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [f"resource/{package_name}"],
        ),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["numpy>=1.23"],
    zip_safe=True,
    maintainer="RedRHex Team",
    maintainer_email="redrhex@example.com",
    description="ROS adapter for the canonical RedRHex policy I/O package.",
    license="BSD-3-Clause",
)
