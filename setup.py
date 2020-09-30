import setuptools
import os

setuptools.setup(
    name="z-quantum-qcbm",
    version="0.2.0",
    author="Zapata Computing, Inc.",
    author_email="info@zapatacomputing.com",
    description="QCBM package for Orquestra.",
    url="https://github.com/zapatacomputing/z-quantum-qcbm ",
    packages=["zquantum.qcbm"],
    package_dir={"": "src/python"},
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
    install_requires=[
        "z-quantum-core",
    ],
)
