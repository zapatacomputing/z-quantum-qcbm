################################################################################
# © Copyright 2020-2022 Zapata Computing Inc.
################################################################################
import site
import sys
import warnings

import setuptools

try:
    from subtrees.z_quantum_actions.setup_extras import extras
except ImportError:
    warnings.warn("Unable to import extras")
    extras = {}

# Workaound for https://github.com/pypa/pip/issues/7953
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="z-quantum-qcbm",
    use_scm_version=True,
    author="Zapata Computing, Inc.",
    author_email="info@zapatacomputing.com",
    description="QCBM package for Orquestra.",
    url="https://github.com/zapatacomputing/z-quantum-qcbm ",
    packages=setuptools.find_namespace_packages(
        include=["zquantum.*"], where="src/python"
    ),
    package_dir={"": "src/python"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=["z-quantum-core"],
    extras_require=extras,
    zip_safe=False,
    package_data={"ops": ["py.typed"]},
)
