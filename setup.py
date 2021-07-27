import setuptools

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
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
    setup_requires=["setuptools_scm~=6.0"],
    install_requires=["z-quantum-core"],
)
