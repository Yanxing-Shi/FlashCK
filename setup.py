"""Installation script."""

from build_tools.build_ext import CMakeExtension, get_build_ext
from build_tools.version import version
from build_tools.utils import (
    found_cmake,
    found_ninja)

def setup_requirements() -> Tuple[List[str], List[str], List[str]]:
    """Setup Python dependencies

    Returns dependencies for build, runtime, and testing.
    """

    # Common requirements
    setup_reqs: List[str] = []
    install_reqs: List[str] = [
        "pydantic",
        "importlib-metadata>=1.0",
        "packaging",
    ]
    test_reqs: List[str] = ["pytest>=8.2.1"]

    # Requirements that may be installed outside of Python
    if not found_cmake():
        setup_reqs.append("cmake>=3.21")
    if not found_ninja():
        setup_reqs.append("ninja")
    if not found_pybind11():
        setup_reqs.append("pybind11")

    # Framework-specific requirements
    if not bool(int(os.getenv("NVTE_RELEASE_BUILD", "0"))):
        if "pytorch" in frameworks:
            install_reqs.extend(["torch>=2.1"])
            test_reqs.extend(["numpy", "torchvision", "prettytable", "PyYAML"])

    return [remove_dups(reqs) for reqs in [setup_reqs, install_reqs, test_reqs]]


if __name__ == "__main__":
    __version__ = te_version()

    # Configure package
    setuptools.setup(
        name="FlashCK",
        version=__version__,
        packages=setuptools.find_packages(
            include=[
                "transformer_engine",
                "transformer_engine.*",
                "transformer_engine/build_tools",
            ],
        ),
        extras_require=extras_require,
        description="Transformer acceleration library",
        long_description=long_description,
        long_description_content_type="text/x-rst",
        ext_modules=ext_modules,
        cmdclass={"build_ext": CMakeBuildExtension, "bdist_wheel": TimedBdist},
        python_requires=">=3.8, <3.13",
        classifiers=[
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
        ],
        setup_requires=setup_requires,
        install_requires=install_requires,
        include_package_data=include_package_data,
        package_data=package_data,
    )