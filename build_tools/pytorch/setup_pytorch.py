"""Installation script for flash_ck pytorch extensions."""

import os


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


if __name__ == "__main__":

    # Configure package
    setuptools.setup(
        name="flash_ck_torch",
        version=get_fc_version(),
        description="FlashCK: fast and memory-efficient ck kernel",
        long_description=long_description,
        long_description_content_type="text/markdown",
        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: Unix",
        ],
        ext_modules=ext_modules,
        cmdclass={"build_ext": CUDABuildExtension},
        install_requires=get_install_requirements(),
        tests_require=get_test_requirements(),
    )
