"""Installation script."""

from build_tools.build_ext import CMakeExtension, get_build_ext
from build_tools.version import version
from build_tools.utils import (
    found_cmake,
    found_ninja)
