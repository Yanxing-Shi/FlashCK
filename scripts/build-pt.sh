#!/bin/bash

set -e

# clean
rm -rf build/ dist/ *.egg-info __pycache__ .pytest_cache

# install
pip install -e . --no-build-isolation