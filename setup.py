"""
Setup script for the GFlowNet package.

This module configures the installation of the gflownet package using setuptools.
It handles version management by either using an environment variable or 
automatically determining the next patch version based on git tags.

The setup is minimal and relies on pyproject.toml for most configuration details.
Package dependencies, metadata, and build settings are defined there.
"""

import os
from ast import literal_eval
from subprocess import check_output  # nosec - command is hard-coded, no possibility of injection

from setuptools import setup


def _get_next_version():
    """
    Determine the next version number for the package.
    
    This function follows a two-stage approach:
    1. If SEMVER environment variable is set (typically in CI/CD), use that version
    2. Otherwise, auto-increment the patch version based on existing git tags
    
    For development builds, it reads the major.minor from VERSION file and finds
    the highest patch number in git tags, then increments by 1.
    
    Returns:
        str: Version string in format "major.minor.patch"
        
    Raises:
        ValueError: If no tags exist for the current major.minor combination
    """
    # Use environment variable if available (for production releases)
    if "SEMVER" in os.environ:
        return os.environ.get("SEMVER")

    # Auto-generate version for development builds
    # Note: This should only be used for development builds. Production releases
    # should have the SEMVER variable loaded at runtime by deployment robots.
    with open("VERSION", "r") as f:
        lines = f.read().splitlines()
    
    # Parse VERSION file format (e.g., "MAJOR=1\nMINOR=2")
    version_parts = {k: literal_eval(v) for k, v in map(lambda x: x.split("="), lines)}
    major = int(version_parts["MAJOR"])
    minor = int(version_parts["MINOR"])
    
    # Get all git tags to find the latest patch version
    versions = check_output(["git", "tag", "--list"], encoding="utf-8").splitlines()  # nosec - command is hard-coded
    try:
        # Find highest patch number for current major.minor
        latest_patch = max(int(v.rsplit(".", 1)[1]) for v in versions if v.startswith(f"v{major}.{minor}."))
    except ValueError:  # no tags for this major.minor exist yet
        latest_patch = -1
    
    return f"{major}.{minor}.{latest_patch+1}"

# Configure the package with auto-generated version
# Most configuration is handled in pyproject.toml
setup(name="gflownet", version=_get_next_version())
