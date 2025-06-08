#!/usr/bin/env python3
"""
Script to update version number in both pyproject.toml and __init__.py
Used by semantic-release during automated versioning.
"""

import sys
import re
import tomlkit


def update_version(new_version):
    """Update version in pyproject.toml and __init__.py"""
    print(f"Updating version to {new_version}")
    
    # Update pyproject.toml
    with open("pyproject.toml", "r") as f:
        pyproject = tomlkit.parse(f.read())
    
    pyproject["project"]["version"] = new_version
    
    with open("pyproject.toml", "w") as f:
        f.write(tomlkit.dumps(pyproject))
        
    # Update __init__.py fallback version
    init_path = "src/gcode_simulator/__init__.py"
    with open(init_path, "r") as f:
        content = f.read()
        
    # Match any version string in the fallback version assignment
    updated_content = re.sub(
        r'__version__ = "[^"]+"',
        f'__version__ = "{new_version}"',
        content
    )
    
    with open(init_path, "w") as f:
        f.write(updated_content)
        
    print(f"Version updated successfully to {new_version}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: update_version.py <new_version>")
        sys.exit(1)
    
    update_version(sys.argv[1])
