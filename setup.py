from setuptools import setup, find_packages
import os

# Read dependencies from requirements.txt
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    with open(req_path, encoding="utf-8") as f:
        return f.read().splitlines()

# Find all packages in src/ and ur3e_grasp/src/
packages = find_packages(where="src")
# ur3e_packages = find_packages(where="ur3e_grasp/src", include=["ur3e_grasp*"])
# packages.extend(ur3e_packages)

setup(
    name="disf_ras",
    version="0.0.0",
    # packages=packages,
    # package_dir={
    #     "": "src",
    #     "ur3e_grasp": "ur3e_grasp/src/ur3e_grasp",
    # },
    include_package_data=True,
    install_requires=read_requirements(),
)
