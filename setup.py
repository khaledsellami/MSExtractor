# setup.py
import os
import re
from setuptools import setup, find_packages


with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open(os.path.join(os.path.dirname(__file__), "msextractor", "_version.py")) as f:
    version_file = f.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        __version__ = version_match.group(1)
    else:
        raise RuntimeError("Unable to find version string.")


if __name__ == "__main__":
    setup(
        name='msextractor',
        version=__version__,
        packages=find_packages(exclude=['tests']),
        install_requires=requirements,
        python_requires=">=3.8",
        package_data={'msextractor': ['logging.conf']},
        include_package_data=True,
    )