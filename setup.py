# setup.py
from setuptools import setup, find_packages

from msextractor import __version__


with open('requirements.txt') as f:
    requirements = f.read().splitlines()

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