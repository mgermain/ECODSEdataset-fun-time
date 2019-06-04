"""Installation script."""
from os import path

from setuptools import setup, find_packages

HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, 'README.rst'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

setup(
    name='ecodse-funtime-alpha',
    version='0.1a',
    description='Super mega happy fun time',
    long_description=LONG_DESCRIPTION,
    url='https://github.com/mgermain/ECODSEdataset-fun-time',
    license='MIT',
    classifiers=[
        'Development Status :: 1 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7'
    ],
    packages=find_packages(exclude=['docs', 'tests']),
    install_requires=['tensorflow', 'matplotlib', 'numpy'],
    extras_require={
        'test': ['flake8', 'pytest', 'codecov',
                 'pytest-cov', 'pydocstyle'],
    }
)