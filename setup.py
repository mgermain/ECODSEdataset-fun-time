"""Installation script."""
from setuptools import setup, find_packages

with open('requirements/pip.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='ecodse-funtime-alpha',
    python_requires='>3.7',
    version='0.1a',
    description='Super mega happy fun time',
    long_description="Super LONG mega happy fun time",
    url='https://github.com/mgermain/ECODSEdataset-fun-time',
    license='MIT',
    classifiers=[
        'Development Status :: 1 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7'
    ],
    packages=find_packages(exclude=['docs', 'tests']),
    install_requires=requirements,
    extras_require={'static': ['flake8'],
                    'unit': ['pytest', 'pytest-cov', 'pillow', 'codecov']}
)
