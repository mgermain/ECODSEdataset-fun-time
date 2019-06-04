"""Installation script."""
from setuptools import setup, find_packages

setup(
    name='ecodse-funtime-alpha',
    python_requires='~=3.6',
    version='0.1a',
    description='Super mega happy fun time',
    long_description="Super LONG mega happy fun time",
    url='https://github.com/mgermain/ECODSEdataset-fun-time',
    license='MIT',
    classifiers=[
        'Development Status :: 1 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6'
    ],
    packages=find_packages(exclude=['docs', 'tests']),
    install_requires=['tf-nightly-2.0-preview', 'matplotlib', 'numpy', 'comet-ml', 'scikit-learn'],
    extras_require={'test': ['flake8', 'pytest', 'pytest-cov']}
)
