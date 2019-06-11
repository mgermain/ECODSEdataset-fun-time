"""Installation script."""
from setuptools import setup, find_packages

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
    install_requires=['tensorflow==1.13', 'matplotlib', 'numpy', 'comet-ml', 'scikit-learn', 'pillow'],
    extras_require={'test': ['flake8', 'pytest', 'pytest-cov', 'pillow', 'codecov']}
)
