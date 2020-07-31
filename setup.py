from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md')) as f:
    long_description = f.read()

setup(
    name='wgmm',
    version='0.1.0',
    description='Python Module to Train weighted GMMs using CUDA',
    long_description=long_description,
    url='https://github.com/sahibdhanjal/Weighted-Expectation-Maximization',
    author='Sahib Dhanjal',
    author_email='dhanjalsahib@gmail.com',
    license='MIT',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering'
    ],
    keywords='GMM CUDA',
    packages=find_packages(exclude=['tests']),
    install_requires=['CUDAMat', 'future']
)
