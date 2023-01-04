import os
from setuptools import setup, find_packages

def src(pth):
    return os.path.join(os.path.dirname(__file__), pth)

# Project description
descr = 'Slope estimation and interpolation of multi-channel seismic data.'

setup(
    name="mcslopes", # Choose your package name
    description=descr,
    long_description=open(src('README.md')).read(),
    keywords=['inverse problems',
              'seismic processing'
              'signal processing'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    author='Matteo Ravasi',
    author_email='matteo.ravasi@kaust.edu.sa',
    install_requires=['numpy >= 1.15.0',
                      'pylops >= 2.0.0'],
    packages=find_packages(),
    use_scm_version=dict(root='.',
                         relative_to=__file__,
                         write_to=src('mcslopes/version.py')),
    setup_requires=['setuptools_scm'],

)
