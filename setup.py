#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

# To update the package version number, edit egp/__version__.py
version = {}
with open(os.path.join(here, 'egp', '__version__.py')) as f:
    exec(f.read(), version)

with open('README.rst') as readme_file:
    readme = readme_file.read()

setup(
    name='egp',
    version=version['__version__'],
    description="Python code used and developed for my cosmology research",
    long_description=readme + '\n\n',
    author="E. G. Patrick Bos",
    author_email='p.bos@esciencecenter.nl',
    url='https://github.com/egpbos/egp',
    packages=[
        'egp',
    ],
    package_dir={'egp':
                 'egp'},
    include_package_data=True,
    license="Apache Software License 2.0",
    zip_safe=False,
    keywords='egp',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    test_suite='tests',
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib'
    ],
    setup_requires=[
        # dependency for `python setup.py test`
        'pytest-runner',
        # dependencies for `python setup.py build_sphinx`
        'sphinx',
        'sphinx_rtd_theme',
        'recommonmark'
    ],
    tests_require=[
        'pytest',
        'pytest-cov',
        'pycodestyle',
    ],
    extras_require={
        'dev':  ['prospector[with_pyroma]', 'yapf', 'isort'],
        'iconstrain': ['FuncDesigner', 'openopt', 'mayavi'],
        # note: crunch will be replaced by egpcrunch, which is based on pybind11 instead of pyublas
        'crunch': ['pyublas'],
        # set of random testing scripts used during research:
        'testing': ['mayavi']
    }
)
