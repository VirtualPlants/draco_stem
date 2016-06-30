#!/usr/bin/env python
# -*- coding: utf-8 -*-

# {# pkglts, pysetup.kwds
# format setup arguments

from setuptools import setup, find_packages


short_descr = "Generating high-quality meshes of cell tissue from 3D segmented images"
readme = open('README.rst').read()
history = open('HISTORY.rst').read().replace('.. :changelog:', '')


def parse_requirements(fname):
    with open(fname, 'r') as f:
        txt = f.read()

    reqs = []
    for line in txt.splitlines():
        line = line.strip()
        if len(line) > 0 and not line.startswith("#"):
            reqs.append(line)

    return reqs

# find version number in src/openalea/draco_stem/version.py
version = {}
with open("src/openalea/draco_stem/version.py") as fp:
    exec(fp.read(), version)


setup_kwds = dict(
    name='openalea.draco_stem',
    version=version["__version__"],
    description=short_descr,
    long_description=readme + '\n\n' + history,
    author="Guillaume Cerutti, ",
    author_email="guillaume.cerutti@inria.fr, ",
    url='https://github.com/Guillaume Cerutti/draco_stem',
    license='cecill-c',
    zip_safe=False,

    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=parse_requirements("requirements.txt"),
    tests_require=parse_requirements("dvlpt_requirements.txt"),
    entry_points={},
    keywords='',
    test_suite='nose.collector',
)
# #}
# change setup_kwds below before the next pkglts tag

# do not change things below
# {# pkglts, pysetup.call
setup(**setup_kwds)
# #}
