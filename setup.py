#   -*- coding: utf-8 -*-
#
#  --------------------------------------------------------------------
#  Copyright (c) 2022 Vlad Popovici <popovici@bioxlab.org>
#
#  Licensed under the MIT License. See LICENSE file in root folder.
#  --------------------------------------------------------------------

from setuptools import setup

setup(
    name='compath',
    version='0.2.0',
    description='Computational Pathology',
    url='https://github.com/vladpopovici/ComPath',
    author='Vlad Popovici',
    author_email='popovici@bioxlab.org',
    license='MIT',
    packages=['compath'],
    install_requires=['shapely',
                      'numpy',
                      'zarr',
                      'cp_mri',
                      'cp_annot',
                      'configargparse',
                      ],

    classifiers=[
        'Development Status :: 4 - Beta'
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8'
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
