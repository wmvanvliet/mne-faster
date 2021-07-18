#! /usr/bin/env python
from setuptools import setup
import os
import os.path as op

# get the version (don't import mne here, so dependencies are not needed)
version = None
with open(op.join('mne_faster', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')

if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    with open('README.rst', 'r') as fid:
        long_description = fid.read()

    setup(
        name='mne-faster',
        maintainer='Marijn van Vliet',
        maintainer_email='w.m.vanvliet@gmail.com',
        description='Code for performing the FASTER pipeline on MNE-Python data structures.',
        license='BSD-3',
        url='https://github.com/wmvanvliet/mne-faster',
        version=version,
        download_url='https://github.com/wmvanvliet/mne-faster/archive/master.zip',
        long_description=long_description,
        classifiers=['Intended Audience :: Science/Research',
                     'Intended Audience :: Developers',
                     'License :: OSI Approved',
                     'Programming Language :: Python',
                     'Topic :: Software Development',
                     'Topic :: Scientific/Engineering',
                     'Operating System :: Microsoft :: Windows',
                     'Operating System :: POSIX',
                     'Operating System :: Unix',
                     'Operating System :: MacOS'],
        platforms='any',
        packages=['mne_faster'],
        install_requires=['numpy', 'mne'],
    )
