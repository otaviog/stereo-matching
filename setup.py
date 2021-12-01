"""A setuptools based setup module for fusion-toolbox
"""

import sys
from os import path


from setuptools import find_packages

try:
    from skbuild import setup
except ImportError:
    print('scikit-build is required to build from source.', file=sys.stderr)
    print('Please run:', file=sys.stderr)
    print('', file=sys.stderr)
    print('  python -m pip install scikit-build')
    sys.exit(1)


def _forbid_publish():
    argv = sys.argv
    blacklist = ['register', 'upload']

    for command in blacklist:
        if command in argv:
            values = {'command': command}
            print('Command "%(command)s" has been blacklisted, exiting...' %
                  values)
            sys.exit(2)


_forbid_publish()

REQUIREMENTS = [    
    'numpy',
    'matplotlib',
    'scipy',
    'torch'
]


setup(
    name='stereomatch',
    version='1.0.0',
    author='Otavio Gomes',
    author_email='otavio.b.gomes@gmail.com',
    description='Sample implementation of stereo matching algorithms',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5'
    ],
    packages=find_packages(exclude=['*._test']),
    install_requires=REQUIREMENTS
)
