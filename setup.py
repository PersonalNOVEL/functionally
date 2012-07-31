#!/usr/bin/env python

import os
from distutils.core import setup

setup(name='functionally',
      version='1.0',
      description='Simple & extensive functional programming library',
      long_description=open(os.path.join(os.path.dirname(__file__), 'README.md')).read(),
      author='Daniel Werner',
      author_email='dwerner@personalnovel.de',
      url='http://github.com/PersonalNOVEL/functionally',
      packages=['functionally'],
      keywords="functional",
      classifiers=[
          "Development Status :: 5 - Production/Stable",
          "License :: OSI Approved :: MIT License",
          "Topic :: Utilities",
      ],
      license="MIT",
     )
