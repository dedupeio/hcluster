#!/usr/bin/env python
try:
    from setuptools import setup, Extension
except ImportError :
    raise ImportError("setuptools module required, please go to https://pypi.python.org/pypi/setuptools and follow the instructions for installing setuptools")

class NumpyExtension(Extension):

    def __init__(self, *args, **kwargs):
        Extension.__init__(self, *args, **kwargs)

        self._include_dirs = self.include_dirs
        del self.include_dirs  # restore overwritten property

    # warning: Extension is a classic class so it's not really read-only

    @property
    def include_dirs(self):
        from numpy import get_include

        return self._include_dirs + [get_include()]

setup(maintainer="SciPy Developers",
      author="Eric Jones",
      maintainer_email="scipy-dev@scipy.org",
      description="Hierarchical Clustering Algorithms (Information Theory)",
      url="http://www.scipy.org",
      license="SciPy License (BSD Style)",
      ext_modules=[NumpyExtension('hcluster._hierarchy', 
                                  ['hcluster/_hierarchy.c'])],
  )
