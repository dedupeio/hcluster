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

    def get_include_dirs(self):
        from numpy import get_include

        return self._include_dirs + [get_include()]

    def set_include_dirs(self, value):
        self._include_dirs = value

    def del_include_dirs(self):
        pass
        
    include_dirs = property(get_include_dirs, 
                            set_include_dirs, 
                            del_include_dirs)


setup(maintainer="Forest Gregg",
      version="0.3.0",
      name='hcluster',
      maintainer_email="fgregg@datamade.us",
      description="Hierarchical Clustering Algorithms (Information Theory)",
      url="https://github.com/datamade/hcluster",
      license="SciPy License (BSD Style)",
      install_requires=['future', 'numpy'],
      ext_modules=[NumpyExtension('hcluster._hierarchy', 
                                  ['hcluster/_hierarchy.c']),

                   NumpyExtension('hcluster._distance_wrap',
                                  ['hcluster/distance_wrap.c'])],
      long_description="""
This library provides Python functions for hierarchical clustering. Its features
include

    * generating hierarchical clusters from distance matrices
    * computing distance matrices from observation vectors
    * computing statistics on clusters
    * cutting linkages to generate flat clusters
    * and visualizing clusters with dendrograms.

The interface is very similar to MATLAB's Statistics Toolbox API to make code
easier to port from MATLAB to Python/Numpy. The core implementation of this
library is in C for efficiency.
""",
      keywords=['dendrogram', 'linkage', 'cluster', 'agglomorative', 'hierarchical', 'hierarchy', 'ward', 'distance'],
      classifiers = ["Topic :: Scientific/Engineering :: Information Analysis",
                     "Topic :: Scientific/Engineering :: Artificial Intelligence",
                     "Topic :: Scientific/Engineering :: Bio-Informatics",
                     "Programming Language :: Python",
                     "Operating System :: OS Independent",
                     "License :: OSI Approved :: BSD License",
                     "Intended Audience :: Science/Research",
                     "Development Status :: 4 - Beta"],
  )
