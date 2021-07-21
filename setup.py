#!/usr/bin/env python
from setuptools import setup, Extension
import numpy

try:
    from Cython.Build import cythonize
    use_cython = True
except ImportError:
    use_cython = False


ext_modules = [Extension('hcluster._distance_wrap',
                         ['hcluster/distance_wrap.c'],
                         include_dirs=[numpy.get_include()])]

if use_cython:
    ext_modules += cythonize([Extension('hcluster._hierarchy',
                                        ['hcluster/_hierarchy.pyx'],
                                        include_dirs=[numpy.get_include()])])
else:
    ext_modules += [Extension('hcluster._hierarchy',
                              ['hcluster/_hierarchy.c'],
                              include_dirs=[numpy.get_include()])]

setup(maintainer="Forest Gregg",
      version="0.3.8",
      name='dedupe-hcluster',
      packages=['hcluster'],
      maintainer_email="fgregg@datamade.us",
      description="Hierarchical Clustering Algorithms (Information Theory)",
      url="https://github.com/datamade/hcluster",
      license="SciPy License (BSD Style)",
      install_requires=["numpy>=1.10.4 ;python_version<'3.6'",
                        "numpy>=1.12.1 ;python_version=='3.6'",
	                "numpy>=1.15.0; python_version=='3.7'",
		        "numpy>=1.17.3; python_version=='3.8'",
		        "numpy; python_version>'3.8'",
		        "Cython"],
      ext_modules=ext_modules,
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
