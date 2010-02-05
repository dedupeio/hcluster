from numpy.distutils.core import setup, Extension
import sys, os, os.path, string

extra_link_args = []

setup(name='hcluster', \
      version='0.2.0.svn', \
      py_modules=['hcluster.hierarchy', 'hcluster.distance'], \
      description='A hierarchical clustering package for Scipy.', \
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
      ext_modules=[Extension('_hierarchy_wrap',
                             ['hcluster/hierarchy.c', 'hcluster/hierarchy_wrap.c'],
                             extra_link_args = extra_link_args),
                   Extension('_distance_wrap',
                             ['hcluster/distance.c', 'hcluster/distance_wrap.c'],
                             extra_link_args = extra_link_args)],
      author="Damian Eads",
      author_email="damian XDOTX eads XATX gmail XDOTX com",
      license="New BSD License",
      packages = ['hcluster'],
      classifiers = ["Topic :: Scientific/Engineering :: Information Analysis",
                     "Topic :: Scientific/Engineering :: Artificial Intelligence",
                     "Topic :: Scientific/Engineering :: Bio-Informatics",
                     "Programming Language :: Python",
                     "Operating System :: OS Independent",
                     "License :: OSI Approved :: BSD License",
                     "Intended Audience :: Science/Research",
                     "Development Status :: 4 - Beta"],
      url = 'http://scipy-cluster.googlecode.com',
      )
