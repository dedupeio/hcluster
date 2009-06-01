from distutils.core import setup, Extension
import sys, os, os.path, string

def contains_arrayobject_h(path):
    """
    Returns True iff the python path string contains the arrayobject.h
    include file where it is supposed to be.
    """
    f=False
    try:
        s=os.stat(os.path.join(path, 'numpy', 'core', 'include', \
                               'numpy', 'arrayobject.h'))
        f=True
    except OSError:
        pass
    return f

if sys.platform != 'darwin':
    extra_link_args = ['-s']

valid_paths = filter(contains_arrayobject_h, sys.path)
if len(valid_paths) == 0:
    print "No paths in the python path contain numpy/arrayobject.h"
    sys.exit(1)

# The base path is by default the first python path with arrayobject.h in it.
include_numpy_array=valid_paths[0]

if len(valid_paths) > 1:
    print "There are several valid include directories containing numpy/arrayobject.h"
    l=[('%d: %s' % (i+1, valid_paths[i])) for i in xrange(0, len(valid_paths))]
    s = -1
    print string.join(l, '\n')
    # Prompt the user with a list of selections.
    while not (s >= 1 and s <= len(valid_paths)):
        s = input('Selection [default=1]:')
        if s == '':
            s = 1
        else:
            s = int(s)
    include_numpy_array=valid_paths[s-1]

# Add the children directory path suffix to the base path.
include_numpy_array=os.path.join(include_numpy_array, 'numpy', 'core', \
                                 'include')
extra_link_args = []


setup(name='hcluster', \
      version='0.2.0', \
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
      ext_modules=[Extension('_hierarchy_wrap',
                             ['hcluster/hierarchy.c', 'hcluster/hierarchy_wrap.c'],
                             extra_link_args = extra_link_args,
                             include_dirs=['hcluster/', include_numpy_array]),
                   Extension('_distance_wrap',
                             ['hcluster/distance.c', 'hcluster/distance_wrap.c'],
                             extra_link_args = extra_link_args,
                             include_dirs=['hcluster/', include_numpy_array])],
      keywords=['dendrogram', 'linkage', 'cluster', 'agglomorative', 'hierarchical', 'hierarchy', 'ward', 'distance'],
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
