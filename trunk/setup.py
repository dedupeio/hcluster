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
    sys.exit(0)

# The base path is by default the first python path with arrayobject.h in it.
include_numpy_array=valid_paths[0]

if len(valid_paths) > 1:
    print "There are several valid include directories containing numpy/arrayobject.h"
    l=[('%d: %s' % (i+1, valid_paths[i])) for i in xrange(0, len(valid_paths))]
    s = -1
    print string.join(l, '\n')
    # Prompt the user with a list of selections.
    while not (s >= 1 and s <= len(valid_paths)):
        s = input('Selection [default=1]:' % s)
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
      version='0.1.3', \
      py_modules=['hcluster.cluster'], \
      description='A hierarchical clustering package written for Scipy.', \
      long_description='A hierarchical clustering package written in C and Python.', \
      ext_modules=[Extension('_cluster_wrap', \
                             ['hcluster/cluster.c', 'hcluster/cluster_wrap.c'], \
                             extra_link_args = extra_link_args, \
                             include_dirs=['hcluster/', include_numpy_array])], \
      keywords=['dendrogram', 'linkage', 'cluster', 'agglomorative', 'hierarchical', 'hierarchy'], \
      author="Damian Eads", \
      author_email="freesoftware@eadsware.com", \
      license="New BSD License", \
      packages = ['hcluster'], \
      url = 'http://scipy-cluster.googlecode.com', \
      )
