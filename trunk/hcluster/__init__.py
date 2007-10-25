import cluster as _cp
from cluster import *
from inspect import getmembers
from types import FunctionType
from pydoc import getdoc
from cluster import __doc__ as _ds

__doc__ = _ds

__doc__ += """

Detailed Documentation
----------------------
"""

for (n, f) in getmembers(_cp):
    if type(f) is types.FunctionType and not n.startswith('_'):
        __doc__ += "===== %s\n %s \n\n" % (n, getdoc(f))


__doc__ += """
\n\n============= cnode Class

Brief Summary:

%s

""" % getdoc(_cp.cnode)

for (n, f) in getmembers(_cp.cnode):
    if type(f) is types.MethodType and not n.startswith('_'):
        __doc__ += "Method %s:\n %s \n\n" % (n, getdoc(f))

