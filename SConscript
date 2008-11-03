# Last Change: Thu Oct 18 09:00 PM 2007 J
# vim:syntax=python
from os.path import join

from numscons import GetNumpyEnvironment

env = GetNumpyEnvironment(ARGUMENTS)

env.NumpyPythonExtension('_distance_wrap', source = [join('src', 'distance_wrap.c'),
                                          join('src', 'distance.c')])
