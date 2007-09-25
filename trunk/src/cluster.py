# cluster.py
#
# Author: Damian Eads
# Date:   September 22, 2007
#
# Copyright (c) 2007, Damian Eads
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#   - Redistributions of source code must retain the above
#     copyright notice, this list of conditions and the
#     following disclaimer.
#   - Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer
#     in the documentation and/or other materials provided with the
#     distribution.
#   - Neither the name of the author nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import _cluster_wrap
import fts
import scipy

__method_ids = {'single': 0, 'complete': 1, 'average': 2}

__unavailable_method_id = {'centroid': 3, 'ward': 4}

def randdm(pnts):
    """ Generates a random distance matrix stored in condensed form. A
        pnts * (pnts - 1) sized vector is returned.
    """
    if pnts >= 2:
        D = scipy.rand(pnts * (pnts - 1))
    else:
        raise AttributeError("The number of points in the distance matrix must be at least 2.")
    return D

def linkage(y, method='single'):
    """ Performs hierarchical clustering on the condensed distance matrix y.
        y must be a n * (n - 1) sized vector where n is the number of points
        paired in the distance matrix. The behavior of this function is
        very similar to the MATLAB linkage function.

        A (n - 1) * 4 matrix Z is returned. At the i'th iteration, clusters
        with indices Z[i, 0] and Z[i, 1] are combined to form cluster n + i.
        A cluster with an index less than n corresponds to one of the n
        original clusters. The distance between clusters Z[i, 0] and
        Z[i, 1] is given by Z[i, 2]. The fourth value Z[i, 3] represents the
        number of nodes in the cluster n + i.

        The following methods are used to compute the distance dist(s, t)
        between two clusters s and t. Suppose there are s_n original objects
        s[0], s[1], ..., s[n-1] in cluster s and t_n original objects
        t[0], t[1], ..., t[n-1] in cluster t.
        
          * method='single' assigns dist(s,t) = MIN(dist(s[i],t[j]) for
            all points i in cluster s and j in cluster t.

          * method='complete' assigns dist(s,t) = MAX(dist(s[i],t[j]) for
            all points i in cluster s and j in cluster t.

          * other methods are still not implemented.
        """
    s = y.shape
    d = scipy.ceil(scipy.sqrt(s[0] * 2))
    if y.dtype != 'double':
        raise AttributeError('Incompatible data type. y must be a matrix of doubles.')
    if d * (d - 1)/2 != s[0]:
        raise AttributeError('Incompatible vector size. It must be a binomial coefficient.')
    Z = scipy.zeros((d - 1, 3))
    _cluster_wrap.cluster_impl(y, Z, int(d), int(__method_ids[method]))
    return Z

class cnode:

    def __init__(self, id, left=None, right=None, dist=0, count=1):
        self.id = id
        self.left = left
        self.right = right
        self.dist = dist
        self.count = count

def totree(Z, return_dict=False):
    """
    Converts a hierarchical clustering encoded in the matrix Z (by linkage)
    into a tree. The root cnode object is returned.
    
    Each cnode object has a left, right, dist, id, and count attribute. The
    left and right attributes point to cnode objects that were combined to
    generate the cluster. If both are None then the cnode object is a
    leaf node, its count must be 1, and its distance is meaningless but
    set to 0.0.

    A reference to the root of the tree is returned.

    If return_dict is True the object returned is a tuple (t,Z) where
    """

    if len(Z.shape) != 2:
        raise AttributeError('Z must be a matrix')

    if Z.shape[1] != 4:
        raise AttributeError('Z must be a (n-1) by 4 matrix')

    # The number of original objects is equal to the number of rows minus
    # 1.
    n = Z.shape[0] + 1

    # Create an empty dictionary.
    d = {}

    # If we encounter a cluster being combined more than once, the matrix
    # must be corrupt.
    if scipy.unique(Z[:, 0:2].reshape((2 * (n - 1),))) != 2 * (n - 1):
        raise AttributeError('Corrupt matrix Z. Some clusters are more than once.')
    # If a cluster index is out of bounds, report an error.
    if (Z[:, 0:2] >= 2 * n - 1).sum() > 0:
        raise AttributeError('Corrupt matrix Z. Some cluster indices (first and second) are out of bounds.')
    if (Z[:, 0:2] < 0).sum() > 0:
        raise AttributeError('Corrupt matrix Z. Some cluster indices (first and second columns) are negative.')
    if Z[:, 2] < 0:
        raise AttributeError('Corrupt matrix Z. Some distances (third column) are negative.')

    if Z[:, 3] < 0:
        raise AttributeError('Counts (fourth column) are invalid.')

    # Create the nodes corresponding to the n original objects.
    for i in xrange(0, n):
        d[i] = cnode(i)

    nd = None

    for i in xrange(0, n - 1):
        fi = Z[i, 0]
        fj = Z[i, 1]
        if fi < i + n:
            raise AttributeError('Corrupt matrix Z. Index to derivative cluster is used before it is formed. See row %d, column 0' % fi)
        if fi < i + n:
            raise AttributeError('Corrupt matrix Z. Index to derivative cluster is used before it is formed. See row %d, column 1' % fj)
        nd = cnode(i + n, d[Z[i, 0]], d[Z[i, 1]])
        if d[(int)Z[i, 0]].count + d[(int)Z[i, 1]].count != nd.count:
            raise AttributeError('Corrupt matrix Z. The count Z[%d,3] is incorrect.' % i)
        d[n + i] = nd

    return nd
