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
import scipy
import types
import math

__method_ids = {'single': 0, 'complete': 1, 'average': 2}

__unavailable_method_id = {'centroid': 3, 'ward': 4}

def randdm(pnts):
    """ Generates a random distance matrix stored in condensed form. A
        pnts * (pnts - 1) / 2 sized vector is returned.
    """
    if pnts >= 2:
        D = scipy.rand(pnts * (pnts - 1) / 2)
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
    t = totree(Z)
    
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

    a = scipy.array(())

    if type(a) != type(Z):
        raise AttributeError('Z must be a numpy.ndarray')

    if Z.dtype != 'double':
        raise AttributeError('Z must have double elements, not %s', str(Z.dtype))
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
        if d[int(Z[i, 0])].count + d[int(Z[i, 1])].count != nd.count:
            raise AttributeError('Corrupt matrix Z. The count Z[%d,3] is incorrect.' % i)
        d[n + i] = nd

    return nd

def squareform(X, force=None):
    """ Converts a vectorform distance vector to a squareform distance
    matrix, and vice-versa.

    v = squareform(X)

      Given a square dxd symmetric distance matrix X, v=squareform(X)
      returns a d*(d-1)/2 (n \choose 2) sized vector v.

      v[(i + 1) \choose 2 + j] is the distance between points i and j.
      If X is non-square or asymmetric, an error is returned.

    X = squareform(v)

      Given a d*d(-1)/2 sized v for some integer d>=2 encoding distances
      as described, X=squareform(v) returns a dxd distance matrix X. The
      X[i, j] and X[j, i] value equals v[(i + 1) \choose 2 + j] and all
      diagonal elements are zero.

    As with MATLAB, if force is equal to 'tovector' or 'tomatrix',
    the input will be treated as a distance matrix or distance vector
    respectively.
    """
    a = scipy.array(())
    
    if type(X) != type(a):
        raise AttributeError('The parameter passed must be an array.')

    if X.dtype != 'double':
        raise AttributeError('A double array must be passed.')

    s = X.shape
    
    # X = squareform(v)
    if len(s) == 1 and force != 'tomatrix':
        # Grab the closest value to the square root of the number
        # of elements times 2 to see if the number of elements
        # is indeed a binomial coefficient.
        d = int(scipy.ceil(scipy.sqrt(X.shape[0] * 2)))

        print d, s[0]
        # Check that v is of valid dimensions.
        if d * (d - 1) / 2 != int(s[0]):
            raise AttributeError('Incompatible vector size. It must be a binomial coefficient n choose 2 for some integer n >= 2.')
        
        # Allocate memory for the distance matrix.
        M = scipy.zeros((d, d), 'double')

        # Fill in the values of the distance matrix.
        _cluster_wrap.to_squareform_from_vector(M, X)

        # Return the distance matrix.
        M = M + M.transpose()
        return M
    elif len(s) != 1 and force.lower() == 'tomatrix':
        raise AttributeError("Forcing 'tomatrix' but input X is not a distance vector.")
    elif len(s) == 2 and force.lower() != 'tovector':
        if s[0] != s[1]:
            raise AttributeError('The matrix argument must be square.')
        if scipy.sum(scipy.sum(X == X.transpose())) != scipy.product(X.shape):
            raise AttributeError('The distance matrix must be symmetrical.')
        if (X.diagonal() != 0).any():
            raise AttributeError('The distance matrix must have zeros along the diagonal.')

        # One-side of the dimensions is set here.
        d = s[0]
        
        # Create a vector.
        v = scipy.zeros(((d * (d - 1) / 2),), 'double')

        # Convert the vector to squareform.
        _cluster_wrap.to_vector_from_squareform(X, v)
        return v
    elif len(s) != 2 and force.lower() == 'tomatrix':
        raise AttributeError("Forcing 'tomatrix' but input X is not a distance vector.")
    else:
        raise AttributeError('The first argument must be a vector or matrix. A %d-dimensional array is not permitted' % len(s))

def pdist(X, metric='euclidean', p=2):
    a = scipy.array(())
    
    if type(X) != type(a):
        raise AttributeError('The parameter passed must be an array.')

    if X.dtype != 'double':
        raise AttributeError('A double array must be passed.')

    s = X.shape

    if len(s) != 2:
        raise AttributeError('A matrix must be passed.');

    m = s[0]
    n = s[1]
    dm = scipy.zeros((m * (m - 1) / 2,), dtype='double')

    if type(metric) is types.FunctionType:
        k = 0
        for i in xrange(0, m - 1):
            for j in xrange(i+1, m):
                dm[k] = metric(X[i, :], X[j, :])
                k = k + 1
    elif type(metric) is types.StringType:
        if metric.lower() in set(['euclidean', 'euclid', 'eu', 'e']):
            _cluster_wrap.pdist_euclidean_wrap(X, dm)
        elif metric.lower() in set(['cityblock', 'cblock', 'cb', 'c']):
            _cluster_wrap.pdist_city_block_wrap(X, dm)
        elif metric.lower() in set(['minkowski', 'mi', 'm']):
            _cluster_wrap.pdist_minkowski_wrap(X, dm, p)
        elif metric.lower() in set(['seuclidean', 'mahalanobis', \
                                    'cosine', 'correlation', 'hamming', \
                                    'jaccard', 'chebychev']):
            dm = pdist(X, 'test_' + metric.lower())
        elif metric == 'test_euclidean':
            dm = pdist(X, (lambda u, v: scipy.sqrt(((u-v)*(u-v).T).sum())))
        elif metric == 'test_seuclidean':
            D = scipy.diagflat(scipy.var(X, axis=0))
            DI = scipy.linalg.inv(D)
            dm = pdist(X, (lambda u, v: scipy.sqrt(((u-v)*DI*(u-v).T).sum())))
        elif metric == 'test_mahalanobis':
            V = scipy.cov(X.T)
            VI = scipy.linalg.inv(V)
            dm = pdist(X, (lambda u, v: scipy.sqrt(((u-v)*VI*(u-v).T).sum())))            
        elif metric == 'test_cityblock':
            dm = pdist(X, (lambda u, v: abs(u-v).sum()))
        elif metric == 'test_minkowski':
            dm = pdist(X, (lambda u, v: math.pow((abs(u-v)**p).sum(), 1.0/p)))
        elif metric == 'test_cosine':
            dm = pdist(X, \
                       (lambda u, v: \
                        (1 - scipy.dot(u, v)) / \
                        (math.sqrt(scipy.dot(u, u)) * \
                         math.sqrt(scipy.dot(v, v)))))
        elif metric == 'test_correlation':
            dm = pdist(X, \
                       (lambda u, v: 1 - \
                        scipy.dot(u - u.mean(), v - v.mean()) / \
                        math.sqrt(scipy.dot(u - u.mean(), \
                                            u - u.mean())) \
                        * math.sqrt(scipy.dot(v - v.mean(), \
                                              v - v.mean()))))
        elif metric == 'test_hamming':
            dm = pdist(X, (lambda u, v: (u != v).mean()))
        elif metric == 'test_jaccard':
            dm = pdist(X, \
                       (lambda u, v: \
                        ((scipy.bitwise_and((u != v),
                                       scipy.bitwise_or(u != 0, \
                                                   v != 0))).sum()) / \
                        (scipy.bitwise_or(u != 0, v != 0)).sum()))
        elif metric == 'test_chebyshev':
            dm = pdist(X, lambda u, v: max(abs(u-v)))
        else:
            raise AttributeError('Unknown Distance Metric: %s' % chebyshev)
    else:
        raise AttributeError('2nd argument metric must be a string identifier or a function.')
    return dm
