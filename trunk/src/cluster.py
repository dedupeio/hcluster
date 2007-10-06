"""
cluster.py

Author: Damian Eads
Date:   September 22, 2007

Copyright (c) 2007, Damian Eads

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
  - Redistributions of source code must retain the above
    copyright notice, this list of conditions and the
    following disclaimer.
  - Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer
    in the documentation and/or other materials provided with the
    distribution.
  - Neither the name of the author nor the names of its
    contributors may be used to endorse or promote products derived
    from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

MATLAB and MathWorks are registered trademarks of The MathWorks, Inc.
"""

import _cluster_wrap
import scipy, scipy.stats, numpy
import types
import math

_cpy_non_euclid_methods = {'single': 0, 'complete': 1, 'average': 2, 'weighted': 6}
_cpy_euclid_methods = {'centroid': 3, 'median': 4, 'ward': 5}
_cpy_linkage_methods = set(_cpy_non_euclid_methods.keys()).union(set(_cpy_euclid_methods.keys()))
_array_type = type(scipy.array([]))

def randdm(pnts):
    """ Generates a random distance matrix stored in condensed form. A
        pnts * (pnts - 1) / 2 sized vector is returned.
    """
    if pnts >= 2:
        D = scipy.rand(pnts * (pnts - 1) / 2)
    else:
        raise AttributeError("The number of points in the distance matrix must be at least 2.")
    return D

def linkage(y, method='single', metric='euclidean'):
    """ linkage(y, method)

        Performs hierarchical clustering on the condensed distance matrix y.
        y must be a n * (n - 1) sized vector where n is the number of points
        paired in the distance matrix. The behavior of this function is
        very similar to the MATLAB(R) linkage function.

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

               (also called Nearest Point Algorithm)

          * method='complete' assigns dist(s,t) = MAX(dist(s[i],t[j]) for
            all points i in cluster s and j in cluster t.

               (also called Farthest Point Algorithm
                     or the Voor Hees Algorithm)

          * method='average' assigns dist(s,t) =
               sum_{ij} { dist(s[i], t[j]) } / (|s|*|t|)
            for all points i and j where |s| and |t| are the
            cardinalities of clusters s and t, respectively.

               (also called UPGMA)

          * method='weighted' assigns

               dist(q,u) = (dist(s,u) + dist(t,u))/2

            where q is the newly formed cluster consisting of s and t,
            and u is a remaining cluster in the unused forest of
            clusters. (also called WPGMA)

        linkage(X, method, metric='euclidean')

        Performs hierarchical clustering on the objects defined by the
        n by m observation matrix X.

        If the metric is 'euclidean' then the following methods may be
        used:

          * method='centroid' assigns dist(s,t) = euclid(c_s, c_t) where
            c_s and c_t are the centroids of clusters s and t,
            respectively. When two clusters s and t are combined into a new
            cluster q, the new centroid is computed over all the original
            objects in clusters s and t. (also called UPGMC)

          * method='median' assigns dist(s,t) as above. When two clusters
            s and t are combined into a new cluster q, the average of
            centroids s and t give the new centroid q. (also called WPGMC)
           
          * method='ward' uses the Ward variance minimization algorithm.
            The new entry dist(q, u) is computed as follows,

                 dist(q,u) =

             ----------------------------------------------------
             | |u|+|s|            |u|+|t|            |u|
             | ------- d(u,s)^2 + ------- d(u,t)^2 - --- d(s,t)^2
            \|    T                  T                T

            where q is the newly formed cluster consisting of clusters
            s and t, u is an unused cluster in the forest, T=|u|+|s|+|t|,
            and |*| is the cardinality of its argument.
            (also called incremental)
        """
    if type(y) != _array_type:
        raise AttributeError('Incompatible data type. y must be an array.')
    s = y.shape
    if type(method) != types.StringType:
        raise AttributeError("Argument 'method' must be a string.")
    if y.dtype != 'double':
        raise AttributeError('Incompatible data type. y must be a matrix of doubles.')

    if len(s) == 1:
        d = scipy.ceil(scipy.sqrt(s[0] * 2))
        if d * (d - 1)/2 != s[0]:
            raise AttributeError('Incompatible vector size. It must be a binomial coefficient.')
        if method not in _cpy_non_euclid_methods.keys():
            raise AttributeError("Valid methods when the raw observations are omitted are 'single', 'complete', 'weighted', and 'average'.")
        Z = scipy.zeros((d - 1, 4))
        _cluster_wrap.linkage_wrap(y, Z, int(d), \
                                   int(_cpy_non_euclid_methods[method]))
    elif len(s) == 2:
        X = y
        n = s[0]
        m = s[1]
        if method not in _cpy_linkage_methods:
            raise AttributeError('Invalid method: %s' % method)
        if method in _cpy_non_euclid_methods.keys():
            dm = pdist(X, metric)
            Z = scipy.zeros((n - 1, 4))
            _cluster_wrap.linkage_wrap(dm, Z, n, \
                                       int(_cpy_non_euclid_methods[method]))
        elif method in _cpy_euclid_methods.keys():
            if metric != 'euclidean':
                raise AttributeError('Method %s requires the distance metric to be euclidean' % s)
            dm = pdist(X, metric)
            Z = scipy.zeros((n - 1, 4))
            _cluster_wrap.linkage_euclid_wrap(dm, Z, X, m, n,
                                              int(_cpy_euclid_methods[method]))
    return Z

class cnode:
    """
    A tree node class for representing a cluster. Leaf nodes correspond
    to original observations, while non-leaf nodes correspond to
    non-singleton clusters.

    The totree function converts a matrix returned by the linkage
    function into a tree representation.
    """

    def __init__(self, id, left=None, right=None, dist=0, count=1):
        if id < 0:
            raise AttributeError('The id must be non-negative.')
        if dist < 0:
            raise AttributeError('The distance must be non-negative.')
        if (left is None and right is not None) or \
           (left is not None and right is None):
            raise AttributeError('Only full or proper binary trees are permitted. This node has one child.')
        if count < 1:
            raise AttributeError('A cluster must contain at least one original observation.')
        self.id = id
        self.left = left
        self.right = right
        self.dist = dist
        if self.left is None:
            self.count = count
        else:
            self.count = left.count + right.count

    def getId(self):
        """
        i = nd.getId()
        
        Returns the id number of the node nd. For 0 <= i < n, i
        corresponds to original observation i. For n <= i < 2n - 1,
        i corresponds to non-singleton cluster formed at iteration i-n.
        """
        return self.id

    def getCount(self):
        """
        c = nd.getCount()

        Returns the number of leaf nodes below and including nd. This
        represents the number of original observations in the cluster
        represented by the node. If the nd is a leaf, this number is
        1.
        """
        return self.count

    def getLeft(self):
        """
        left = nd.getLeft()

        Returns a reference to the left child. If the node is a
        leaf, None is returned.
        """
        return self.left

    def getRight(self):
        """
        left = nd.getLeft()

        Returns a reference to the right child. If the node is a
        leaf, None is returned.
        """
        return self.right

    def isLeaf(self):
        """
        Returns True if the node is a leaf.
        """
        return self.left is None

    def preOrder(self, func=(lambda x: x.id)):
        """
        vlst = preOrder(func)
    
        Performs preorder traversal but without recursive function calls.
        When a leaf node is first encountered, func is called with the
        leaf node as the argument, and its result is appended to the list vlst.
    
        For example, the statement
        
        ids = root.preOrder(lambda x: x.id)
    
        returns a list of the node ids corresponding to the leaf nodes of
        the tree starting at the root defin.
        """
    
        # Do a preorder traversal, caching the result. To avoid having to do
        # recursion, we'll store the previous index we've visited in a vector.
        n = self.count
        
        curNode = [None] * (2 * n)
        lvisited = scipy.zeros((2 * n,), dtype='bool')
        rvisited = scipy.zeros((2 * n,), dtype='bool')
        curNode[0] = self
        k = 0
        preorder = []
        while k >= 0:
            nd = curNode[k]
            ndid = nd.id
            if nd.isLeaf():
                preorder.append(func(nd))
                k = k - 1
            else:
                if not lvisited[ndid]:
                    curNode[k + 1] = nd.left
                    lvisited[ndid] = True
                    k = k + 1
                elif not rvisited[ndid]:
                    curNode[k + 1] = nd.right
                    rvisited[ndid] = True
                    k = k + 1
                # If we've visited the left and right of this non-leaf
                # node already, go up in the tree.
                else:
                    k = k - 1
            
        return preorder

_cnode_bare = cnode(0)
_cnode_type = type(cnode)

def totree(Z, rd=False):
    """
    r = totree(Z)
    
      Converts a hierarchical clustering encoded in the matrix Z (by linkage)
      into a tree. The reference r to the root cnode object is returned.
    
      Each cnode object has a left, right, dist, id, and count attribute. The
      left and right attributes point to cnode objects that were combined to
      generate the cluster. If both are None then the cnode object is a
      leaf node, its count must be 1, and its distance is meaningless but
      set to 0.

    (r, d) = totree(Z, rd=True)

      Same as totree(Z) except a tuple is returned where r is the reference
      to the root cnode and d is a reference to a dictionary mapping
      cluster ids to cnodes. If a cluster id is less than n, then it
      corresponds to a singleton cluster (leaf node).      
    """

    if type(Z) is not _array_type:
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

    # Create a list full of None's to store the node objects
    d = [None] * (n*2-1)

    # If we encounter a cluster being combined more than once, the matrix
    # must be corrupt.
    if len(scipy.unique(Z[:, 0:2].reshape((2 * (n - 1),)))) != 2 * (n - 1):
        raise AttributeError('Corrupt matrix Z. Some clusters are more than once.')
    # If a cluster index is out of bounds, report an error.
    if (Z[:, 0:2] >= 2 * n - 1).any():
        raise AttributeError('Corrupt matrix Z. Some cluster indices (first and second) are out of bounds.')
    if (Z[:, 0:2] < 0).any():
        raise AttributeError('Corrupt matrix Z. Some cluster indices (first and second columns) are negative.')
    if (Z[:, 2] < 0).any():
        raise AttributeError('Corrupt matrix Z. Some distances (third column) are negative.')

    if (Z[:, 3] < 0).any():
        raise AttributeError('Counts (fourth column) are invalid.')

    # Create the nodes corresponding to the n original objects.
    for i in xrange(0, n):
        d[i] = cnode(i)

    nd = None

    for i in xrange(0, n - 1):
        fi = int(Z[i, 0])
        fj = int(Z[i, 1])
        if fi > i + n:
            raise AttributeError('Corrupt matrix Z. Index to derivative cluster is used before it is formed. See row %d, column 0' % fi)
        if fj > i + n:
            raise AttributeError('Corrupt matrix Z. Index to derivative cluster is used before it is formed. See row %d, column 1' % fj)
        nd = cnode(i + n, d[fi], d[fj],  Z[i, 2])
        #          ^ id   ^ left ^ right ^ dist
        if Z[i,3] != nd.count:
            raise AttributeError('Corrupt matrix Z. The count Z[%d,3] is incorrect.' % i)
        d[n + i] = nd

    if rd:
        return (nd, d)
    else:
        return nd

    

def squareform(X, force="no", checks=True):
    """ Converts a vectorform distance vector to a squareform distance
    matrix, and vice-versa. 

    v = squareform(X)

      Given a square dxd symmetric distance matrix X, v=squareform(X)
      returns a d*(d-1)/2 (n \choose 2) sized vector v.

      v[(n \choose 2)-(n-i \choose 2) + (j-i-1)] is the distance
      between points i and j. If X is non-square or asymmetric, an error
      is returned.

    X = squareform(v)

      Given a d*d(-1)/2 sized v for some integer d>=2 encoding distances
      as described, X=squareform(v) returns a dxd distance matrix X. The
      X[i, j] and X[j, i] values are set to
      v[(n \choose 2)-(n-i \choose 2) + (j-u-1)] and all
      diagonal elements are zero.

    As with MATLAB(R), if force is equal to 'tovector' or 'tomatrix',
    the input will be treated as a distance matrix or distance vector
    respectively.

    If checks is set to False, no checks will be made for matrix
    symmetry nor zero diaganols. This is useful if it is known that
    X - X.T is small and diag(X) is close to zero. These values are
    ignored any way so they do not disrupt the squareform
    transformation.
    """
    
    if type(X) is not _array_type:
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
        _cluster_wrap.to_squareform_from_vector_wrap(M, X)

        # Return the distance matrix.
        M = M + M.transpose()
        return M
    elif len(s) != 1 and force.lower() == 'tomatrix':
        raise AttributeError("Forcing 'tomatrix' but input X is not a distance vector.")
    elif len(s) == 2 and force.lower() != 'tovector':
        if s[0] != s[1]:
            raise AttributeError('The matrix argument must be square.')
        if checks:
            if scipy.sum(scipy.sum(X == X.transpose())) != scipy.product(X.shape):
                raise AttributeError('The distance matrix must be symmetrical.')
            if (X.diagonal() != 0).any():
                raise AttributeError('The distance matrix must have zeros along the diagonal.')

        # One-side of the dimensions is set here.
        d = s[0]
        
        # Create a vector.
        v = scipy.zeros(((d * (d - 1) / 2),), 'double')

        # Convert the vector to squareform.
        _cluster_wrap.to_vector_from_squareform_wrap(X, v)
        return v
    elif len(s) != 2 and force.lower() == 'tomatrix':
        raise AttributeError("Forcing 'tomatrix' but input X is not a distance vector.")
    else:
        raise AttributeError('The first argument must be a vector or matrix. A %d-dimensional array is not permitted' % len(s))

def pdist(X, metric='euclidean', p=2):
    """ Computes the distance between m points in n-dimensional space.

        1. pdist(X)

        Computes the distance between m points using Euclidean distance
        (2-norm) as the distance metric between the points. The points
        are arranged as m n-dimensional row vectors in the matrix X.

        2. pdist(X, 'minkowski', p)

        Computes the distances using the Minkowski distance (p-norm) where
        p is a number.

        3. pdist(X, 'cityblock')

        Computes the city block or manhattan distance between the points.

        4. pdist(X, 'seuclidean')

        Computes the standardized euclidean distance so that the distances
        are of unit variance.

        5. pdist(X, 'cosine')

        Computes the cosine distance between vectors u and v. This is
        
           1 - uv^T
           -----------
           |u|_2 |v|_2

        where |*|_2 is the 2 norm of its argument *.

        6. pdist(X, 'correlation')

        Computes the correlation distance between vectors u and v. This is

           1 - (u - n|u|_1)(v - n|v|_1)^T
           --------------------------------- ,
           |(u - n|u|_1)|_2 |(v - n|v|_1)|^T

        where |*|_1 is the Manhattan (or 1-norm) of its argument *,
        and n is the common dimensionality of the vectors.

        7. pdist(X, 'hamming')

        Computes the normalized Hamming distance, or the proportion
        of those vector elements between two vectors u and v which
        disagree. To save memory, the matrix X can be of type boolean.

        8. pdist(X, 'jaccard')

        Computes the Jaccard distance between the points. Given two
        vectors, u and v, the Jaccard disaance is the proportion of
        those elements u_i and v_i that disagree where at least one
        of them is non-zero.

        9. pdist(X, 'chebyshev')

        Computes the Chebyshev distance between the points. The
        Chebyshev distance between two vectors u and v is the maximum
        norm-1 distance between their respective elements. More
        precisely, the distance is given by

           d(u,v) = max_{i=1}^{n}{|u_i-v_i|}.

        10. pdist(X, f)
        
        Computes the distance between all pairs of vectors in X
        using the user supplied 2-arity function f. For example,
        Euclidean distance between the vectors could be computed
        as follows,

            dm = pdist(X, (lambda u, v: scipy.sqrt(((u-v)*(u-v).T).sum())))

        11. pdist(X, 'test_Y')

        Computes the distance between all pairs of vectors in X
        using the distance metric Y but with a more succint,
        verifiable, but less efficient implementation.

       """

    # FIXME: need more efficient mahalanobis distance.
    # TODO: canberra, bray-curtis, matching, dice, rogers-tanimoto,
    #       russell-rao, sokal-sneath, yule
    
    if type(X) is not _array_type:
        raise AttributeError('The parameter passed must be an array.')
    
    s = X.shape

    if len(s) != 2:
        raise AttributeError('A matrix must be passed.');

    m = s[0]
    n = s[1]
    dm = scipy.zeros((m * (m - 1) / 2,), dtype='double')

    mtype = type(metric)
    if mtype is types.FunctionType:
        k = 0
        for i in xrange(0, m - 1):
            for j in xrange(i+1, m):
                dm[k] = metric(X[i, :], X[j, :])
                k = k + 1
    elif mtype is types.StringType:
        mstr = metric.lower()
        if X.dtype != 'double' and (mstr != 'hamming' and mstr != 'jaccard'):
            AttributeError('A double array must be passed.')
        if mstr in set(['euclidean', 'euclid', 'eu', 'e']):
            _cluster_wrap.pdist_euclidean_wrap(X, dm)
        elif mstr in set(['cityblock', 'cblock', 'cb', 'c']):
            _cluster_wrap.pdist_city_block_wrap(X, dm)
        elif mstr in set(['hamming', 'hamm', 'ha', 'h']):
            if X.dtype == 'double':
                _cluster_wrap.pdist_hamming_wrap(X, dm)
            elif X.dtype == 'bool':
                _cluster_wrap.pdist_hamming_bool_wrap(X, dm)
            else:
                raise AttributeError('Invalid input matrix type %s for hamming.' % str(X.dtype))
        elif mstr in set(['jaccard', 'jacc', 'ja', 'j']):
            if X.dtype == 'double':
                _cluster_wrap.pdist_hamming_wrap(X, dm)
            elif X.dtype == 'bool':
                _cluster_wrap.pdist_hamming_bool_wrap(X, dm)
            else:
                raise AttributeError('Invalid input matrix type %s for jaccard.' % str(X.dtype))
        elif mstr in set(['chebyshev', 'cheby', 'cheb', 'ch']):
            _cluster_wrap.pdist_chebyshev_wrap(X, dm)            
        elif mstr in set(['minkowski', 'mi', 'm']):
            _cluster_wrap.pdist_minkowski_wrap(X, dm, p)
        elif mstr in set(['seuclidean', 'se', 's']):
            VV = scipy.stats.var(X, axis=0)
            _cluster_wrap.pdist_seuclidean_wrap(X, VV, dm)
        # Need to test whether vectorized cosine works better.
        # Find out: Is there a dot subtraction operator so I can
        # subtract matrices in a similar way to multiplying them?
        # Need to get rid of as much unnecessary C code as possible.
        elif mstr in set(['cosine_old', 'cos_old']):
            norms = scipy.sqrt(scipy.sum(X * X, axis=1))
            _cluster_wrap.pdist_cosine_wrap(X, dm, norms)
        elif mstr in set(['cosine', 'cos']):
            norms = scipy.sqrt(scipy.sum(X * X, axis=1))
            nV = norms.reshape(m, 1)
            # The numerator u * v
            nm = scipy.dot(X, X.T)
            
            # The denom. ||u||*||v||
            de = scipy.dot(nV, nV.T);

            dm = 1 - (nm / de)
            dm[xrange(0,m),xrange(0,m)] = 0
            dm = squareform(dm)
        elif mstr in set(['correlation', 'co']):
            X2 = X - scipy.repmat(scipy.mean(X, axis=1).reshape(m, 1), 1, n)
            norms = scipy.sqrt(scipy.sum(X2 * X2, axis=1))
            _cluster_wrap.pdist_cosine_wrap(X2, dm, norms)
        elif mstr in set(['stub_mahalanobis']):
            k = 0;
            XV = scipy.dot(X, scipy.cov(X.T))
            dm = scipy.dot(XV, X.T)
            print dm.shape
            dm[xrange(0,m),xrange(0,m)] = 0
            dm = squareform(dm, checks=False)
        elif metric == 'test_euclidean':
            dm = pdist(X, (lambda u, v: scipy.sqrt(((u-v)*(u-v).T).sum())))
        elif metric == 'test_seuclidean':
            D = scipy.diagflat(scipy.stats.var(X, axis=0))
            DI = scipy.linalg.inv(D)
            dm = pdist(X, (lambda u, v: scipy.sqrt(((u-v)*DI*(u-v).T).sum())))
        elif metric == 'mahalanobis':
            V = scipy.cov(X.T)
            VI = scipy.linalg.inv(V)
            dm = pdist(X, (lambda u, v: scipy.sqrt(scipy.dot(scipy.dot((u-v),VI),(u-v).T).sum())))
        elif metric == 'test_cityblock':
            dm = pdist(X, (lambda u, v: abs(u-v).sum()))
        elif metric == 'test_minkowski':
            dm = pdist(X, (lambda u, v: math.pow((abs(u-v)**p).sum(), 1.0/p)))
        elif metric == 'test_cosine':
            dm = pdist(X, \
                       (lambda u, v: \
                        (1.0 - (scipy.dot(u, v.T) / \
                                (math.sqrt(scipy.dot(u, u.T)) * \
                                 math.sqrt(scipy.dot(v, v.T)))))))
        elif metric == 'test_correlation':
            dm = pdist(X, \
                       (lambda u, v: 1.0 - \
                        (scipy.dot(u - u.mean(), (v - v.mean()).T) / \
                         (math.sqrt(scipy.dot(u - u.mean(), \
                                              (u - u.mean()).T)) \
                          * math.sqrt(scipy.dot(v - v.mean(), \
                                                (v - v.mean()).T))))))
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
            raise AttributeError('Unknown Distance Metric: %s' % mstr)
    else:
        raise AttributeError('2nd argument metric must be a string identifier or a function.')
    return dm

def cophenet(*args, **kwargs):
    """
    d = cophenet(Z)

      Calculates the cophenetic distances between each observation in a
      hierarchical clustering defined by the linkage Z.

      Suppose p and q are original observations in disjoint clusters
      s and t, respectively and that s and t are joined by a direct
      parent cluster u. The cophenetic distance between observations
      i and j is simply the distance between clusters s and t.

      d is cophenetic distance matrix in condensed form. The ij'th
      entry is the cophenetic distance between original observations
      i and j.

    c = cophenet(Z, Y)

      Calculates the cophenetic correlation coefficient of a hierarchical
      clustering of a set of n observations in m dimensions. Returns the
      distance as a scalar. Y is the condensed distance matrix generated
      by pdist.

    (c, d) = cophenet(Z, Y, [])

      Same as cophenet(Z, Y) except the distance matrix is returned as
      the second element of a tuple.
      
    """
    nargs = len(args)

    if nargs < 1:
        raise AttributeError('At least one argument must be passed to cophenet.')
    Z = args[0]

    if (type(Z) is not _array_type) or Z.dtype != 'double':
        raise AttributeError('First argument Z must be an array of doubles.')
    Zs = Z.shape

    if len(Zs) != 2:
        raise AttributeError('First argument Z must be a 2-dimensional array.')

    if Zs[1] != 4:
        raise AttributeError('First argument Z must have exactly 4 columns.')
    
    n = Zs[0] + 1

    zz = scipy.zeros((n*(n-1)/2,), dtype='double')
    _cluster_wrap.cophenetic_distances_wrap(Z, zz, int(n))
    if nargs == 1:
        return zz

    Y = args[1]
    if (type(Y) is not _array_type) and Y.dtype != 'double':
        raise AttributeError('Second argument Y must be an array of doubles.')

    Ys = Y.shape

    if len(Ys) != 1:
        raise AttributeError('Second argument Y must be a 1-D array.')

    if Ys[0] != n*(n-1)/2:
        raise AttributeError('Incorrect size of Y. It must be a distance vector containing n*(n-1) elements.')
    
    z = zz.mean()
    y = Y.mean()
    Yy = Y - y
    Zz = zz - z
    #print Yy.shape, Zz.shape
    numerator = (Yy * Zz)
    denomA = Yy ** 2
    denomB = Zz ** 2
    c = numerator.sum() / scipy.sqrt((denomA.sum() * denomB.sum()))
    #print c, numerator.sum()
    if nargs == 2:
        return c

    if nargs == 3:
        return (c, zz)

def inconsistent(Z, d=2):
    """
    R = inconsistent(Z, d=2)
    
      Calculates statistics on links up to d levels below each
      non-singleton cluster defined in the (n-1)x4 linkage matrix Z.

      R is a (n-1)x5 matrix where the i'th row contains the link
      statistics for the non-singleton cluster i. The link statistics
      are computed over the link heights for links d levels below the
      cluster i. Z[i,0] and Z[i,1] are the mean and standard deviation of
      the link heights, respectively; Z[i,2] is the number of links
      included in the calculation; and Z[i,3] is the inconsistency
      coefficient, (Z[i, 2]-R[i,0])/R[i,2].

      The behavior of this function is very similar to the MATLAB(R)
      inconsistent function.
    """

    Zs = Z.shape
    if not is_valid_linkage(Z):
        raise AttributeError('The first argument Z is not a valid linkage.')
    if (not d == numpy.floor(d)) or d < 0:
        raise AttributeError('The second argument d must be a nonnegative integer value.')
    if d == 0:
        d = 1

    n = Zs[0] + 1
    R = scipy.zeros((n - 1, 4), dtype='double')

    _cluster_wrap.inconsistent_wrap(Z, R, int(n), int(d));
    return R
    
def from_mlab_linkage(Z):
    """
    Z2 = from_mlab_linkage(Z)
    
    Converts a linkage matrix Z generated by MATLAB(R) to a new linkage
    matrix Z2 compatible with this module. The conversion does two
    things:

     * the indices are converted from 1..N to 0..(N-1) form, and
    
     * a fourth column Z[:,3] is added where Z[i,3] is equal to
       the number of original observations (leaves) in the non-singleton
       cluster i.
    """

    if type(Z) is not _array_type:
        raise AttributeError('First argument Z must be a two-dimensional array.')
    if Z.dtype != 'double':
        raise AttributeError('First argument Z must contain doubles.')
    if Z.shape[1] != 3:
        raise AttributeError('First argument Z must have 3 columns.')
    if Z.shape[0] < 1:
        raise AttributeError('First argument Z must have at least one row.')

    Zs = Z.shape
    Zpart = Z[:,0:2]
    Zd = Z[:,2].reshape(Zs[0], 1)
    if Zpart.min() != 1.0 and Zpart.max() != 2 * Zs[0]:
        raise AttributeError('The format of the indices is not 1..N');
    CS = scipy.zeros((Zs[0], 1), dtype='double')
    Zpart = Zpart - 1
    _cluster_wrap.calculate_cluster_sizes_wrap(scipy.hstack([Zpart, \
                                                             Zd]), \
                                               CS, int(Zs[0]) + 1)
    return scipy.hstack([Zpart, Zd, CS])

def to_mlab_linkage(Z):
    """
    Z2 = to_mlab_linkage(Z)

    Converts a linkage matrix Z generated by the linkage function of this
    module to one compatible with matlab. Z2 is the same as Z with the last
    column removed and the indices converted to 1..N form.
    """
    if type(Z) is not _array_type:
        raise AttributeError('First argument Z must be a two-dimensional array.')
    if Z.dtype != 'double':
        raise AttributeError('First argument Z must contain doubles.')
    if Z.shape[1] != 4:
        raise AttributeError('First argument Z must have 4 columns.')
    if Z.shape[0] < 1:
        raise AttributeError('First argument Z must have at least one row.')
    
    return scipy.hstack([Z[:,0:2] + 1, Z[:,2]])

def is_linkage_monotonic(Z):
    """
      Returns True if the linkage Z is monotonic. The linkage is monotonic
      if for every cluster s and t joined, the distance between them is
      no less than the distance between any previously joined clusters.
    """
    if not is_valid_linkage(Z):
        raise AttributeError("The variable Z passed is not a valid linkage.")
    return (Z[:-1,2]-Z[1:,2] >= 0).any()

def is_valid_linkage(Z, warning=False, throw=False):
    """
    is_valid_linkage(Z, t)

      Returns True if Z is a valid linkage matrix. The variable must
      be a 2-dimensional double numpy array with n rows and 4 columns.
      The first two columns must contain indices between 0 and 2n-1. For a
      given row i, 0 <= Z[i,0] <= i+n-1 and 0 <= Z[i,1] <= i+n-1 (i.e.
      a cluster cannot join another cluster unless the cluster being joined
      has been generated.)
    """
    valid = type(Z) is _array_type
    valid = valid and Z.dtype == 'double'
    if valid:
        s = Z.shape
    valid = valid and len(s) == 2
    valid = valid and s[1] == 4
    if valid:
        n = s[0]
        valid = valid and (Z[:,0]-xrange(n-1, n*2-1) <= 0).any()
        valid = valid and (Z[:,1]-xrange(n-1, n*2-1) <= 0).any()
    return valid

def is_valid_y(y):
    """
    is_valid_y(y)

      Returns True if the variable y passed is a valid condensed
      distance matrix. Condensed distance matrices must be
      1-dimensional numpy arrays containing doubles. Their length
      must be a binomial coefficient (n choose 2) for some positive
      integer n.
    """
    valid = type(y) is _array_type
    valid = valid and y.dtype == 'double'
    if valid:
        s = y.shape
    valid = valid and len(s) == 1
    if valid:
        d = int(scipy.ceil(scipy.sqrt(s[0] * 2)))
        valid = valid and (d*(d-1)/2) == s[0]
    return valid

def is_valid_dm(D, t=0.0):
    """
    is_valid_dm(D)
    
      Returns True if the variable D passed is a valid distance matrix.
      Distance matrices must be 2-dimensional numpy arrays containing
      doubles. They must have a zero-diagnoal, and they must be symmetric.

    is_valid_dm(D, t)

      Returns True if the variable D passed is a valid distance matrix.
      Small numerical differences in D and D.T and non-zeroness of the
      diagonal are ignored if they are within the tolerance specified
      by t.
    """
    valid = type(D) is _array_type
    if valid:
        s = D.shape
    valid = valid and len(s) == 2
    valid = valid and s[0] == s[1]
    if t == 0.0:
        valid = valid and (D == D.T).all()
        valid = valid and (D[xrange(0, s[0]), xrange(0, s[0])] == 0).all()
    else:
        valid = valid and (D - D.T <= t).all()
        valid = valid and (D[xrange(0, s[0]), xrange(0, s[0])] <= t).all()
    return valid

def numobs_linkage(Z):
    """
    Returns the number of original observations that correspond to a
    linkage matrix Z.
    """
    if not is_valid_linkage(Z):
        raise AttributeError('Z is not a valid linkage.')
    return (Z.shape[0] - 1)

def numobs_dm(D):
    """
    Returns the number of original observations that correspond to a
    square, non-condensed distance matrix D.
    """
    if not is_valid_dm(D, tol=Inf):
        raise AttributeError('Z is not a valid linkage.')
    return D.shape[0]

def numobs_y(Y):
    """
    Returns the number of original observations that correspond to a
    condensed distance matrix Y.
    """
    if not is_valid_y(y):
        raise AttributeError('Z is not a valid condensed distance matrix.')
    d = int(scipy.ceil(scipy.sqrt(y.shape[0] * 2)))
    return d

def Z_y_correspond(Z, Y):
    """
    Returns True if a linkage matrix Z and condensed distance matrix
    Y could possibly correspond to one another. They must have the same
    number of original observations. This function is useful as a sanity
    check in algorithms that make use of many linkage and distance matrices.
    """
    return numobs_y(Y) == numobs_Z(Z)

def cluster(*args, **kwargs):
    """
    T = cluster(Z, 'cutoff', c)
    T = cluster(Z, cutoff=c)
    T = cluster(Z, 'cutoff', c, 'depth', d)
    T = cluster(Z, cutoff=c, depth=d)

    T = cluster(..., inconsistency=Q)

    T = cluster(
    """
    Z = args[0]

def cluster_cutoff(Z, R, cutoff):
    

def clusterdata(*args, **kwargs):
    """
    T = clusterdata(X, cutoff)

      Clusters the original observations in the n by m data matrix X
      (n observations in m dimensions) using the euclidean distance
      metric to calculate distances between original observations,
      the single linkage algorithm for hierarchical clustering, and
      the cut-off cluster formation algorithm to transform the linkage
      into flat clusters. The cutoff threshold is the maximum
      inconsistent value a node can have for membership in a cluster.

      A one-dimensional numpy array T of length n is returned. T[i]
      is the cluster group to which original observation i belongs.

    T = clusterdata(X, 'param1', val1, 'param2', val2, ...)

      Valid parameters (for paramX) include:
      
        'criterion': either 'inconsistent' or 'distance' cluster formation
                     algorithms. See cluster for descriptions.
           
        'linkage':   the linkage method to use. See linkage for
                     descriptions.

        'distance':  the distance metric for calculating pairwise
                     distances. See pdist for descriptions and
                     linkage to verify compatibility with the linkage
                     method.
                     
        'maxclust':  the maximum number of clusters to form. This
                     parameter is only valid for the distance
                     cluster formation algorithm.

        'depth':     the maximum depth for the inconsistency
                     calculation. See inconsistent for more information.

        'cutoff':    the threshold value to use for the cut-off cluster
                     formation algorithm. This value is ignored when the
                     'distance' algorithm is used instead.

     T = clusterdata(X, criterion='inconsistent', linkage='single', \
                     distance='euclidean', maxclust=X, depth=2, )

       Similar to the above but uses Python's more conveinent named
       argument syntax.

    """
    if len(args) < 2:
        raise AttributeError('At least 2 arguments are needed.')

    if len(args == 2):
        X = args[0]
        cutoff = args[1]
        Y = pdist(X, 'euclidean')
        Z = linkage(Y, 'single')
        T = cluster(Z, 'cutoff', cutoff)
    else:
        X = args[0]
        remArgs = args[1:]
        validParams = set(['criterion', 'linkage', 'distance', 'maxclust', \
                           'depth', 'cutoff'])
        parm = {'criterion': 'inconsistent',
                'linkage': 'single',
                'distance': 'euclidean',
                'maxclust': None,
                'depth': None,
                'cutoff': None}
        for i in xrange(0, len(remArgs), 2):
            param = validParams[i]
            val = validParams[i+1]
            if type(param) is not StringType:
                raise AttributeError('Expecting string for paramX argument to clusterdata.')
            if param not in validParams:
                raise AttributeError('Invalid parameter: %s valid ones are %s' % (param, str(validParams)))
            parm[param] = val
        metric = parm['distance']
        method = parm['linkage']
        Y = pdist(X, metric)
        Z = linkage(Y, method)
        if parm['criterion'] == 'inconsistent':
            if parm['cutoff'] is None:
                raise AttributeError("A cut-off threshold must be supplied if inconsistent criterion algorithm is used for cluster formation.")
            cutoff = parm['cutoff']
            if parm['depth'] is None:
                T = cluster(Z, 'cutoff', cutoff)
            else:
                depth = parm['depth']
                T = cluster(Z, 'cutoff', cutoff, 'depth', depth)
        else:
            if parm['maxclust'] is None:
                raise AttributeError("The maximum number of clusters must be supplied if the distance algorithm is used for cluster formation.")
            maxclust = parm['maxclust']
            T = cluster(Z, 'maxclust', maxclust)
    return T

def dendrogram(*args, **kwargs):
    """
    H = dendrogram(root, p=30)
    H = dendrogram(Z, p=30)

      Plots the hiearchical clustering defined by the linkage Z as a
      dendrogram. The dendrogram illustrates how each cluster is
      composed by drawing a U-shaped link between a non-singleton
      cluster and its descendents. The height of the top of a node
      is the distance between its descendents. It is also the cophenetic
      distance between original observations in the two descendent
      clusters.

      Corresponding to MATLAB behavior, if there are more than p
      leaf nodes in the data set, some nodes and their descendents
      are contracted into leaf nodes, leaving exactly p nodes in the
      plot.

      Returns a reference H to the list of line objects for this
      dendrogram.

    H = dendrogram(..., 'colorthreshold', t)
    H = dendrogram(..., colorthreshold=t)

      Colors all the links below a cluster node a unique color if it is
      the first node among its ancestors to have a distance below the
      threshold t. (An alternative named-argument syntax can be used.)

    (H,T) = dendrogram(..., 'get_leaves')
    (H,T) = dendrogram(..., get_leaves=True)
    
      Returns a tuple with the handle H and a m-sized int32 array T.
      The T[i] value is the leaf node index to which original observation
      with index i appears. This vector has duplicates only if m > p.
      (An alternative named-argument syntax can be used.)

    ... = dendrogram(..., 'orientation', 'orient')
    ... = dendrogram(..., orientation='top')

      Plots the dendrogram in a particular direction. The orientation
      parameter can be any of:

        * 'top': plots the root at the top, and plot descendent
          links going downwards. (default).
           
        * 'bottom': plots the root at the bottom, and plot descendent
          links going upwards.
           
        * 'left': plots the root at the left, and plot descendent
          links going right.

        * 'right': plots the root at the right, and plot descendent
          links going left.

    ... = dendrogram(..., 'labels', S)
    ... = dendrogram(..., labels=None)

        S is a p-sized list (or tuple) passed with the text of the labels
        to render by the leaf nodes. Passing None causes the index of
        the original observation to be used. A label only appears if
        its associated.

        (MLab features end here.)
        
    ... = dendrogram(..., leaves_order=None)

        Plots the leaves in the order specified by a vector of
        original observation indices. If the vector contains duplicates
        or results in a crossing, an exception will be thrown. Passing
        None orders leaf nodes based on the order they appear in the
        pre-order traversal.

    ... = dendrogram(..., count_sort=False)

        When plotting a cluster node and its directly descendent links,
        the order the two descendent links and their descendents are
        plotted is determined by the count_sort parameter. Valid values
        of count_sort are:

          * 'no'/False: nothing is done.
          
          * 'ascending'/True: the descendent with the minimum number of
          original objects in its cluster is plotted first.

          * 'descendent': the descendent with the maximum number of
          original objects in its cluster is plotted first.

    ... = dendrogram(..., distance_sort='no')

        When plotting a cluster node and its directly descendent links,
        the order the two descendent links and their descendents are
        plotted is determined by the distance_sort parameter. Valid
        values of count_sort are:

          * 'no'/False: nothing is done.

          * 'ascending'/True: the descendent with the minimum distance
          between its direct descendents is plotted first.

          * 'descending': the descendent with the maximum distance
          between its direct descendents is plotted first.

    ... = dendrogram(..., show_leaf_counts=False)

        When show_leaf_counts=True, leaf nodes representing k>1
        original observation are labeled with the number of observations
        they contain in parenthesis.
        
    """
    nargs = len(args)
    if nargs == 0:
        raise AttributeError('At least one argument is needed.')
    if nargs >= 1:
        if type(arg[0]) is _array_type:
            Z = arg[0]
            if not is_valid_linkage(Z):
                raise AttributeError('If the first argument is an array, it must be a valid linkage.')
            root = totree(Z)
        elif type(arg[0]) is _cnode_type:
            root = arg[0]
        else:
            raise AttributeError('The first argument must be a linkage array Z or a cnode of the root cluster.')
    restArgs = []
    ks = kwargs.keys()
    n = root.count
    p = min(30, n)
    if nargs >= 2:
        if type(arg[0]) is IntType:
            p = arg[0]
        elif type(arg[0]) is FloatType:
            p = int(arg[0])
        else:
            raise AttributeError('The second argument must be a number')
        restArgs = args[2:]
