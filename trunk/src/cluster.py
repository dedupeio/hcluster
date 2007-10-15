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
    """ Z=linkage(y, method)

          Performs hierarchical clustering on the condensed distance
          matrix y. y must be a n * (n - 1) sized vector where n is
          the number of original observations paired in the distance
          matrix. The behavior of this function is very similar to
          the MATLAB(R) linkage function.

          A (n - 1) * 4 matrix Z is returned. At the i'th iteration,
          clusters with indices Z[i, 0] and Z[i, 1] are combined to
          form cluster n + i. A cluster with an index less than n
          corresponds to one of the n original observations. The
          distance between clusters Z[i, 0] and Z[i, 1] is given by
          Z[i, 2]. The fourth value Z[i, 3] represents the number of
          original observations in the cluster n + i.

          The following linkage methods are used to compute the
          distance dist(s, t) between two clusters s and t. The
          algorithm begins with a forest of clusters that have yet
          to be used in the master hierarchy. When two clusters
          s and t from this forest are combined into a single
          cluster u, s and t are removed from the forest, and u
          appears in the forest. When only one cluster remains in the
          forest, the algorithm stops, and this cluster becomes
          the root.

          A distance matrix is maintained at each iteration. The
          d[i,j] entry corresponds to the distance between cluster
          i and j in the original forest.
          
          At each iteration, the algorithm must update the distance
          matrix to reflect the distance of the newly formed cluster
          u with the remaining clusters in the forest.
          
          Suppose there are |u| original observations u[0], ..., u[|u|-1]
          in cluster u and |v| original objects v[0], ..., v[|v|-1]
          in cluster v. Recall s and t are combined to form cluster
          u. Let v be any remaining cluster in the forest that is not
          u.

          The following are methods for calculating the distance between
          the newly formed cluster u and each v.
        
            * method='single' assigns dist(u,v) = MIN(dist(u[i],v[j])
              for all points i in cluster u and j in cluster v.

                (also called Nearest Point Algorithm)

            * method='complete' assigns dist(u,v) = MAX(dist(u[i],v[j])
              for all points i in cluster u and j in cluster v.

                (also called Farthest Point Algorithm
                      or the Voor Hees Algorithm)

           * method='average' assigns dist(u,v) =
                sum_{ij} { dist(u[i], v[j]) } / (|u|*|v|)
             for all points i and j where |u| and |v| are the
             cardinalities of clusters u and v, respectively.

                (also called UPGMA)

           * method='weighted' assigns

               dist(u,v) = (dist(s,v) + dist(t,v))/2
               
             where cluster u was formed with cluster s and t and v
             is a remaining cluster in the forest. (also called WPGMA)

        Z=linkage(X, method, metric='euclidean')

         Performs hierarchical clustering on the objects defined by the
         n by m observation matrix X.

         If the metric is 'euclidean' then the following methods may be
         used:

           * method='centroid' assigns dist(s,t) = euclid(c_s, c_t) where
             c_s and c_t are the centroids of clusters s and t,
             respectively. When two clusters s and t are combined into a new
             cluster u, the new centroid is computed over all the original
             objects in clusters s and t. The distance then becomes
             the Euclidean distance between the centroid of u and the
             centroid of a remaining cluster v in the forest.
             (also called UPGMC)
 
           * method='median' assigns dist(s,t) as above. When two clusters
             s and t are combined into a new cluster u, the average of
             centroids s and t give the new centroid u. (also called WPGMC)
           
           * method='ward' uses the Ward variance minimization algorithm.
             The new entry dist(u, v) is computed as follows,

                 dist(u,v) =

                ----------------------------------------------------
                | |v|+|s|            |v|+|t|            |v|
                | ------- d(v,s)^2 + ------- d(v,t)^2 - --- d(s,t)^2
               \|    T                  T                T

             where u is the newly joined cluster consisting of clusters
             s and t, v is an unused cluster in the forest, T=|v|+|s|+|t|,
             and |*| is the cardinality of its argument.
             (also called incremental)

           Warning to MATLAB(R) users: when the minimum distance pair in
           the forest is chosen, there may be two or more pairs with the
           same minimum distance. This implementation may chose a
           different minimum than the MATLAB version.
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

        Returns the number of leaf nodes (original observations)
        belonging to the cluster node nd. If the nd is a leaf, c=1.
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
    
          Performs preorder traversal without recursive function calls.
          When a leaf node is first encountered, func is called with the
          leaf node as the argument, and its result is appended to the
          list vlst.
    
          For example, the statement
        
            ids = root.preOrder(lambda x: x.id)
    
          returns a list of the node ids corresponding to the leaf
          nodes of the tree as they appear from left to right.
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
    
      Converts a hierarchical clustering encoded in the matrix Z
      (by linkage) into a tree. The reference r to the root cnode
      object is returned.
    
      Each cnode object has a left, right, dist, id, and count
      attribute. The left and right attributes point to cnode
      objects that were combined to generate the cluster. If
      both are None then the cnode object is a leaf node, its
      count must be 1, and its distance is meaningless but set
      to 0.

    (r, d) = totree(Z, rd=True)

      Same as totree(Z) except a tuple is returned where r is
      the reference to the root cnode and d is a reference to a
      dictionary mapping cluster ids to cnodes. If a cluster id
      is less than n, then it corresponds to a singleton cluster
      (leaf node).
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
    symmetry nor zero diagonals. This is useful if it is known that
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
    """ Y=pdist(X, method='euclidean', p=2)
    
           Computes the distance between m original observations in
           n-dimensional space. Returns a condensed distance matrix Y.
           For each i and j (i<j), the metric dist(u=X[i], v=X[j]) is
           computed and stored in the ij'th entry. See squareform
           to learn how to retrieve this entry.

        1. Y=pdist(X)

          Computes the distance between m points using Euclidean distance
          (2-norm) as the distance metric between the points. The points
          are arranged as m n-dimensional row vectors in the matrix X.

        2. Y=pdist(X, 'minkowski', p)

          Computes the distances using the Minkowski distance (p-norm)
          where p is a number.

        3. Y=pdist(X, 'cityblock')

          Computes the city block or manhattan distance between the
          points.

        4. Y=pdist(X, 'seuclidean')

          Computes the standardized euclidean distance so that the
          distances are of unit variance.

        5. Y=pdist(X, 'cosine')

          Computes the cosine distance between vectors u and v. This is
        
               1 - uv^T
             -----------
             |u|_2 |v|_2

          where |*|_2 is the 2 norm of its argument *.

        6. Y=pdist(X, 'correlation')

          Computes the correlation distance between vectors u and v. This is

            1 - (u - n|u|_1)(v - n|v|_1)^T
            --------------------------------- ,
            |(u - n|u|_1)|_2 |(v - n|v|_1)|^T

          where |*|_1 is the Manhattan (or 1-norm) of its argument *,
          and n is the common dimensionality of the vectors.

        7. Y=pdist(X, 'hamming')

          Computes the normalized Hamming distance, or the proportion
          of those vector elements between two vectors u and v which
          disagree. To save memory, the matrix X can be of type boolean.

        8. Y=pdist(X, 'jaccard')

          Computes the Jaccard distance between the points. Given two
          vectors, u and v, the Jaccard disaance is the proportion of
          those elements u_i and v_i that disagree where at least one
          of them is non-zero.

        9. Y=pdist(X, 'chebyshev')

          Computes the Chebyshev distance between the points. The
          Chebyshev distance between two vectors u and v is the maximum
          norm-1 distance between their respective elements. More
          precisely, the distance is given by

            d(u,v) = max_{i=1}^{n}{|u_i-v_i|}.

        10. Y=pdist(X, f)
        
          Computes the distance between all pairs of vectors in X
          using the user supplied 2-arity function f. For example,
          Euclidean distance between the vectors could be computed
          as follows,

            dm = pdist(X, (lambda u, v: scipy.sqrt(((u-v)*(u-v).T).sum())))

        11. Y=pdist(X, 'test_Y')

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

      Calculates the cophenetic distances between each observation in the
      hierarchical clustering defined by the linkage Z.

      Suppose p and q are original observations in disjoint clusters
      s and t, respectively and s and t are joined by a direct parent
      cluster u. The cophenetic distance between observations i and j
      is simply the distance between clusters s and t.

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
      cluster i. R[i,0] and R[i,1] are the mean and standard deviation of
      the link heights, respectively; R[i,2] is the number of links
      included in the calculation; and R[i,3] is the inconsistency
      coefficient, (Z[i, 2]-R[i,0])/R[i,2].

      The behavior of this function is very similar to the MATLAB(R)
      inconsistent function.
    """

    Zs = Z.shape
    if not is_valid_linkage(Z):
        raise AttributeError('The first argument Z is not a valid linkage.')
    if (not d == numpy.floor(d)) or d < 0:
        raise AttributeError('The second argument d must be a nonnegative integer value.')
#    if d == 0:
#        d = 1

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
                                                             Zd]).copy(), \
                                               CS, int(Zs[0]) + 1)
    return scipy.hstack([Zpart, Zd, CS]).copy()

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
    is_linkage_monotonic(Z)
    
      Returns True if the linkage Z is monotonic. The linkage is monotonic
      if for every cluster s and t joined, the distance between them is
      no less than the distance between any previously joined clusters.
    """
    if not is_valid_linkage(Z):
        raise AttributeError("The variable Z passed is not a valid linkage.")

    # We expect the i'th value to be greater than its successor.
    return (Z[:-1,2]>=Z[1:,2]).all()

def is_valid_im(R):
    """
    is_valid_im(R)
    
      Returns True if the inconsistency matrix passed is valid. It must
      be a n by 4 numpy array of doubles. The standard deviations R[:,1]
      must be nonnegative. The link counts R[:,2] must be positive and
      no greater than n-1.
    """
    valid = type(R) is _array_type
    valid = valid and R.dtype == 'double'
    valid = valid and len(R.shape) == 2
    valid = valid and R.shape[0] > 0
    valid = valid and R.shape[1] == 4
    return True

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

#def cluster(*args, **kwargs):

def cluster(Z, t, criterion='inconsistent', depth=2, R=None):
    """

    T = cluster(Z, t, criterion='inconsistent', d=2, R=None):

      Forms flat clusters from the hiearchical clustering defined by
      the linkage matrix Z. The threshold t is a required parameter.

      T is a vector of length n; T[i] is the cluster number to which the
      original observation i belongs. 

      The criterion parameter can be any of the following values,
      
        * 'inconsistent': A cluster node and all its decendents have an
        inconsistent value less than or equal to c iff all their leaf
        descendents belong to the same cluster. When no non-singleton
        cluster meets this criterion, every node is assigned to its
        own cluster. The d parameter is the maximum depth to perform
        the inconsistency calculation; it has no meaning for the other
        criteria.

        * 'distance': Forms flat clusters so that the original
        observations in each cluster has no greater a cophenetic
        distance than c.

        * 'maxclust': Finds a minimum threshold r such that the cophenetic
        distance between any two original observations in the same flat
        cluster is no more than r and no more than t flat clusters are
        formed.
    """
    if not is_valid_linkage(Z):
        raise AttributeError('Z is not a valid linkage matrix.')

    n = Z.shape[0] + 1
    T = scipy.zeros((n,), dtype='int32')
    if criterion == 'inconsistent':
        if R is None:
            R = inconsistent(Z, depth)
        else:
            if not is_valid_im(R):
                raise AttributeError('R passed is not a valid inconsistency matrix.')
        _cluster_wrap.cluster_in_wrap(Z, R, T, float(t), int(n), int(0))
    elif criterion == 'distance':
        if R is None:
            R = inconsistent(Z, depth)
        else:
            if not is_valid_im(R):
                raise AttributeError('R passed is not a valid inconsistency matrix.')
        _cluster_wrap.cluster_in_wrap(Z, R, T, float(t), int(n), int(1))
    else:
        raise AttributeError('Invalid cluster formation criterion: %s' % str(criterion))
    return T

def clusterdata(X, t, criterion='inconsistent', linkage='single', \
                distance='euclid', d=2):
    """
    T = clusterdata(X, t)

      Clusters the original observations in the n by m data matrix X
      (n observations in m dimensions) using the euclidean distance
      metric to calculate distances between original observations,
      performs hierarchical clustering using the single linkage
      algorithm, and forms flat clusters using the inconsistency
      method with t as the cut-off threshold.

      A one-dimensional numpy array T of length n is returned. T[i]
      is the index of the flat cluster to which the original
      observation i belongs.

    T = clusterdata(X, t, criterion='inconsistent', linkage='single',
                    dist='euclid', depth=2, R=None)

      Named parameters are described below.
      
        criterion:  specifies the criterion for forming flat clusters.
                    Valid values are 'inconsistent', 'distance', or
                    'maxclust' cluster formation algorithms. See
                    cluster for descriptions.
           
        lmethod:    the linkage method to use. See linkage for
                    descriptions.

        dmethod:    the distance metric for calculating pairwise
                    distances. See pdist for descriptions and
                    linkage to verify compatibility with the linkage
                    method.
                     
        t:          the cut-off threshold for the cluster function.

        depth:      the maximum depth for the inconsistency calculation.
                    See inconsistent for more information.

        R:          the inconsistency matrix.

     T = clusterdata(X, criterion='inconsistent', linkage='single', \
                     distance='euclidean', maxclust=X, depth=2, )

       Similar to the above but uses Python's more convenient named
       argument syntax.

    """

    if type(X) is not _array_type or len(X.shape) != 2:
        raise AttributeError('X must be an n by m numpy array.')

    Y = pdist(X, method=dmethod)
    Z = linkage(Y, method=lmethod)
    T = cluster(Z, criterion=criterion, depth=depth, R=R, t=t)
    return T

def prelist(Z):
    """
    L = prelist(Z):

      Returns a list of leaf node indices as they appear in the pre-order
      traversal of the tree defined by the linkage matrix Z.
    """
    if not is_valid_linkage(Z):
        raise AttributeError('Linkage matrix is not valid.')
    n = Z.shape[0] + 1
    ML = scipy.zeros((n,), dtype='int32')
    _cluster_wrap.prelist_wrap(Z, ML, int(n))
    return ML

# Let's do a conditional import. If matplotlib is not available, 
try:
    import matplotlib
    import matplotlib.pylab
    mpl = True
    def _plot_dendrogram(ivlines, dvlines, ivl, p, n, mh, orientation):
        axis = matplotlib.pylab.gca()
        # Independent variable plot width
        ivw = p * 10
        # Depenendent variable plot height
        dvw = mh + mh * 0.05
        ivticks = scipy.arange(0, p*10+5, 10)
        if orientation == 'top':
            axis.set_ylim([0, dvw])
            axis.set_xlim([0, ivw])
            xlines = ivlines
            ylines = dvlines
            axis.set_xticks(ivticks)
            axis.set_xticklabels(ivl)
        elif orientation == 'bottom':
            axis.set_ylim([dvw, 0])
            axis.set_xlim([0, ivw])
            ivl.reverse()
            xlines = ivlines
            ylines = dvlines
            axis.set_xticks(ivticks)
            axis.set_xticklabels(ivl)
        elif orientation == 'left':
            axis.set_xlim([0, dvw])
            axis.set_ylim([0, ivw])
            xlines = dvlines
            ylines = ivlines
            axis.set_yticks(ivticks)
            axis.set_yticklabels(ivl)
        elif orientation == 'right':
            axis.set_xlim([dvw, 0])
            axis.set_ylim([0, ivw])
            xlines = dvlines
            ylines = ivlines
            ivl.reverse()
            axis.set_yticks(ivticks)
            axis.set_yticklabels(ivl)
        for (xline,yline) in zip(xlines, ylines):
            line = matplotlib.lines.Line2D(xline, yline)
            axis.add_line(line)
        matplotlib.pylab.draw_if_interactive()
            
except ImportError:
    mpl = False
    def _plot_dendrogram(ivlines, dvlines, p, n, mh, orientation):
        raise AttributeError('matplotlib not available. Plot request denied.')

def dendrogram(Z, p=30, colorthreshold=scipy.inf, get_leaves=True,
               orientation='top', labels=None, count_sort=False,
               distance_sort=False, show_leaf_counts=True):
    """
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
      plot. The nodes chosen to be non-leaf nodes in the condensed
      dendrogram are the last p non-singleton clusters in the linkage
      Z (i.e. nodes corresponding to row vectors Z[n-p-2:end,:]).

      Returns a reference H to the list of line objects for this
      dendrogram.

    H = dendrogram(..., colorthreshold=t)

      Colors all the links below a cluster node a unique color if it is
      the first node among its ancestors to have a distance below the
      threshold t. (An alternative named-argument syntax can be used.)

    (H,T) = dendrogram(..., get_leaves=True)
    
      Returns a tuple with the handle H and a m-sized numpy array T of
      integer values. The T[i] value is the leaf node index in which
      original observation with index i appears. This vector has
      duplicates iff m > p.
      
      (An alternative named-argument syntax can be used.)

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

    ... = dendrogram(..., labels=None)

        S is a p-sized list (or tuple) passed with the text of the labels
        to render by the leaf nodes. Passing None causes the index of
        the original observation to be used. A label only appears if
        its associated leaf node corresponds to a singleton cluster.

        (MLab features end here.)
        
    ... = dendrogram(..., count_sort=False)

        When plotting a cluster node and its directly descendent links,
        the order the two descendent links and their descendents are
        plotted is determined by the count_sort parameter. Valid values
        of count_sort are:

          * False: nothing is done.
          
          * 'ascending'/True: the descendent with the minimum number of
          original objects in its cluster is plotted first.

          * 'descendent': the descendent with the maximum number of
          original objects in its cluster is plotted first.

    ... = dendrogram(..., distance_sort=False)

        When plotting a cluster node and its directly descendent links,
        the order the two descendent links and their descendents are
        plotted is determined by the distance_sort parameter. Valid
        values of count_sort are:

          * False: nothing is done.

          * 'ascending'/True: the descendent with the minimum distance
          between its direct descendents is plotted first.

          * 'descending': the descendent with the maximum distance
          between its direct descendents is plotted first.

        Note that either count_sort or distance_sort must be False.

    ... = dendrogram(..., show_leaf_counts)

        When show_leaf_counts=True, leaf nodes representing k>1
        original observation are labeled with the number of observations
        they contain in parenthesis.
        
    """

    # Features under consideration.
    #
    #         ... = dendrogram(..., leaves_order=None)
    #
    #         Plots the leaves in the order specified by a vector of
    #         original observation indices. If the vector contains duplicates
    #         or results in a crossing, an exception will be thrown. Passing
    #         None orders leaf nodes based on the order they appear in the
    #         pre-order traversal.

    if not is_valid_linkage(Z):
        raise AttributeError('If the first argument is an array, it must be a valid linkage.')
    Zs = Z.shape
    n = Zs[0] + 1
    if type(p) in (types.IntType, types.FloatType):
        p = int(p)
    else:
        raise AttributeError('The second argument must be a number')
    if p > n:
        p = n

    ivline_list=[]
    dvline_list=[]
    ivl=[]
    _dendrogram_calculate_info(Z=Z, p=p, \
                               colorthreshold=colorthreshold, \
                               get_leaves=get_leaves, \
                               orientation=orientation, \
                               labels=labels, \
                               count_sort=count_sort, \
                               distance_sort=distance_sort, \
                               show_leaf_counts=show_leaf_counts, \
                               i=2*n-2, iv=0.0, ivl=[], n=n, \
                               ivline_list=ivline_list, \
                               dvline_list=dvline_list)
    mh = max(Z[:,2])
    _plot_dendrogram(ivline_list, dvline_list, ivl, p, n, mh, orientation)

def _dendrogram_calculate_info(Z, p=30, colorthreshold=scipy.inf, get_leaves=True, \
                               orientation='top', labels=None, \
                               count_sort=False, distance_sort=False, \
                               show_leaf_counts=False, i=-1, iv=0.0, \
                               ivl=[], n=0, ivline_list=[], dvline_list=[]):
    """
    (l,w) = _dendrogram_calculate_info(Z, p=30, colorthreshold=inf, get_leaves=True,
               orientation='top', labels=None, count_sort=False,
               distance_sort=False, show_leaf_counts=False, i=0, iv=0.0,
               ivl=[], n=0, ivline_list=[], dvline_list=[]):

    Calculates the endpoints of the links as well as the labels for the
    the dendrogram rooted at the node with index i. iv is the independent
    variable value to plot the left-most leaf node below the root node i
    (if orientation='top', this would be the left-most x value where the
    plotting of this root node i and its descendents should begin).
    
    ivl is a list to store the labels of the leaf nodes. Nodes with an index
    below 2*n-p-1 are condensed into a leaf node. p is the maximum number
    of non-leaf nodes to plot.

    Returns a tuple with l being the independent variable coordinate that
    corresponds to the midpoint of cluster to the left of cluster i if
    i is non-singleton, otherwise the independent coordinate of the leaf
    node if i is a leaf node.

    w is the amount of space used in independent variable units.
    """
    if n == 0:
        raise AttributeError("Invalid singleton cluster count n.")

    if i == -1:
        raise AttributeError("Invalid root cluster index i.")

    # If the node is a leaf node but corresponds to a non-single cluster,
    # it's label is either the empty string or the number of original
    # observations belonging to cluster i.
    if i < 2*n-p and i >= n:
        if show_leaf_counts:
            ivl.append("(" + str(Z[i-n, 3]) + ")")
        else:
            ivl.append("")
        return (iv + 5.0, 10.0, 0.0)
    elif i < n:
        if labels is not None:
            ivl.append(labels[i-n])
        else:        
            ivl.append(str(i))
        return (iv + 5.0, 10.0, 0.0)
    elif i >= 2*n-p:
        # Actual indices of a and b
        aa = Z[i-n, 0]
        ab = Z[i-n, 1]
        if aa > n:
            # The number of singletons below cluster a
            na = Z[aa-n, 3]
            # The distance between a's two direct children.
            da = Z[aa-n, 2]
        else:
            na = 1
            da = 0.0
        if ab > n:
            nb = Z[ab-n, 3]
            db = Z[ab-n, 2]
        else:
            nb = 1
            da = 0.0

        if count_sort == 'ascending' or count_sort == True:
            # If a has a count greater than b, it and its descendents should
            # be drawn to the right. Otherwise, to the left.
            if na > nb:
                # The cluster index to draw to the left (ua) will be ab
                # and the one to draw to the right (ub) will be aa
                ua = ab
                ub = aa
            else:
                ua = aa
                ub = ab
        elif count_sort == 'descending':
            # If a has a count less than or equal to b, it and its
            # descendents should be drawn to the left. Otherwise, to
            # the right.
            if na <= nb:
                ua = aa
                ub = ab
            else:
                ua = ab
                ub = aa
        elif distance_sort == 'ascending' or distance_sort == True:
            # If a has a distance greater than b, it and its descendents should
            # be drawn to the right. Otherwise, to the left.
            if da > db:
                ua = ab
                ub = aa
            else:
                ua = aa
                ub = ab
        elif distance_sort == 'descending':
            # If a has a distance less than or equal to b, it and its
            # descendents should be drawn to the left. Otherwise, to
            # the right.
            if da <= db:
                ua = aa
                ub = ab
            else:
                ua = ab
                ub = aa
        else:
            ua = aa
            ub = ab

        # The distance of the cluster to draw to the left (ua) is uad
        # and its count is uan. Likewise, the cluster to draw to the
        # right has distance ubd and count ubn.
        if ua < n:
            uad = 0.0
            uan = 1
        else:
            uad = Z[ua-n, 2]
            uan = Z[ua-n, 3]
        if ub < n:
            ubd = 0.0
            ubn = 1
        else:
            ubd = Z[ub-n, 2]
            ubn = Z[ub-n, 3]

        # Updated iv variable and the amount of space used.
        (uiva, uwa, uah) = \
              _dendrogram_calculate_info(Z=Z, p=p, \
                                         colorthreshold=colorthreshold, \
                                         get_leaves=get_leaves, \
                                         orientation=orientation, \
                                         labels=labels, \
                                         count_sort=count_sort, \
                                         distance_sort=distance_sort, \
                                         show_leaf_counts=show_leaf_counts, \
                                         i=ua, iv=iv, ivl=ivl, n=n, \
                                         ivline_list=ivline_list, \
                                         dvline_list=dvline_list)
        (uivb, uwb, ubh) = \
              _dendrogram_calculate_info(Z=Z, p=p, \
                                         colorthreshold=colorthreshold, \
                                         get_leaves=get_leaves, \
                                         orientation=orientation, \
                                         labels=labels, \
                                         count_sort=count_sort, \
                                         distance_sort=distance_sort, \
                                         show_leaf_counts=show_leaf_counts, \
                                         i=ub, iv=iv+uwa, ivl=ivl, n=n, \
                                         ivline_list=ivline_list, \
                                         dvline_list=dvline_list)

        # The height of clusters a and b
        ah = uad
        bh = ubd
        h = Z[i-n, 2]

        ivline_list.append([uiva, uiva, uivb, uivb])
        dvline_list.append([uah, h, h, ubh])
        
        return ( ((uiva + uivb) / 2), uwa+uwb, h )

def is_cluster_isomorphic(T1, T2):
    """
      Returns True iff two different cluster assignments T1 and T2 are
      equivalent. T1 and T2 must be arrays of the same size.
    """
    if type(T1) is not _array_type:
        raise AttributeError('T1 must be a numpy array.')
    if type(T2) is not _array_type:
        raise AttributeError('T2 must be a numpy array.')

    T1S = T1.shape
    T2S = T2.shape

    if len(T1S) != 1:
        raise AttributeError('T1 must be one-dimensional.')
    if len(T2S) != 1:
        raise AttributeError('T2 must be one-dimensional.')
    if T1S[0] != T2S[0]:
        raise AttributeError('T1 and T2 must have the same number of elements.')
    n = T1S[0]
    d = {}
    for i in xrange(0,n):
        if T1[i] in d.keys():
            if d[T1[i]] != T2[i]:
                return False
        else:
            d[T1[i]] = T2[i]
    return True
    
