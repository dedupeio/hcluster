/**
 * cluster.c
 *
 * Author: Damian Eads
 * Date:   September 22, 2007
 *
 * Copyright (c) 2007, Damian Eads
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *   - Redistributions of source code must retain the above
 *     copyright notice, this list of conditions and the
 *     following disclaimer.
 *   - Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer
 *     in the documentation and/or other materials provided with the
 *     distribution.
 *   - Neither the name of the author nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#define NCHOOSE2(_n) ((_n)*(_n-1)/2)
#define ISCLUSTER(_nd) ((_nd)->id >= n)
#define GETCLUSTER(_id) ((lists + _id - n))

#define CPY_MAX(_x, _y) ((_x > _y) ? (_x) : (_y))
#define CPY_MIN(_x, _y) ((_x < _y) ? (_x) : (_y))
/** The number of link stats (for the inconsistency computation) for each
    cluster. */

#define CPY_NIS 4

/** The column offsets for the different link stats for the inconsistency
    computation. */
#define CPY_INS_MEAN 0
#define CPY_INS_STD 1
#define CPY_INS_N 2
#define CPY_INS_INS 3

/** The number of linkage stats for each cluster. */
#define CPY_LIS 4

/** The column offsets for the different link stats for the linkage matrix. */
#define CPY_LIN_LEFT 0
#define CPY_LIN_RIGHT 1
#define CPY_LIN_DIST 2
#define CPY_LIN_CNT 3

#define CPY_BITS_PER_CHAR (sizeof(unsigned char) * 8)
#define CPY_FLAG_ARRAY_SIZE_BYTES(num_bits) (CPY_CEIL_DIV((num_bits), \
                                                          CPY_BITS_PER_CHAR))
#define CPY_GET_BIT(_xx, i) (((_xx)[(i) / CPY_BITS_PER_CHAR] >> \
                             ((CPY_BITS_PER_CHAR-1) - \
                              ((i) % CPY_BITS_PER_CHAR))) & 0x1)
#define CPY_SET_BIT(_xx, i) ((_xx)[(i) / CPY_BITS_PER_CHAR] |= \
                              ((0x1) << ((CPY_BITS_PER_CHAR-1)-((i) % 8))))
#define CPY_CLEAR_BIT(_xx, i) ((_xx)[(i) / CPY_BITS_PER_CHAR] &= \
                              ~((0x1) << ((CPY_BITS_PER_CHAR-1)-((i) % 8))))

#ifndef CPY_CEIL_DIV
#define CPY_CEIL_DIV(x, y) ((((double)x)/(double)y) == \
                            ((double)((x)/(y))) ? ((x)/(y)) : ((x)/(y) + 1))
#endif

#include <malloc.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "cluster.h"

double euclidean_distance(const double *u, const double *v, int n) {
  int i = 0;
  double s = 0.0, d;
  for (i = 0; i < n; i++) {
    d = u[i] - v[i];
    s = s + d * d;
  }
  return sqrt(s);
}

double ess_distance(const double *u, const double *v, int n) {
  int i = 0;
  double s = 0.0, d;
  for (i = 0; i < n; i++) {
    d = fabs(u[i] - v[i]);
    s = s + d * d;
  }
  return s;
}

double chebyshev_distance(const double *u, const double *v, int n) {
  int i = 0;
  double d, maxv = 0.0;
  for (i = 0; i < n; i++) {
    d = fabs(u[i] - v[i]);
    if (d > maxv) {
      maxv = d;
    }
  }
  return maxv;
}

double mahalanobis_distance(const double *u, const double *v, int n) {
  int i = 0;
  double s = 0.0, d;
  for (i = 0; i < n; i++) {
    d = fabs(u[i] - v[i]);
    s = s + d;
  }
  return sqrt(s);
}

double hamming_distance(const double *u, const double *v, int n) {
  int i = 0;
  double s = 0.0;
  for (i = 0; i < n; i++) {
    s = s + (u[i] != v[i]);
  }
  return s / (double)n;
}

double hamming_distance_bool(const char *u, const char *v, int n) {
  int i = 0;
  double s = 0.0;
  for (i = 0; i < n; i++) {
    s = s + (u[i] != v[i]);
  }
  return s / (double)n;
}

double jaccard_distance(const double *u, const double *v, int n) {
  int i = 0;
  double denom = 0.0, num = 0.0;
  for (i = 0; i < n; i++) {
    num += (u[i] != v[i]) && ((u[i] != 0) || (v[i] != 0));
    denom += (u[i] != 0) || (v[i] != 0);
  }
  return num / denom;
}

double jaccard_distance_bool(const char *u, const char *v, int n) {
  int i = 0;
  double s = 0.0;
  for (i = 0; i < n; i++) {
    s = s + (u[i] != v[i]);
  }
  return s / (double)n;
}

double dot_product(const double *u, const double *v, int n) {
  int i;
  double s = 0.0;
  for (i = 0; i < n; i++) {
    s += u[i] * v[i];
  }
  return s;
}

double dot_product_sub(const double *u, double ub,
		       const double *v, double vb, int n) {
  int i;
  double s = 0.0;
  for (i = 0; i < n; i++) {
    s = s + ((u[i] - ub) * (v[i] - vb));
  }
  return s;
}

double vector_2norm(const double *u, int n) {
  return sqrt(dot_product(u, u, n));
}

double vector_mean(const double *u, int n) {
  int i;
  double s = 0.0;
  for (i = 0; i < n; i++) {
    s = s + u[i];
  }
  return s / (double)n;
}

double cosine_distance(const double *u, const double *v, int n,
		       const double nu, const double nv) {
  return 1.0 - (dot_product(u, v, n) / (nu * nv));
}

double seuclidean_distance(const double *var,
			   const double *u, const double *v, int n) {
  int i = 0;
  double s = 0.0, d;
  for (i = 0; i < n; i++) {
    d = u[i] - v[i];
    s = s + (d * d) / var[i];
  }
  return sqrt(s);
}

double city_block_distance(const double *u, const double *v, int n) {
  int i = 0;
  double s = 0.0, d;
  for (i = 0; i < n; i++) {
    d = fabs(u[i] - v[i]);
    s = s + d;
  }
  return s;
}

double minkowski_distance(const double *u, const double *v, int n, double p) {
  int i = 0;
  double s = 0.0, d;
  for (i = 0; i < n; i++) {
    d = fabs(u[i] - v[i]);
    s = s + pow(d, p);
  }
  return pow(s, 1.0 / p);
}

void compute_mean_vector(double *res, const double *X, int m, int n) {
  int i, j;
  const double *v;
  for (i = 0; i < n; i++) {
    res[i] = 0.0;
  }
  for (j = 0; j < m; j++) {

    v = X + (j * n);
    for (i = 0; i < n; i++) {
      res[i] += v[i];
    }
  }
  for (i = 0; i < n; i++) {
    res[i] /= (double)m;
  }
}

void vector_subtract(double *result, const double *u, const double *v, int n) {
  int i;
  for (i = 0; i < n; i++) {
    result[i] = u[i] - v[i];
  }
}

void pdist_euclidean(const double *X, double *dm, int m, int n) {
  int i, j;
  const double *u, *v;
  double *it = dm;
  for (i = 0; i < m; i++) {
    for (j = i + 1; j < m; j++, it++) {
      u = X + (n * i);
      v = X + (n * j);
      *it = euclidean_distance(u, v, n);
    }
  }
}

void pdist_hamming(const double *X, double *dm, int m, int n) {
  int i, j;
  const double *u, *v;
  double *it = dm;
  for (i = 0; i < m; i++) {
    for (j = i + 1; j < m; j++, it++) {
      u = X + (n * i);
      v = X + (n * j);
      *it = hamming_distance(u, v, n);
    }
  }
}

void pdist_hamming_bool(const char *X, double *dm, int m, int n) {
  int i, j;
  const char *u, *v;
  double *it = dm;
  for (i = 0; i < m; i++) {
    for (j = i + 1; j < m; j++, it++) {
      u = X + (n * i);
      v = X + (n * j);
      *it = hamming_distance_bool(u, v, n);
    }
  }
}

void pdist_jaccard(const double *X, double *dm, int m, int n) {
  int i, j;
  const double *u, *v;
  double *it = dm;
  for (i = 0; i < m; i++) {
    for (j = i + 1; j < m; j++, it++) {
      u = X + (n * i);
      v = X + (n * j);
      *it = jaccard_distance(u, v, n);
    }
  }
}

void pdist_jaccard_bool(const char *X, double *dm, int m, int n) {
  int i, j;
  const char *u, *v;
  double *it = dm;
  for (i = 0; i < m; i++) {
    for (j = i + 1; j < m; j++, it++) {
      u = X + (n * i);
      v = X + (n * j);
      *it = jaccard_distance_bool(u, v, n);
    }
  }
}


void pdist_chebyshev(const double *X, double *dm, int m, int n) {
  int i, j;
  const double *u, *v;
  double *it = dm;
  for (i = 0; i < m; i++) {
    for (j = i + 1; j < m; j++, it++) {
      u = X + (n * i);
      v = X + (n * j);
      *it = chebyshev_distance(u, v, n);
    }
  }
}

void pdist_cosine(const double *X, double *dm, int m, int n, const double *norms) {
  int i, j;
  const double *u, *v;
  double *it = dm;
  for (i = 0; i < m; i++) {
    for (j = i + 1; j < m; j++, it++) {
      u = X + (n * i);
      v = X + (n * j);
      *it = cosine_distance(u, v, n, norms[i], norms[j]);
    }
  }
}

void pdist_seuclidean(const double *X, const double *var,
		     double *dm, int m, int n) {
  int i, j;
  const double *u, *v;
  double *it = dm;
  for (i = 0; i < m; i++) {
    for (j = i + 1; j < m; j++, it++) {
      u = X + (n * i);
      v = X + (n * j);
      *it = seuclidean_distance(var, u, v, n);
    }
  }
}

void pdist_city_block(const double *X, double *dm, int m, int n) {
  int i, j;
  const double *u, *v;
  double *it = dm;
  for (i = 0; i < m; i++) {
    for (j = i + 1; j < m; j++, it++) {
      u = X + (n * i);
      v = X + (n * j);
      *it = city_block_distance(u, v, n);
    }
  }
}

void pdist_minkowski(const double *X, double *dm, int m, int n, double p) {
  int i, j;
  const double *u, *v;
  double *it = dm;
  for (i = 0; i < m; i++) {
    for (j = i + 1; j < m; j++, it++) {
      u = X + (n * i);
      v = X + (n * j);
      *it = minkowski_distance(u, v, n, p);
    }
  }
}


void chopmins(int *ind, int mini, int minj, int np) {
  int i;
  /**  if (mini < np - 2) {**/
  for (i = mini; i < minj - 1; i++) {
    ind[i] = ind[i + 1];
  }
  /**}**/
  /**  if (minj < np - 2) {**/
  for (i = minj - 1; i < np - 2; i++) {
    ind[i] = ind[i + 2];
  }
  /**  }**/
  /**  if (np > 0) {
    ind[np - 1] = HUGE_VALF;
  }
  if (np > 1) {
    ind[np - 2] = INFINITY;
    }***/
  /**  fprintf(stderr, "[Remove mini=%d minj=%d]\n", mini, minj);**/
}

void chopmin(int *ind, int minj, int np) {
  int i;
  for (i = minj; i < np - 1; i++) {
    ind[i] = ind[i + 1];
  }
  /**  }**/
  /**  if (np > 0) {
    ind[np - 1] = HUGE_VALF;
  }
  if (np > 1) {
    ind[np - 2] = INFINITY;
    }***/
  /**  fprintf(stderr, "[Remove mini=%d minj=%d]\n", mini, minj);**/
}

void chopmins_ns_ij(double *ind, int mini, int minj, int np) {
  int i;
  /**if (mini < np - 2) {**/
  for (i = mini; i < minj - 1; i++) {
    ind[i] = ind[i + 1];
  }
  /**}**/
  /**if (minj < np - 2) {**/
  for (i = minj - 1; i < np - 2; i++) {
    ind[i] = ind[i + 2];
  }
  /**}**/
  /**  if (np > 0) {
    ind[np - 1] = INFINITY;
  }
  if (np > 1) {
    ind[np - 2] = INFINITY;
    }**/
}

void chopmins_ns_i(double *ind, int mini, int np) {
  int i;
    for (i = mini; i < np - 1; i++) {
      ind[i] = ind[i + 1];
    }
    /**  if (np > 0) {
    ind[np - 1] = INFINITY;
    }**/
}

void dist_single(cinfo *info, int mini, int minj, int np, int n) {
  double **rows = info->rows;
  double *buf = info->buf;
  double *bit;
  int i;
  bit = buf;
  for (i = 0; i < mini; i++, bit++) {
    *bit = CPY_MIN(*(rows[i] + mini - i - 1), *(rows[i] + minj - i - 1));
  }
  for (i = mini + 1; i < minj; i++, bit++) {
    *bit = CPY_MIN(*(rows[mini] + i - mini - 1), *(rows[i] + minj - i - 1));
  }
  for (i = minj + 1; i < np; i++, bit++) {
    *bit = CPY_MIN(*(rows[mini] + i - mini - 1), *(rows[minj] + i - minj - 1));
  }
  /**  fprintf(stderr, "[");
  for (i = 0; i < np - 2; i++) {
    fprintf(stderr, "%5.5f ", buf[i]);
  }
  fprintf(stderr, "]");**/
}

void dist_complete(cinfo *info, int mini, int minj, int np, int n) {
  double **rows = info->rows;
  double *buf = info->buf;
  double *bit;
  int i;
  bit = buf;
  for (i = 0; i < mini; i++, bit++) {
    *bit = CPY_MAX(*(rows[i] + mini - i - 1), *(rows[i] + minj - i - 1));
  }
  for (i = mini + 1; i < minj; i++, bit++) {
    *bit = CPY_MAX(*(rows[mini] + i - mini - 1), *(rows[i] + minj - i - 1));
  }
  for (i = minj + 1; i < np; i++, bit++) {
    *bit = CPY_MAX(*(rows[mini] + i - mini - 1), *(rows[minj] + i - minj - 1));
  }
}

void dist_average(cinfo *info, int mini, int minj, int np, int n) {
  double **rows = info->rows, *buf = info->buf, *bit;
  int *inds = info->ind;
  double drx, dsx, mply, rscnt, rc, sc;
  int i, xi, xn;
  cnode *rn = info->nodes + inds[mini];
  cnode *sn = info->nodes + inds[minj];
  bit = buf;
  rc = (double)rn->n;
  sc = (double)sn->n;
  rscnt = rc + sc;

  for (i = 0; i < mini; i++, bit++) {
    /** d(r,x) **/
    drx = *(rows[i] + mini - i - 1);
    dsx = *(rows[i] + minj - i - 1);
    xi = inds[i];
    cnode *xnd = info->nodes + xi;
    xn = xnd->n;
    mply = 1.0 / (((double)xn) * rscnt);
    *bit = mply * ((drx * (rc * xn)) + (dsx * (sc * xn)));
  }
  for (i = mini + 1; i < minj; i++, bit++) {
    drx = *(rows[mini] + i - mini - 1);
    dsx = *(rows[i] + minj - i - 1);
    xi = inds[i];
    cnode *xnd = info->nodes + xi;
    xn = xnd->n;
    mply = 1.0 / (((double)xn) * rscnt);
    *bit = mply * ((drx * (rc * xn)) + (dsx * (sc * xn)));
  }
  for (i = minj + 1; i < np; i++, bit++) {
    drx = *(rows[mini] + i - mini - 1);
    dsx = *(rows[minj] + i - minj - 1);
    xi = inds[i];
    cnode *xnd = info->nodes + xi;
    xn = xnd->n;
    mply = 1.0 / (((double)xn) * rscnt);
    *bit = mply * ((drx * (rc * xn)) + (dsx * (sc * xn)));
  }
}

void dist_centroid(cinfo *info, int mini, int minj, int np, int n) {
  double *buf = info->buf, *bit;
  int *inds = info->ind;
  const double *centroid_tq;
  int i, m, xi;
  centroid_tq = info->centroids[info->nid];
  bit = buf;
  m = info->m;
  for (i = 0; i < np; i++, bit++) {
    /** d(r,x) **/
    if (i == mini || i == minj) {
      bit--;
      continue;
    }
    xi = inds[i];
    *bit = euclidean_distance(info->centroids[xi], centroid_tq, m);
    /**    fprintf(stderr, "%5.5f ", *bit);**/
  }
  /**  fprintf(stderr, "\n");**/
}

void combine_centroids(double *centroidResult,
		       const double *centroidA, const double *centroidB,
		       double na, double nb, int n) {
  int i;
  double nr = (double)na + (double)nb;
  for (i = 0; i < n; i++) {
    centroidResult[i] = ((centroidA[i] * na) + (centroidB[i] * nb)) / nr;
  }
}

void dist_weighted(cinfo *info, int mini, int minj, int np, int n) {
  double **rows = info->rows, *buf = info->buf, *bit;
  int i;
  double drx, dsx;

  bit = buf;

  for (i = 0; i < mini; i++, bit++) {
    /** d(r,x) **/
    drx = *(rows[i] + mini - i - 1);
    dsx = *(rows[i] + minj - i - 1);
    *bit = (drx + dsx) / 2;
  }
  for (i = mini + 1; i < minj; i++, bit++) {
    drx = *(rows[mini] + i - mini - 1);
    dsx = *(rows[i] + minj - i - 1);
    *bit = (drx + dsx) / 2;
  }
  for (i = minj + 1; i < np; i++, bit++) {
    drx = *(rows[mini] + i - mini - 1);
    dsx = *(rows[minj] + i - minj - 1);
    *bit = (drx + dsx) / 2;
  }
  /**  fprintf(stderr, "\n");**/
}

void dist_ward(cinfo *info, int mini, int minj, int np, int n) {
  double **rows = info->rows, *buf = info->buf, *bit;
  int *inds = info->ind;
  const double *centroid_tq;
  int i, m, xi, rind, sind;
  double drx, dsx, rf, sf, xf, xn, rn, sn, drsSq;
  cnode *newNode;

  rind = inds[mini];
  sind = inds[minj];
  rn = (double)info->nodes[rind].n;
  sn = (double)info->nodes[sind].n;
  newNode = info->nodes + info->nid;
  drsSq = newNode->d;
  drsSq = drsSq * drsSq;
  centroid_tq = info->centroids[info->nid];
  bit = buf;
  m = info->m;

  for (i = 0; i < mini; i++, bit++) {
    /** d(r,x) **/
    drx = *(rows[i] + mini - i - 1);
    dsx = *(rows[i] + minj - i - 1);
    xi = inds[i];
    cnode *xnd = info->nodes + xi;
    xn = xnd->n;
    rf = (rn + xn) / (rn + sn + xn);
    sf = (sn + xn) / (rn + sn + xn);
    xf = -xn / (rn + sn + xn);
    *bit = sqrt(rf * (drx * drx) +
		sf * (dsx * dsx) +
		xf * drsSq);
		
  }
  for (i = mini + 1; i < minj; i++, bit++) {
    drx = *(rows[mini] + i - mini - 1);
    dsx = *(rows[i] + minj - i - 1);
    xi = inds[i];
    cnode *xnd = info->nodes + xi;
    xn = xnd->n;
    rf = (rn + xn) / (rn + sn + xn);
    sf = (sn + xn) / (rn + sn + xn);
    xf = -xn / (rn + sn + xn);
    *bit = sqrt(rf * (drx * drx) +
		sf * (dsx * dsx) +
		xf * drsSq);
  }
  for (i = minj + 1; i < np; i++, bit++) {
    drx = *(rows[mini] + i - mini - 1);
    dsx = *(rows[minj] + i - minj - 1);
    xi = inds[i];
    cnode *xnd = info->nodes + xi;
    xn = xnd->n;
    rf = (rn + xn) / (rn + sn + xn);
    sf = (sn + xn) / (rn + sn + xn);
    xf = -xn / (rn + sn + xn);
    *bit = sqrt(rf * (drx * drx) +
		sf * (dsx * dsx) +
		xf * drsSq);
  }
  /**  fprintf(stderr, "\n");**/
}


void print_dm(const double **rows, int np) {
  int i, j, k;
  const double *row;
  fprintf(stderr, "[DM, np=%d\n", np);
  for (i = 0; i < np - 1; i++) {
    row = rows[i];
    for (j = 0; j <= i; j++) {
      fprintf(stderr, "%5.5f ", 0.0);
    }

    for (k = 0, j = i + 1; j < np; j++, k++) {
      fprintf(stderr, "%5.5f ", *(row + k));
    }
    fprintf(stderr, "|j=%d|\n", i + 1);
  }
}

void print_ind(const int *inds, int np) {
  int i;
  fprintf(stderr, "[IND, np=%d || ", np);
  for (i = 0; i < np; i++) {
    fprintf(stderr, "%d ", inds[i]);
  }
  fprintf(stderr, "]\n");
}

void print_vec(const double *d, int n) {
  int i;
  fprintf(stderr, "[");
  for (i = 0; i < n; i++) {
    fprintf(stderr, "%5.5f ", d[i]);
  }
  fprintf(stderr, "]");
}

/**
 * notes to self:
 * dm:    The distance matrix.
 * Z:     The result of the linkage, a (n-1) x 3 matrix.
 * X:     The original observations as row vectors (=NULL if not needed).
 * n:     The number of objects.
 * ml:    A boolean indicating whether a list of objects in the forest
 *        clusters should be maintained.
 * kc:    Keep track of the centroids.
 */
void linkage(double *dm, double *Z, double *X,
	     int m, int n, int ml, int kc, distfunc dfunc,
	     int method) {
  int i, j, k, t, np, nid, mini, minj, npc2;
  double min, ln, rn, qn;
  int *ind;
  /** An iterator through the distance matrix. */
  double *dmit, *buf;

  int *rowsize;

  /** Temporary array to store modified distance matrix. */
  double *dmt, **rows, *Zrow;
  double *centroidsData;
  double **centroids;
  const double *centroidL, *centroidR;
  double *centroid;
  clist *lists, *listL, *listR, *listC;
  clnode *lnodes;
  cnode *nodes, *node;

  cinfo info;

  /** The next two are only necessary for euclidean distance methods. */
  if (ml) {
    lists = (clist*)malloc(sizeof(clist) * (n-1));
    lnodes = (clnode*)malloc(sizeof(clnode) * n);
  }
  else {
    lists = 0;
    lnodes = 0;
  }
  if (kc) {
    centroids = (double**)malloc(sizeof(double*) * (2 * n));
    centroidsData = (double*)malloc(sizeof(double) * n * m);
    for (i = 0; i < n; i++) {
      centroids[i] = X + i * m;
    }
    for (i = 0; i < n; i++) {
      centroids[i+n] = centroidsData + i * m;
    }
  }
  else {
    centroids = 0;
    centroidsData = 0;
  }

  nodes = (cnode*)malloc(sizeof(cnode) * (n * 2) - 1);
  ind = (int*)malloc(sizeof(int) * n);
  dmt = (double*)malloc(sizeof(double) * NCHOOSE2(n));
  buf = (double*)malloc(sizeof(double) * n);
  rows = (double**)malloc(sizeof(double*) * n);
  rowsize = (int*)malloc(sizeof(int) * n);
  memcpy(dmt, dm, sizeof(double) * NCHOOSE2(n));

  info.X = X;
  info.m = m;
  info.n = n;
  info.nodes = nodes;
  info.ind = ind;
  info.dmt = dmt;
  info.buf = buf;
  info.rows = rows;
  info.rowsize = rowsize;
  info.dm = dm;
  info.centroids = centroids;
  if (kc) {
    info.centroidBuffer = centroids[2*n - 1];
  }
  else {
    info.centroidBuffer = 0;
  }
  info.lists = lists;
  for (i = 0; i < n; i++) {
    ind[i] = i;
    node = nodes + i;
    node->left = 0;
    node->right = 0;
    node->id = i;
    node->n = 1;
    node->d = 0.0;
    rowsize[i] = n - 1 - i;
  }
  rows[0] = dmt;
  for (i = 1; i < n; i++) {
    rows[i] = rows[i-1] + n - i;
  }
  
  if (ml) {
    for (i = 0; i < n; i++) {
      (lnodes + i)->val = nodes + i;
      (lnodes + i)->next = 0;
    }
  }

  for (k = 0, nid = n; k < n - 1; k++, nid++) {
    info.nid = nid;
    np = n - k;
    npc2 = NCHOOSE2(np);
    /**    fprintf(stderr, "k=%d, nid=%d, n=%d np=%d\n", k, nid, n, np);**/
    min = dmt[0];
    mini = 0;
    minj = 1;
    /** Note that mini < minj since j > i is always true. */
    for (i = 0; i < np - 1; i++) {
      dmit = rows[i];
      for (j = i + 1; j < np; j++, dmit++) {
	if (*dmit <= min) {
	  min = *dmit;
	  mini = i;
	  minj = j;
	}
      }
    }

    node = nodes + nid;
    node->left = nodes + ind[mini];
    node->right = nodes + ind[minj];
    ln = (double)node->left->n;
    rn = (double)node->right->n;
    qn = ln + rn;
    node->n = node->left->n + node->right->n;
    node->d = min;
    node->id = nid;

    Zrow = Z + (k * CPY_LIS);
    Zrow[CPY_LIN_LEFT] = node->left->id;
    Zrow[CPY_LIN_RIGHT] = node->right->id;
    Zrow[CPY_LIN_DIST] = min;
    Zrow[CPY_LIN_CNT] = node->n;

    /**    fprintf(stderr,
	    "[lid=%d, rid=%d, llid=%d, rrid=%d m=%5.8f]",
	    node->left->id, node->right->id, ind[mini], ind[minj], min);**/

    if (ml) {
      listC = GETCLUSTER(nid);
      if (ISCLUSTER(node->left) != 0) {
	listL = GETCLUSTER(node->left->id);
	if (ISCLUSTER(node->right) != 0) {
	  listR = GETCLUSTER(node->right->id);
	  listL->tail->next = listR->head;
	  listC->tail = listR->tail;
	  listR->tail->next = 0;
	}
	else {
	  listC->tail = lnodes + node->right->id;
	  listL->tail->next = listC->tail;
	  listC->tail->next = 0;
	}
	listC->head = listL->head;
      }
      else {
	listC->head = lnodes + node->left->id;
	if (ISCLUSTER(node->right)) {
	  listR = GETCLUSTER(node->right->id);
	  listC->head->next = listR->head;
	  listC->tail = listR->tail;
	  listC->tail->next = 0;
	}
	else {
	  listC->tail = lnodes + node->right->id;
	  listC->tail->next = 0;
	  listC->head->next = listC->tail;
	}
      }
    }
    if (kc) {
      centroidL = centroids[ind[mini]];
      centroidR = centroids[ind[minj]];
      centroid = centroids[nid];
      switch(method) {
      case CPY_LINKAGE_MEDIAN:
	for (t = 0; t < m; t++) {
	  centroid[t] = (centroidL[t] * 0.5 + centroidR[t] * 0.5);
	}
	break;
      case CPY_LINKAGE_CENTROID:
      case CPY_LINKAGE_WARD:
      default:
	for (t = 0; t < m; t++) {
	  centroid[t] = (centroidL[t] * ln + centroidR[t] * rn) / qn;
	}
	break;
      }
      /**      fprintf(stderr, "L: ");
      print_vec(centroidL, m);
      fprintf(stderr, "\nR: ");
      print_vec(centroidR, m);
      fprintf(stderr, "\nT: ");
      print_vec(centroid, m);**/
    }

    /**    print_dm(rows, np);**/
    /**    dfunc(buf, rows, mini, minj, np, dm, n, ind, nodes);**/
    dfunc(&info, mini, minj, np, n);

    /** For these rows, we must remove, i and j but leave all unused space
        at the end. This reduces their size by two.*/
    for (i = 0; i < mini; i++) {
      chopmins_ns_ij(rows[i], mini - i - 1, minj - i - 1, rowsize[i]);
    }

    /** We skip the i'th row. For rows i+1 up to j-1, we just remove j. */
    for (i = mini + 1; i < minj; i++) {
      chopmins_ns_i(rows[i], minj - i - 1, rowsize[i]);
    }

    /** For rows 0 to mini - 1, we move them down the matrix, leaving the
	first row free. */
    /**    for (i = mini; i > 0; i--) {
      memcpy(rows[i], rows[i-1], sizeof(double) * rowsize[i]-k);
      }**/

    for (i = mini; i < minj - 1; i++) {
      memcpy(rows[i], rows[i+1], sizeof(double) * (rowsize[i+1]));
    }

    /** For rows mini+1 to minj-1, we do nothing since they are in the
	right place for the next iteration. For rows minj+1 onward,
	we move them to the right. */
	
    for (i = minj - 1; i < np - 2; i++) {
      memcpy(rows[i], rows[i+2], sizeof(double) * (rowsize[i+2]));
    }

    /** Rows i+1 to j-1 lose one unit of space, so we move them up. */
    /** Rows j to np-1 lose no space. We do nothing to them. */

    /**    memcpy(rows[0], buf, sizeof(double) * rowsize[0] - k);*/

    for (i = 0; i < np - 2; i++) {
      *(rows[i] + np - 3 - i) = buf[i];
    }

    /**    print_dm(rows, np - 1);
	   print_ind(ind, np);**/
    chopmins(ind, mini, minj, np);
    ind[np - 2] = nid;
    /**    print_ind(ind, np - 1);**/
  }
  free(lists);
  free(lnodes);
  free(nodes);
  free(ind);
  free(dmt);
  free(buf);
  free(rows);
  free(rowsize);
  free(centroidsData);
  free(centroids);
}

void linkage_alt(double *dm, double *Z, double *X,
	     int m, int n, int ml, int kc, distfunc dfunc,
	     int method) {
  int i, j, k, t, np, nid, mini, minj, npc2;
  double min, ln, rn, qn;
  int *ind;
  /** An iterator through the distance matrix. */
  double *dmit, *buf;

  int *rowsize;

  /** Temporary array to store modified distance matrix. */
  double *dmt, **rows, *Zrow;
  double *centroidsData;
  double **centroids;
  const double *centroidL, *centroidR;
  double *centroid;
  clist *lists, *listL, *listR, *listC;
  clnode *lnodes;
  cnode *nodes, *node;

  cinfo info;

  /** The next two are only necessary for euclidean distance methods. */
  if (ml) {
    lists = (clist*)malloc(sizeof(clist) * (n-1));
    lnodes = (clnode*)malloc(sizeof(clnode) * n);
  }
  else {
    lists = 0;
    lnodes = 0;
  }
  if (kc) {
    centroids = (double**)malloc(sizeof(double*) * (2 * n));
    centroidsData = (double*)malloc(sizeof(double) * n * m);
    for (i = 0; i < n; i++) {
      centroids[i] = X + i * m;
    }
    for (i = 0; i < n; i++) {
      centroids[i+n] = centroidsData + i * m;
    }
  }
  else {
    centroids = 0;
    centroidsData = 0;
  }

  nodes = (cnode*)malloc(sizeof(cnode) * (n * 2) - 1);
  ind = (int*)malloc(sizeof(int) * n);
  dmt = (double*)malloc(sizeof(double) * NCHOOSE2(n));
  buf = (double*)malloc(sizeof(double) * n);
  rows = (double**)malloc(sizeof(double*) * n);
  rowsize = (int*)malloc(sizeof(int) * n);
  memcpy(dmt, dm, sizeof(double) * NCHOOSE2(n));

  info.X = X;
  info.m = m;
  info.n = n;
  info.nodes = nodes;
  info.ind = ind;
  info.dmt = dmt;
  info.buf = buf;
  info.rows = rows;
  info.rowsize = rowsize;
  info.dm = dm;
  info.centroids = centroids;
  if (kc) {
    info.centroidBuffer = centroids[2*n - 1];
  }
  else {
    info.centroidBuffer = 0;
  }
  info.lists = lists;
  for (i = 0; i < n; i++) {
    ind[i] = i;
    node = nodes + i;
    node->left = 0;
    node->right = 0;
    node->id = i;
    node->n = 1;
    node->d = 0.0;
    rowsize[i] = n - 1 - i;
  }
  rows[0] = dmt;
  for (i = 1; i < n; i++) {
    rows[i] = rows[i-1] + n - i;
  }
  
  if (ml) {
    for (i = 0; i < n; i++) {
      (lnodes + i)->val = nodes + i;
      (lnodes + i)->next = 0;
    }
  }

  for (k = 0, nid = n; k < n - 1; k++, nid++) {
    info.nid = nid;
    np = n - k;
    npc2 = NCHOOSE2(np);
    /**    fprintf(stderr, "k=%d, nid=%d, n=%d np=%d\n", k, nid, n, np);**/
    min = dmt[0];
    mini = 0;
    minj = 1;
    /** Note that mini < minj since j > i is always true. */
    /** BEGIN NEW CODE **/
    for (i = 0; i < np - 1; i++) {
      dmit = rows[i];
      for (j = i + 1; j < np; j++, dmit++) {
	if (*dmit < min) {
	  min = *dmit;
	  mini = i;
	  minj = j;
	}
      }
    }

    node = nodes + nid;
    node->left = nodes + ind[mini];
    node->right = nodes + ind[minj];
    ln = (double)node->left->n;
    rn = (double)node->right->n;
    qn = ln + rn;
    node->n = node->left->n + node->right->n;
    node->d = min;
    node->id = nid;

    Zrow = Z + (k * CPY_LIS);
    Zrow[CPY_LIN_LEFT] = node->left->id;
    Zrow[CPY_LIN_RIGHT] = node->right->id;
    Zrow[CPY_LIN_DIST] = min;
    Zrow[CPY_LIN_CNT] = node->n;

    /**    fprintf(stderr,
	    "[lid=%d, rid=%d, llid=%d, rrid=%d m=%5.8f]",
	    node->left->id, node->right->id, ind[mini], ind[minj], min);**/

    if (ml) {
      listC = GETCLUSTER(nid);
      if (ISCLUSTER(node->left) != 0) {
	listL = GETCLUSTER(node->left->id);
	if (ISCLUSTER(node->right) != 0) {
	  listR = GETCLUSTER(node->right->id);
	  listL->tail->next = listR->head;
	  listC->tail = listR->tail;
	  listR->tail->next = 0;
	}
	else {
	  listC->tail = lnodes + node->right->id;
	  listL->tail->next = listC->tail;
	  listC->tail->next = 0;
	}
	listC->head = listL->head;
      }
      else {
	listC->head = lnodes + node->left->id;
	if (ISCLUSTER(node->right)) {
	  listR = GETCLUSTER(node->right->id);
	  listC->head->next = listR->head;
	  listC->tail = listR->tail;
	  listC->tail->next = 0;
	}
	else {
	  listC->tail = lnodes + node->right->id;
	  listC->tail->next = 0;
	  listC->head->next = listC->tail;
	}
      }
    }
    if (kc) {
      centroidL = centroids[ind[mini]];
      centroidR = centroids[ind[minj]];
      centroid = centroids[nid];
      switch(method) {
      case CPY_LINKAGE_MEDIAN:
	for (t = 0; t < m; t++) {
	  centroid[t] = (centroidL[t] * 0.5 + centroidR[t] * 0.5);
	}
	break;
      case CPY_LINKAGE_CENTROID:
      case CPY_LINKAGE_WARD:
      default:
	for (t = 0; t < m; t++) {
	  centroid[t] = (centroidL[t] * ln + centroidR[t] * rn) / qn;
	}
	break;
      }
      /**      fprintf(stderr, "L: ");
      print_vec(centroidL, m);
      fprintf(stderr, "\nR: ");
      print_vec(centroidR, m);
      fprintf(stderr, "\nT: ");
      print_vec(centroid, m);**/
    }

    /**    print_dm(rows, np);**/
    /**    dfunc(buf, rows, mini, minj, np, dm, n, ind, nodes);**/
    dfunc(&info, mini, minj, np, n);

    /** For these rows, we must remove, i and j but leave all unused space
        at the end. This reduces their size by two.*/
    for (i = 0; i < minj; i++) {
      chopmins_ns_i(rows[i], minj - i - 1, rowsize[i]);
    }

    /** We skip the i'th row. For rows i+1 up to j-1, we just remove j. */
    /**for (i = mini + 1; i < minj; i++) {
      chopmins_ns_i(rows[i], minj - i - 1, rowsize[i]);
      }**/

    /** For rows 0 to mini - 1, we move them down the matrix, leaving the
	first row free. */
    /**for (i = mini; i > 0; i--) {
      memcpy(rows[i], rows[i-1], sizeof(double) * rowsize[i]-k);
    }

    for (i = mini; i < minj - 1; i++) {
      memcpy(rows[i], rows[i+1], sizeof(double) * (rowsize[i+1]));
      }**/

    /** For rows mini+1 to minj-1, we do nothing since they are in the
	right place for the next iteration. For rows minj+1 onward,
	we move them to the right. */
	
    for (i = minj; i < np - 1; i++) {
      memcpy(rows[i], rows[i+1], sizeof(double) * (rowsize[i+1]));
    }

    /** Rows i+1 to j-1 lose one unit of space, so we move them up. */
    /** Rows j to np-1 lose no space. We do nothing to them. */

    /**    memcpy(rows[0], buf, sizeof(double) * rowsize[0] - k);*/

    for (i = 0; i < mini; i++) {
      *(rows[i] + mini - i - 1) = buf[i];
    }

    for (i = mini + 1; i < np - 2; i++) {
      *(rows[mini] + i - mini - 1) = buf[i-1];
    }

    /**    print_dm(rows, np - 1);
	   print_ind(ind, np);**/
    chopmin(ind, minj, np);
    ind[mini] = nid;
    /**    print_ind(ind, np - 1);**/
  }
  free(lists);
  free(lnodes);
  free(nodes);
  free(ind);
  free(dmt);
  free(buf);
  free(rows);
  free(rowsize);
  free(centroidsData);
  free(centroids);
}

void dist_to_squareform_from_vector(double *M, const double *v, int n) {
  double *it;
  const double *cit;
  int i, j;
  cit = v;
  for (i = 0; i < n - 1; i++) {
    it = M + (i * n) + i + 1;
    for (j = i + 1; j < n; j++, it++, cit++) {
      *it = *cit;
    }
  }
}

void dist_to_vector_from_squareform(const double *M, double *v, int n) {
  double *it;
  const double *cit;
  int i, j;
  it = v;
  for (i = 0; i < n - 1; i++) {
    cit = M + (i * n) + i + 1;
    for (j = i + 1; j < n; j++, it++, cit++) {
      *it = *cit;
    }
  }
}

void cpy_to_tree(const double *Z, cnode **tnodes, int n) {
  const double *row;
  cnode *node;
  cnode *nodes;
  int i;
  nodes = (cnode*)malloc(sizeof(cnode) * (n * 2) - 1);
  *tnodes = nodes;
  for (i = 0; i < n; i++) {
    node = nodes + i;
    node->left = 0;
    node->right = 0;
    node->id = i;
    node->n = 1;
    node->d = 0.0;
  }
  for (i = 0; i < n - 1; i++) {
    node = nodes + i + n;
    row = Z + (i * CPY_LIS);
    node->id = i + n;
    node->left = nodes + (int)row[CPY_LIN_LEFT];
    node->right = nodes + (int)row[CPY_LIN_RIGHT];
    node->d = row[CPY_LIN_DIST];
    node->n = (int)row[CPY_LIN_CNT];
    /**    fprintf(stderr, "l: %d r: %d d: %5.5f n: %d\n", (int)row[0],
	   (int)row[1], row[2], (int)row[3]);**/
  }
}

inline void set_dist_entry(double *d, double val, int i, int j, int n) {
  if (i < j) {
    *(d + (NCHOOSE2(n)-NCHOOSE2(n - i)) + j) = val;
  }
  if (j < i) {
    *(d + (NCHOOSE2(n)-NCHOOSE2(n - j)) + i) = val;
  }
}

void cophenetic_discover(double *d, int n, const cnode *current, int *members) {
  const cnode *left = current->left;
  const cnode *right = current->right;
  int *membersRight;
  int ii, jj, i, j, k, ln, rn;
  int nc2 = NCHOOSE2(n);
  double dist;

  /** If leaf node. */
  if (current->id < n) {
    *members = current->id;
  }
  else {
    membersRight = members + left->n;
    cophenetic_discover(d, n, left, members);
    cophenetic_discover(d, n, right, membersRight);
    dist = current->d;
    ln = current->left->n;
    rn = current->right->n;
    for (ii = 0; ii < ln; ii++) {
      i = members[ii];
      for (jj = 0; jj < rn; jj++) {
	j = membersRight[jj];
	if (i < j) {
	  k = nc2 - NCHOOSE2(n - i) + (j - i - 1);
	}
	if (j < i) {
	  k = nc2 - NCHOOSE2(n - j) + (i - j - 1);
	}
	d[k] = dist;
	/**	fprintf(stderr, "i=%d j=%d k=%d d=%5.5f \n", i, j, k, dist);**/
      }
    }
  }

}

/** need non-recursive implementation. */
void cophenetic_distances(const double *Z, double *d, int n) {
  int *members = (int*)malloc(n * sizeof(int));
  cnode *nodes, *root;
  /**  fprintf(stderr, "copying into tree.\n");**/
  cpy_to_tree(Z, &nodes, n);
  /**  fprintf(stderr, "done copying into tree.\n");**/
  root = nodes + (n * 2) - 2; /** The root node is the 2*n-1'th node,
				  or the last node in the array.*/
  /**  fprintf(stderr, "begin discover.\n");**/
  cophenetic_discover(d, n, root, members);
  /**  fprintf(stderr, "end discover.\n");**/
  free(members);
  free(nodes);
}

void cophenetic_distances_nonrecursive(const double *Z, double *d, int n) {
  int *curNode, *left;
  int ndid, lid, rid, i, j, k, t, ln, rn, ii, jj, nc2;
  unsigned char *lvisited, *rvisited;
  const double *Zrow;
  int *members = (int*)malloc(n * sizeof(int));
  const int bff = CPY_FLAG_ARRAY_SIZE_BYTES(n);
  k = 0;
  curNode = (int*)malloc(n * sizeof(int));
  left = (int*)malloc(n * sizeof(int));
  lvisited = (unsigned char*)malloc(bff);
  rvisited = (unsigned char*)malloc(bff);
  curNode[k] = (n * 2) - 2;
  left[k] = 0;
  nc2 = NCHOOSE2(n);
  bzero(lvisited, bff);
  bzero(rvisited, bff);

  while (k >= 0) {
    ndid = curNode[k];
    Zrow = Z + ((ndid-n) * CPY_LIS);
    lid = (int)Zrow[CPY_LIN_LEFT];
    rid = (int)Zrow[CPY_LIN_RIGHT];
    if (lid >= n) {
      ln = (int)*(Z + (CPY_LIS * (lid-n)) + CPY_LIN_CNT);
    }
    else {
      ln = 1;
    }
    if (rid >= n) {
      rn = (int)*(Z + (CPY_LIS * (rid-n)) + CPY_LIN_CNT);
    }
    else {
      rn = 1;
    }
    
    /**    fprintf(stderr, "[fp] ndid=%d, ndid-n=%d, k=%d, lid=%d, rid=%d\n",
	   ndid, ndid-n, k, lid, rid);**/

    if (lid >= n && !CPY_GET_BIT(lvisited, ndid-n)) {
      CPY_SET_BIT(lvisited, ndid-n);
      curNode[k+1] = lid;
      left[k+1] = left[k];
      k++;
      continue;
    }
    else if (lid < n) {
      members[left[k]] = lid;
    }
    if (rid >= n && !CPY_GET_BIT(rvisited, ndid-n)) {
      CPY_SET_BIT(rvisited, ndid-n);
      curNode[k+1] = rid;
      left[k+1] = left[k] + ln;
      k++;
      continue;
    }
    else if (rid < n) {
      members[left[k]+ln] = rid;
    }

    /** If it's not a leaf node, and we've visited both children,
	record the final mean in the table. */
    if (ndid >= n) {
      for (ii = 0; ii < ln; ii++) {
	i = *(members + left[k] + ii);
	for (jj = 0; jj < rn; jj++) {
	  j = *(members + left[k] + ln + jj);
	  if (i < j) {
	    t = nc2 - NCHOOSE2(n - i) + (j - i - 1);
	  }
	  if (j < i) {
	    t = nc2 - NCHOOSE2(n - j) + (i - j - 1);
	  }
	  d[t] = Zrow[CPY_LIN_DIST];
	  /**	fprintf(stderr, "i=%d j=%d k=%d d=%5.5f \n", i, j, k, dist);**/
	}
      }
    }
    k--;
  }
  free(members);
  free(left);
  free(curNode);
  free(lvisited);
  free(rvisited);
}

void inconsistency_calculation_alt(const double *Z, double *R, int n, int d) {
  int *curNode;
  int ndid, lid, rid, i, k;
  unsigned char *lvisited, *rvisited;
  const double *Zrow;
  double *Rrow;
  double levelSum, levelStdSum;
  int levelCnt;
  const int bff = CPY_FLAG_ARRAY_SIZE_BYTES(n);
  k = 0;
  curNode = (int*)malloc(n * sizeof(int));
  lvisited = (unsigned char*)malloc(bff);
  rvisited = (unsigned char*)malloc(bff);
  /** for each node in the original linkage matrix. */
  for (i = 0; i < n - 1; i++) {
    /** the current depth j */
    k = 0;
    levelSum = 0.0;
    levelCnt = 0;
    levelStdSum = 0.0;
    bzero(lvisited, bff);
    bzero(rvisited, bff);
    curNode[0] = i;
    for (k = 0; k >= 0;) {
      ndid = curNode[k];
      Zrow = Z + ((ndid) * CPY_LIS);
      lid = (int)Zrow[CPY_LIN_LEFT];
      rid = (int)Zrow[CPY_LIN_RIGHT];
      /** fprintf(stderr, "[fp] ndid=%d, ndid-n=%d, k=%d, lid=%d, rid=%d\n",
	          ndid, ndid, k, lid, rid);**/
      if (k < d - 1) {
	if (lid >= n && !CPY_GET_BIT(lvisited, ndid)) {
	  CPY_SET_BIT(lvisited, ndid);
	  k++;
	  curNode[k] = lid-n;
	  continue;
	}
	if (rid >= n && !CPY_GET_BIT(rvisited, ndid)) {
	  CPY_SET_BIT(rvisited, ndid);
	  k++;
	  curNode[k] = rid-n;
	  continue;
	}
      }
      levelCnt++;
      levelSum += Zrow[CPY_LIN_DIST];
      levelStdSum += Zrow[CPY_LIN_DIST] * Zrow[CPY_LIN_DIST];
	/**fprintf(stderr, "  Using range %d to %d, levelCnt[k]=%d\n", lb, ub, levelCnt[k]);**/
      /** Let the count and sum slots be used for the next newly visited
	  node. */
      k--;
    }
    Rrow = R + (CPY_NIS * i);
    Rrow[CPY_INS_N] = (double)levelCnt;
    Rrow[CPY_INS_MEAN] = levelSum / levelCnt;
    if (levelCnt < 2) {
      Rrow[CPY_INS_STD] = (levelStdSum - (levelSum * levelSum)) / levelCnt;
    }
    else {
      Rrow[CPY_INS_STD] = (levelStdSum - ((levelSum * levelSum) / levelCnt)) / (levelCnt - 1);
    }
    Rrow[CPY_INS_STD] = sqrt(CPY_MAX(0, Rrow[CPY_INS_STD]));
    if (Rrow[CPY_INS_STD] > 0) {
      Rrow[CPY_INS_INS] = (Zrow[CPY_LIN_DIST] - Rrow[CPY_INS_MEAN]) / Rrow[CPY_INS_STD];
    }
  }
  
  free(curNode);
  free(lvisited);
  free(rvisited);
}

void calculate_cluster_sizes(const double *Z, double *CS, int n) {
  int i, j, k;
  const double *row;
  for (k = 0; k < n - 1; k++) {
    row = Z + (k * 3);
    i = (int)row[CPY_LIN_LEFT];
    j = (int)row[CPY_LIN_RIGHT];
    /** If the left node is a non-singleton, add its count. */
    if (i >= n) {
      CS[k] = CS[i - n];
    }
    /** Otherwise just add 1 for the leaf. */
    else {
      CS[k] = 1.0;
    }
    /** If the right node is a non-singleton, add its count. */
    if (j >= n) {
      CS[k] = CS[k] + CS[j - n];
    }
    /** Otherwise just add 1 for the leaf. */
    else {
      CS[k] = CS[k] + 1.0;
    }
    /**    fprintf(stderr, "i=%d, j=%d, CS[%d]=%d\n", i, j, n+k, (int)CS[k]);**/
  }
}

/** Returns an array of original observation indices (pre-order traversal). */
void form_member_list(const double *Z, int *members, int n) {
  int *curNode, *left;
  int ndid, lid, rid, k, ln, rn, nc2;
  unsigned char *lvisited, *rvisited;
  const double *Zrow;
  const int bff = CPY_FLAG_ARRAY_SIZE_BYTES(n);

  k = 0;
  curNode = (int*)malloc(n * sizeof(int));
  left = (int*)malloc(n * sizeof(int));
  lvisited = (unsigned char*)malloc(bff);
  rvisited = (unsigned char*)malloc(bff);
  curNode[k] = (n * 2) - 2;
  left[k] = 0;
  nc2 = NCHOOSE2(n);
  bzero(lvisited, bff);
  bzero(rvisited, bff);

  while (k >= 0) {
    ndid = curNode[k];
    Zrow = Z + ((ndid-n) * CPY_LIS);
    lid = (int)Zrow[CPY_LIN_LEFT];
    rid = (int)Zrow[CPY_LIN_RIGHT];
    if (lid >= n) {
      ln = (int)*(Z + (CPY_LIS * (lid-n)) + CPY_LIN_CNT);
    }
    else {
      ln = 1;
    }
    if (rid >= n) {
      rn = (int)*(Z + (CPY_LIS * (rid-n)) + CPY_LIN_CNT);
    }
    else {
      rn = 1;
    }
    
    /**    fprintf(stderr, "[fp] ndid=%d, ndid-n=%d, k=%d, lid=%d, rid=%d\n",
	   ndid, ndid-n, k, lid, rid);**/

    if (lid >= n && !CPY_GET_BIT(lvisited, ndid-n)) {
      CPY_SET_BIT(lvisited, ndid-n);
      curNode[k+1] = lid;
      left[k+1] = left[k];
      k++;
      continue;
    }
    else if (lid < n) {
      members[left[k]] = lid;
    }
    if (rid >= n && !CPY_GET_BIT(rvisited, ndid-n)) {
      CPY_SET_BIT(rvisited, ndid-n);
      curNode[k+1] = rid;
      left[k+1] = left[k] + ln;
      k++;
      continue;
    }
    else if (rid < n) {
      members[left[k]+ln] = rid;
    }
    k--;
  }
  free(left);
  free(curNode);
  free(lvisited);
  free(rvisited);
}

void form_flat_clusters_from_ic_alt(const double *Z, const double *R,
				    int *T, double cutoff, int n, int method) {
  int *curNode;
  int ndid, lid, rid, k, nc2, ms, nc;
  unsigned char *lvisited, *rvisited;
  const double *Zrow, *Rrow;
  double *maxsinconsist;
  double maxinconsist;
  const double * const *crit;
  int crit_off;
  const int bff = CPY_FLAG_ARRAY_SIZE_BYTES(n);

  k = 0;
  curNode = (int*)malloc(n * sizeof(int));
  lvisited = (unsigned char*)malloc(bff);
  rvisited = (unsigned char*)malloc(bff);
  maxsinconsist = (double*)malloc(n * sizeof(double));
  curNode[k] = (n * 2) - 2;
  nc2 = NCHOOSE2(n);
  bzero(lvisited, bff);
  bzero(rvisited, bff);
  /** number of clusters formed so far. */
  nc = 0;

  
  /** if method is distance. */
  if (method == CPY_CRIT_DISTANCE) {
    crit = &Zrow;
    crit_off = CPY_LIN_DIST;
  }
  else if (method == CPY_CRIT_INCONSISTENT) {
    crit = &Rrow;
    crit_off = CPY_INS_INS;
  }

  curNode[0] = 0;
  for (k = 0; k < n - 1; k++) {
    ndid = curNode[k];
    Zrow = Z + (ndid * CPY_LIS);
    Rrow = R + (ndid * CPY_NIS);
    lid = (int)Zrow[CPY_LIN_LEFT];
    rid = (int)Zrow[CPY_LIN_RIGHT];
    maxinconsist = Rrow[CPY_INS_INS];
    if (lid >= n && !CPY_GET_BIT(lvisited, ndid)) {
      maxinconsist = CPY_MAX(maxinconsist, maxinconsist);
      CPY_SET_BIT(lvisited, ndid);
      curNode[k+1] = lid - n;
      k++;
      continue;
    }
    if (rid >= n && !CPY_GET_BIT(rvisited, ndid)) {
      CPY_SET_BIT(lvisited, ndid);
      curNode[k+1] = rid-n;
      k++;
      continue;
    }
  }
  while (k >= 0) {
    ndid = curNode[k];
    Zrow = Z + ((ndid-n) * CPY_LIS);
    Rrow = R + ((ndid-n) * CPY_NIS);
    lid = (int)Zrow[CPY_LIN_LEFT];
    rid = (int)Zrow[CPY_LIN_RIGHT];
    maxinconsist = Rrow[CPY_INS_INS];
    /**    maxinconsist = *(*crit + crit_off);**/
    if (lid >= n && !CPY_GET_BIT(lvisited, ndid-n)) {
      CPY_SET_BIT(lvisited, ndid-n);
      curNode[k+1] = lid;
      k++;
      continue;
    }
    if (rid >= n && !CPY_GET_BIT(rvisited, ndid-n)) {
      CPY_SET_BIT(rvisited, ndid-n);
      curNode[k+1] = rid;
      k++;
      continue;
    }
    if (ndid >= n) {
      if (lid >= n) {
	maxinconsist = CPY_MAX(maxinconsist, maxsinconsist[lid-n]);
      }
      if (rid >= n) {
	maxinconsist = CPY_MAX(maxinconsist, maxsinconsist[rid-n]);
      }
      maxsinconsist[ndid-n] = maxinconsist;
    }
    k--;
  }
  k = 0;
  curNode[k] = (n * 2) - 2;
  bzero(lvisited, bff);
  bzero(rvisited, bff);
  ms = -1;
  while (k >= 0) {
    ndid = curNode[k];
    Zrow = Z + ((ndid-n) * CPY_LIS);
    Rrow = R + ((ndid-n) * CPY_NIS);
    lid = (int)Zrow[CPY_LIN_LEFT];
    rid = (int)Zrow[CPY_LIN_RIGHT];
    maxinconsist = maxsinconsist[ndid-n];
    fprintf(stderr, "cutoff: %5.5f maxi: %5.5f nc: %d\n", cutoff, maxinconsist, nc);
    if (ms == -1 && maxinconsist < cutoff) {
      ms = k;
      nc++;
    }
    if (lid >= n && !CPY_GET_BIT(lvisited, ndid-n)) {
      CPY_SET_BIT(lvisited, ndid-n);
      curNode[k+1] = lid;
      k++;
      continue;
    }
    if (rid >= n && !CPY_GET_BIT(rvisited, ndid-n)) {
      CPY_SET_BIT(rvisited, ndid-n);
      curNode[k+1] = rid;
      k++;
      continue;
    }
    if (ndid >= n) {
      if (lid < n) {
	if (ms == -1) {
	  T[lid] = ++nc;
	}
	else {
	  T[lid] = nc;
	}
      }
      if (rid < n) {
	if (ms == -1) {
	  T[rid] = ++nc;
	}
	else {
	  T[rid] = nc;
	}
      }
      if (ms == k) {
	ms = -1;
      }
    }
    k--;
  }

  free(maxsinconsist);
  free(curNode);
  free(lvisited);
  free(rvisited);  
}

/** form flat cluster from inconsistency coefficient.
    (don't need to assume monotonicity) */
void form_flat_clusters_from_ic(const double *Z, const double *R,
				int *T, double cutoff, int n, int method) {
  int *curNode;
  int ndid, lid, rid, k, nc2, ms, nc;
  unsigned char *lvisited, *rvisited;
  const double *Zrow, *Rrow;
  double *maxsinconsist;
  double maxinconsist;
  const double * const *crit;
  int crit_off;
  const int bff = CPY_FLAG_ARRAY_SIZE_BYTES(n);

  k = 0;
  curNode = (int*)malloc(n * sizeof(int));
  lvisited = (unsigned char*)malloc(bff);
  rvisited = (unsigned char*)malloc(bff);
  maxsinconsist = (double*)malloc(n * sizeof(double));
  curNode[k] = (n * 2) - 2;
  nc2 = NCHOOSE2(n);
  bzero(lvisited, bff);
  bzero(rvisited, bff);
  /** number of clusters formed so far. */
  nc = 0;
  ms = -1;
  /** if method is distance. */
  if (method == CPY_CRIT_DISTANCE) {
    crit = &Zrow;
    crit_off = CPY_LIN_DIST;
  }
  else if (method == CPY_CRIT_INCONSISTENT) {
    crit = &Rrow;
    crit_off = CPY_INS_INS;
  }
  while (k >= 0) {
    ndid = curNode[k];
    Zrow = Z + ((ndid-n) * CPY_LIS);
    Rrow = R + ((ndid-n) * CPY_NIS);
    lid = (int)Zrow[CPY_LIN_LEFT];
    rid = (int)Zrow[CPY_LIN_RIGHT];
    /**    maxinconsist = *(*crit + crit_off);**/
    if (lid >= n && !CPY_GET_BIT(lvisited, ndid-n)) {
      CPY_SET_BIT(lvisited, ndid-n);
      curNode[k+1] = lid;
      k++;
      continue;
    }
    if (rid >= n && !CPY_GET_BIT(rvisited, ndid-n)) {
      CPY_SET_BIT(rvisited, ndid-n);
      curNode[k+1] = rid;
      k++;
      continue;
    }
    maxinconsist = Rrow[CPY_INS_INS];
    if (lid >= n) {
      maxinconsist = CPY_MAX(maxinconsist, maxsinconsist[lid-n]);
    }
    if (rid >= n) {
      maxinconsist = CPY_MAX(maxinconsist, maxsinconsist[rid-n]);
    }
    maxsinconsist[ndid-n] = maxinconsist;
    k--;
  }
  k = 0;
  curNode[k] = (n * 2) - 2;
  bzero(lvisited, bff);
  bzero(rvisited, bff);
  ms = -1;
  while (k >= 0) {
    ndid = curNode[k];
    Zrow = Z + ((ndid-n) * CPY_LIS);
    Rrow = R + ((ndid-n) * CPY_NIS);
    lid = (int)Zrow[CPY_LIN_LEFT];
    rid = (int)Zrow[CPY_LIN_RIGHT];
    maxinconsist = maxsinconsist[ndid-n];
    fprintf(stderr, "cutoff: %5.5f maxi: %5.5f nc: %d\n", cutoff, maxinconsist, nc);
    if (ms == -1 && maxinconsist < cutoff) {
      ms = k;
      nc++;
    }
    if (lid >= n && !CPY_GET_BIT(lvisited, ndid-n)) {
      CPY_SET_BIT(lvisited, ndid-n);
      curNode[k+1] = lid;
      k++;
      continue;
    }
    if (rid >= n && !CPY_GET_BIT(rvisited, ndid-n)) {
      CPY_SET_BIT(rvisited, ndid-n);
      curNode[k+1] = rid;
      k++;
      continue;
    }
    if (ndid >= n) {
      if (lid < n) {
	if (ms == -1) {
	  T[lid] = ++nc;
	}
	else {
	  T[lid] = nc;
	}
      }
      if (rid < n) {
	if (ms == -1) {
	  T[rid] = ++nc;
	}
	else {
	  T[rid] = nc;
	}
      }
      if (ms == k) {
	ms = -1;
      }
    }
    k--;
  }

  free(maxsinconsist);
  free(curNode);
  free(lvisited);
  free(rvisited);  
}
